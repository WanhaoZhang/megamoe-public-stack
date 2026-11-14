"""CPU checks for per-call saved-state mode and autograd lifetime (no NPU)."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_workspace_layout import dispatch_output_size, training_saved_state_size


ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def contiguous(self):
        return self


class Context:
    def save_for_backward(self, *values):
        self.saved_tensors = values

    def mark_non_differentiable(self, *values):
        pass


class Operator:
    def __init__(self):
        self.calls = []

    def compute_train(self, *args):
        self.calls.append(("forward", args[-2:]))
        return Tensor(), Tensor(), Tensor(), Tensor(), Tensor()

    def compute_train_with_timer(self, *args):
        self.calls.append(("forward_timer", args[-2:]))
        return Tensor(), Tensor(), Tensor(), Tensor(), Tensor(), Tensor()

    def backward(self, *args):
        self.calls.append(("backward", args[-2:]))
        return (Tensor(),) * 4

    def backward_with_timer(self, *args):
        self.calls.append(("timer", args[-2:]))
        return (Tensor(),) * 5

    def backward_debug(self, *args):
        self.calls.append(("debug", args[-2:]))
        return (Tensor(),) * 5


def load_bridge():
    # Run the actual autograd bridge without importing the NPU extension.
    tree = ast.parse((ROOT / "ascend_deepep/__init__.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MegaMoeBF16TrainFunction")
    namespace = {"os": os, "torch": SimpleNamespace(autograd=SimpleNamespace(Function=object))}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "autograd_bridge", "exec"), namespace)
    return namespace[cls.name]


class SavedModeTests(unittest.TestCase):
    def test_allocation_drops_only_gate_up_extension(self):
        args = (4096, 7168, 16, 8, 4, 65536, 8192)
        self.assertEqual(training_saved_state_size(*args, save_gate_up=False), 940185600)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=False), 2013927424)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=True), 1493833728)
        self.assertEqual(
            training_saved_state_size(
                *args, save_mxfp8=False, embed_dispatch_output=False),
            1074272256)
        self.assertEqual(dispatch_output_size(7168, 65536), 939655168)

    def test_framework_can_release_dispatch_output(self):
        bridge = load_bridge()
        op = Operator()
        ctx = Context()
        bridge.forward(
            ctx, op, object(), *(Tensor() for _ in range(5)),
            65536, 0.0, "bf16", False, None)
        self.assertEqual(len(ctx.saved_tensors), 5)
        with self.assertRaisesRegex(RuntimeError, "retain_dispatch_output=True"):
            bridge.backward(ctx, Tensor())

    def test_interleaved_forwards_keep_modes_for_every_backward_entry(self):
        bridge = load_bridge()
        for entry, holder in (("backward", None), ("timer", {"capture_backward_timer": True}),
                              ("debug", {"capture_backward_workspace": True})):
            op = Operator()
            contexts = []
            modes = (("none", 0.0), ("bf16", 0.0), ("mxfp8", 0.0), ("auto", 10.0))
            with patch.dict(os.environ, {"ASCEND_DEEPEP_BWD_STREAM_SYNC": "0",
                                        "ASCEND_DEEPEP_BWD_SYNC": "0", "ASCEND_DEEPEP_DEBUG_PTR": "0"}):
                for mode, limit in modes:
                    ctx = Context()
                    bridge.forward(ctx, op, object(), *(Tensor() for _ in range(5)),
                                   65536, limit, mode, True,
                                   None if holder is None else holder.copy())
                    contexts.append(ctx)
                for ctx in reversed(contexts):
                    grads = bridge.backward(ctx, Tensor())
                    self.assertEqual(len(grads), 12)
                    self.assertEqual(grads[-3:], (None, None, None))
            self.assertEqual(op.calls[:4], [("forward", (limit, mode)) for mode, limit in modes])
            self.assertEqual(op.calls[4:], [(entry, (limit, mode)) for mode, limit in reversed(modes)])

    def test_forward_timer_is_opt_in_and_returned_through_holder(self):
        bridge = load_bridge()
        op = Operator()
        ctx = Context()
        holder = {"capture_forward_timer": True}
        bridge.forward(
            ctx, op, object(), *(Tensor() for _ in range(5)),
            65536, 0.0, "bf16", True, holder)
        self.assertEqual(op.calls, [("forward_timer", (0.0, "bf16"))])
        self.assertIsInstance(holder.get("forward_timer"), Tensor)

    def test_cli_modes_and_default(self):
        from test_mega_moe import parse_args
        self.assertEqual(parse_args(["0,1"]).save_gmm1_output, "bf16")
        for mode in ("none", "bf16", "mxfp8", "auto"):
            self.assertEqual(parse_args(["0,1", "--save-gmm1-output", mode]).save_gmm1_output, mode)


if __name__ == "__main__":
    unittest.main()
