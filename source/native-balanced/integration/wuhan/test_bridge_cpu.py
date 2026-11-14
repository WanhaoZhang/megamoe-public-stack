"""CPU lifecycle gate; this does not claim Ascend numerical/FSDP acceptance."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from torch.utils.checkpoint import checkpoint


def load_bridge():
    path = Path(__file__).resolve().parents[2] / 'ascend_deepep/fsdp2_bridge.py'
    # Load real package-relative helpers without importing the native/NPU
    # package initializer in this CPU-only lifecycle test.
    package = '_wuhan_bridge_lifecycle_cpu'
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(path.parent)]
        sys.modules[package] = module
    spec = importlib.util.spec_from_file_location(package+'.fsdp2_bridge', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def math_forward(x, p, w1, w2):
    gate, up = (x @ w1[0].T).chunk(2, -1)
    return (torch.nn.functional.silu(gate) * up) @ w2[0].T * p


class MathPool:
    """Overwrite shared workspace between layers, as the real serial pool does."""
    def __init__(self):
        self.calls = 0
        self.workspace = torch.zeros(16)

    def forward(self, x, ids, p, w1, w2, limit):
        self.calls += 1
        self.workspace.fill_(self.calls)
        # Per-call snapshots, independently owned; no weight Tensor is saved.
        placement = (ids.clone(), ids.clone(), torch.tensor([[-1]]), torch.tensor([len(x)]))
        # Match the production pool's hook-visible device header, not the
        # legacy host tuple accepted only by explicit placement replay.
        status = torch.tensor((0, len(x), self.calls, 0, 0, 0, 0, len(x)),
                              dtype=torch.int32, device=x.device)
        saved = (torch.tensor([len(x)]), self.workspace.clone(), x.clone(), *placement)
        return math_forward(x, p, w1, w2), status, saved

    def backward(self, grad, prob, w1, w2, status, placement, counts, state, dispatch, limit,
                 *, save_gmm1_output='bf16'):
        assert save_gmm1_output == 'bf16'
        assert tuple(status.shape) == (8,) and status.dtype == torch.int32
        assert state[0].item() == status[2].item(), 'another layer overwrote saved state'
        assert torch.equal(placement[0], placement[1]), 'placement changed on tape'
        with torch.enable_grad():
            args = [t.detach().requires_grad_() for t in (dispatch, prob, w1, w2)]
            return torch.autograd.grad(math_forward(*args), args, grad)


def run_case(checkpointed, packed, capture=False, control_device='npu'):
    bridge_module = load_bridge()
    torch.manual_seed(42)
    pool = MathPool()
    pool.control_tape_device = control_device
    captured = []
    if capture:
        from types import SimpleNamespace
        def save(pool, x, y, dy, prob, w1, w2, status, placement, result, limit):
            if pool.capture.written:
                return
            torch.testing.assert_close(y, math_forward(x, prob, w1, w2))
            assert len(result) == 4 and placement[0].shape == ids.shape
            captured.append(True)
            pool.capture.written = True
        pool.capture = SimpleNamespace(written=False, save=save)
    weights = [torch.randn(1, 6, 4, requires_grad=True), torch.randn(1, 4, 3, requires_grad=True),
               torch.randn(1, 6, 4, requires_grad=True), torch.randn(1, 4, 3, requires_grad=True)]
    x = torch.randn(5, 4, requires_grad=True)
    prob = torch.rand(5, 1, requires_grad=True)
    ids = torch.zeros(5, 1, dtype=torch.int32)
    all_inputs = [x, prob, *weights]
    expected = [t.detach().clone().requires_grad_() for t in all_inputs]
    seen = []
    def pack(t):
        seen.append(t.data_ptr())
        return t.detach().clone()
    hooks = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t) if packed else contextlib.nullcontext()
    bridges = [bridge_module.FSDP2ExpertsBridge(pool, lambda i=i: tuple(weights[i:i+2])) for i in (0, 2)]
    # A second optimizer iteration verifies that backward reads new live values.
    for step in range(2):
        with hooks:
            h = x
            for b in bridges:
                def call(v, b=b):
                    return b(v, ids, prob)
                h = checkpoint(call, h, use_reentrant=False) if checkpointed else call(h)
            loss = h.square().sum()
        if packed and not checkpointed:
            assert not ({w.data_ptr() for w in weights} & set(seen)), 'pooled weights pinned on tape'
        loss.backward()
        ref = expected[0]
        for i in (2, 4):
            ref = math_forward(ref, expected[1], expected[i], expected[i+1])
        ref.square().sum().backward()
        torch.testing.assert_close(h, ref, rtol=1e-6, atol=1e-6)
        for actual, golden in zip(all_inputs, expected):
            torch.testing.assert_close(actual.grad, golden.grad, rtol=1e-6, atol=1e-6)
        with torch.no_grad():
            for actual, golden in zip(all_inputs, expected):
                actual.sub_(actual.grad * 1e-4)
                golden.sub_(golden.grad * 1e-4)
                actual.grad = golden.grad = None
    assert not capture or len(captured) == 1
    return dict(checkpoint=checkpointed, saved_tensor_pack=packed, capture=capture,
                control_tape_device=control_device, passed=True, pool_forwards=pool.calls)


if __name__ == '__main__':
    from types import SimpleNamespace
    bridge = load_bridge()
    for mode in ('staged', 'expert'):
        bridge._require_panel_ready(mode, SimpleNamespace())
    bridge._require_panel_ready('panel', SimpleNamespace(restore_operand_ready=1))
    for old_library in (SimpleNamespace(), SimpleNamespace(restore_operand_ready=0)):
        try:
            bridge._require_panel_ready('panel', old_library)
        except RuntimeError as error:
            assert 'rebuild' in str(error)
        else:
            raise AssertionError('old first-panel-only SO accepted')
    results = [run_case(c, p, a, d) for c in (False, True) for p in (False, True)
               for a in (False, True) for d in ('npu', 'cpu')]
    print(json.dumps(dict(kind='cpu-math-lifecycle-only', cases=results), indent=2))
