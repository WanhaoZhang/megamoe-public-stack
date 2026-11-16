"""CPU interface/save-contract tests, not native shared-kernel validation.

Run with a CPU-capable PyTorch interpreter. The fake pool below intentionally
does no expert GEMM; it verifies native delegation, lifecycle and failure paths.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch


PACKAGE = '_shared_expert_cpu_contract'
SOURCE = Path(__file__).resolve().parents[2] / 'ascend_deepep'
package = ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f'{PACKAGE}.{name}', SOURCE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = load('shared_expert')
bridge_module = load('fsdp2_bridge')


def inputs():
    x = torch.ones((3, 256), dtype=torch.bfloat16)
    ids = torch.zeros((3, 2), dtype=torch.int32)
    prob = torch.full((3, 2), .5)
    wg = torch.ones((256, 256), dtype=torch.bfloat16)
    return x, ids, prob, wg, wg.clone(), wg.clone()


class FakePool:
    def __init__(self):
        self.calls = []
        self.old_backward_called = False
        self.backward_calls = []

    def forward_shared(self, x, ids, prob, w1, w2, wg, wu, wd, limit,
                       shared_save_mode='recompute'):
        self.calls.append((w1, w2, shared_save_mode))
        # Sentinel output: the wrapper must return this exact native result,
        # rather than add or compute the shared branch again in Python.
        y = x + 7
        raw = tuple(torch.full((len(x), wg.shape[0]), n, dtype=x.dtype) for n in (1, 2, 3))
        epoch = len(self.calls)
        routed = (torch.tensor([1]), torch.tensor([epoch]), torch.tensor([3]),
                  ids.clone(), ids.clone(), torch.tensor([[-1]]), torch.tensor([len(x)]))
        status = (0, len(x), epoch, 0, 0, 0, 0, len(x))
        return y, status, routed, shared.shared_saved_state(x, *raw, shared_save_mode)

    def backward(self, *args):
        self.old_backward_called = True
        raise AssertionError('routed-only backward must never consume fused shared Y')

    def backward_shared(self, grad, prob, w1, w2, status, placement, counts,
                        state, dispatch, limit, x, wg, wu, wd, raw_gate,
                        raw_up, activation, shared_save_mode='recompute', *, save_gmm1_output=None):
        self.backward_calls.append(dict(status=status, state=state.clone(), weights=(w1, w2, wg, wu, wd),
                                        planes=(raw_gate, raw_up, activation), mode=shared_save_mode, routed_save=save_gmm1_output))
        return tuple(torch.full_like(value, n) for value, n in
                     zip((x, prob, w1, w2, wg, wu, wd), (5, 7, 11, 13, 17, 19, 23)))


class SharedContracts(unittest.TestCase):
    def test_format_capability_checks_loaded_forward_and_backward_independently(self):
        worker = SimpleNamespace(megaflux_shared_expert_forward_saved_modes=lambda: 7,
                                 megaflux_shared_expert_backward_saved_modes=lambda: 1)
        module = SimpleNamespace(_worker_native=worker)
        for mode in ('bf16', 'none', 'mxfp8'):
            shared.require_shared_saved_mode(module, mode)
        for mode in ('none', 'mxfp8'):
            with self.assertRaisesRegex(RuntimeError, 'matching rebuilt'):
                shared.require_shared_saved_mode(module, mode, backward=True)
        shared.require_shared_saved_mode(module, 'bf16', backward=True)
        legacy = SimpleNamespace(_worker_native=SimpleNamespace())
        shared.require_shared_saved_mode(legacy, 'bf16')
        with self.assertRaisesRegex(RuntimeError, 'matching rebuilt'):
            shared.require_shared_saved_mode(legacy, 'mxfp8')

    def test_each_forward_freezes_routed_format_for_backward(self):
        for requested, limit, resolved in (('bf16', 10., 'bf16'), ('none', 10., 'none'),
                                           ('mxfp8', 10., 'mxfp8'), ('auto', 10., 'bf16'),
                                           ('auto', 0., 'mxfp8')):
            for mode in ('keep', 'recompute'):
                with self.subTest(requested=requested, limit=limit, mode=mode):
                    x, ids, prob, wg, wu, wd = inputs()
                    x.requires_grad_()
                    pool = FakePool()
                    pool.save_gmm1_output = requested
                    bridge = bridge_module.FSDP2ExpertsBridge(
                        pool, lambda: (torch.zeros(1), torch.zeros(1)), lambda: (wg, wu, wd))
                    y = bridge.forward_shared(x, ids, prob, wg, wu, wd, limit, mode)
                    self.assertEqual(y.grad_fn.save_gmm1_output, resolved)
                    pool.save_gmm1_output = 'none' if resolved != 'none' else 'bf16'
                    y.sum().backward()
                    self.assertEqual(pool.backward_calls[0]['routed_save'], resolved)

    def test_saved_tensors_are_independent_and_x_is_not_cloned(self):
        x, _, _, wg, _, _ = inputs()
        raw = tuple(torch.ones((len(x), wg.shape[0]), dtype=x.dtype) for _ in range(3))
        kept = shared.shared_saved_state(x, *raw, 'keep')
        self.assertIs(kept[0], x)
        for actual, original in zip(kept[1:4], raw):
            self.assertIs(actual, original)
        self.assertEqual(kept[-1].device.type, 'cpu')
        self.assertEqual(kept[-1].dtype, torch.int32)
        self.assertEqual(kept[-1].tolist(), [1, 1, 3, 256, 256])
        recompute = shared.shared_saved_state(x, *raw, 'recompute')
        self.assertIs(recompute[0], x)
        self.assertTrue(all(value.numel() == 0 for value in recompute[1:4]))
        self.assertEqual(recompute[-1].tolist(), [1, 0, 3, 256, 256])

    def test_shape_dtype_and_mode_validation(self):
        x, _, _, wg, wu, wd = inputs()
        self.assertEqual(shared.validate_shared_inputs(x, wg, wu, wd, 10), (3, 256, 256))
        for changes in ((x.float(), wg, wu, wd, 10),
                        (x, wg.T, wu, wd, 10),
                        (x, wg, wu[:128], wd, 10),
                        (x, wg, wu, wd, -1),
                        (x, wg, wu, wd, float('nan'))):
            with self.assertRaises(ValueError):
                shared.validate_shared_inputs(*changes)
        with self.assertRaises(ValueError):
            shared.validate_shared_save_mode('drop')

    def test_no_grad_delegates_and_reads_live_routed_weights(self):
        x, ids, prob, wg, wu, wd = inputs()
        pool = FakePool()
        weights = [object(), object()]
        bridge = bridge_module.FSDP2ExpertsBridge(pool, lambda: tuple(weights))
        with torch.no_grad():
            y = bridge.forward_shared(x, ids, prob, wg, wu, wd, 10, 'keep')
            self.assertTrue(torch.equal(y, x + 7))
            weights[:] = [object(), object()]
            bridge.forward_shared(x, ids, prob, wg, wu, wd, 10)
        self.assertIs(pool.calls[-1][0], weights[0])
        self.assertEqual([call[2] for call in pool.calls], ['keep', 'recompute'])

    def test_backward_maps_seven_gradients_to_eleven_inputs_without_extra_dx(self):
        for mode in ('keep', 'recompute'):
            x, ids, prob, wg, wu, wd = inputs()
            for value in (x, prob, wg, wu, wd):
                value.requires_grad_()
            pool = FakePool()
            weights = tuple(torch.zeros(1, dtype=torch.bfloat16, requires_grad=True) for _ in range(2))
            bridge = bridge_module.FSDP2ExpertsBridge(pool, lambda: weights,
                                                     lambda: (wg, wu, wd))
            y = bridge.forward_shared(x, ids, prob, wg, wu, wd, 10, mode)
            self.assertTrue(y.requires_grad)
            y.sum().backward()
            for value, expected in zip((x, prob, *weights, wg, wu, wd), (5, 7, 11, 13, 17, 19, 23)):
                self.assertTrue(torch.equal(value.grad, torch.full_like(value, expected)))
            self.assertFalse(pool.old_backward_called)
            self.assertEqual(pool.backward_calls[0]['mode'], mode)
            self.assertEqual([plane.numel() for plane in pool.backward_calls[0]['planes']],
                             [3 * 256] * 3 if mode == 'keep' else [0] * 3)

    def test_grad_enabled_forward_requires_live_provider_before_launch(self):
        x, ids, prob, wg, wu, wd = inputs()
        wg.requires_grad_()
        pool = FakePool()
        bridge = bridge_module.FSDP2ExpertsBridge(pool, lambda: (torch.zeros(1), torch.zeros(1)))
        with self.assertRaisesRegex(RuntimeError, 'live shared-weight provider'):
            bridge.forward_shared(x, ids, prob, wg, wu, wd, 10.)
        self.assertFalse(pool.calls)

    def test_backward_reads_updated_weights_and_the_matching_forward_epoch(self):
        x, ids, prob, wg, wu, wd = inputs()
        x.requires_grad_()
        pool = FakePool()
        routed = [torch.zeros(1, dtype=x.dtype), torch.zeros(1, dtype=x.dtype)]
        first_owner = [wg, wu, wd]
        second_owner = [value.clone() for value in first_owner]
        bridge = bridge_module.FSDP2ExpertsBridge(pool, lambda: tuple(routed))
        first = bridge.forward_shared(x, ids, prob, *first_owner, 10., 'keep',
                                     shared_weight_provider=lambda: tuple(first_owner))
        second = bridge.forward_shared(x, ids, prob, *second_owner, 10., 'recompute',
                                      shared_weight_provider=lambda: tuple(second_owner))
        routed[:] = [value.clone() for value in routed]
        first_owner[:] = [value.clone() for value in first_owner]
        second.sum().backward()
        first.sum().backward()
        self.assertEqual([call['status'][2] for call in pool.backward_calls], [2, 1])
        self.assertEqual([call['state'].item() for call in pool.backward_calls], [2, 1])
        self.assertIs(pool.backward_calls[1]['weights'][2], first_owner[0])
        self.assertIs(pool.backward_calls[0]['weights'][2], second_owner[0])
        self.assertIs(pool.backward_calls[1]['weights'][0], routed[0])

    def test_backward_rejects_changed_unsharded_geometry(self):
        x, ids, prob, wg, wu, wd = inputs()
        x.requires_grad_()
        owner = [wg, wu, wd]
        pool = FakePool()
        bridge = bridge_module.FSDP2ExpertsBridge(pool, lambda: (torch.zeros(1), torch.zeros(1)),
                                                lambda: tuple(owner))
        y = bridge.forward_shared(x, ids, prob, *owner, 10.)
        owner[0] = wg[:128]
        with self.assertRaisesRegex(RuntimeError, 'original weight shape/dtype'):
            y.sum().backward()
        self.assertFalse(pool.backward_calls)

    def test_tape_keeps_matching_control_and_metadata_without_weight_tensors(self):
        for control_device in ('cpu', 'npu'):
            x, ids, prob, wg, wu, wd = inputs()
            wg.requires_grad_()
            pool = FakePool()
            pool.control_tape_device = control_device
            weights = [torch.zeros((2, 512, 256), dtype=x.dtype),
                       torch.zeros((2, 256, 256), dtype=x.dtype)]
            bridge = bridge_module.FSDP2ExpertsBridge(pool, lambda: tuple(weights), lambda: (wg, wu, wd))
            packed = []

            def pack(value):
                packed.append(value.data_ptr())
                return value.detach().clone()

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                first = bridge.forward_shared(x, ids, prob, wg, wu, wd, 10., 'keep')
                second = bridge.forward_shared(x, ids, prob, wg, wu, wd, 7., 'recompute')
            weight_pointers = {value.data_ptr() for value in (*weights, wg, wu, wd)}
            self.assertFalse(weight_pointers.intersection(packed))
            for expected_epoch, out, limit, mode in (
                    (1, first, 10., 'keep'), (2, second, 7., 'recompute')):
                node = out.grad_fn
                tape = node.saved_tensors
                self.assertEqual(len(tape), 14)
                self.assertIs(node.bridge, bridge)
                self.assertEqual(node.limit, limit)
                self.assertEqual(node.shared_save_mode, mode)
                self.assertEqual(node.prob_dtype, prob.dtype)
                self.assertEqual(node.weight_meta, tuple(
                    (tuple(value.shape), value.dtype) for value in (*weights, wg, wu, wd)))
                # counts/state/dispatch, the four placement tensors, then control.
                self.assertEqual(tape[2].item(), expected_epoch)
                self.assertEqual(tape[8].tolist(), [0, len(x), expected_epoch, 0, 0, 0, 0, len(x)])
                self.assertEqual(tape[8].dtype, torch.int32)
                self.assertEqual(tape[8].device, prob.device)
                self.assertEqual(tape[13].tolist(), [1, int(mode == 'keep'), 3, 256, 256])
                self.assertEqual(tape[10].numel(), 3 * 256 if mode == 'keep' else 0)
            # Metadata may outlive a shard, but the context must not pin it.
            old_shapes = first.grad_fn.weight_meta
            weights[:] = [value.clone() for value in weights]
            self.assertEqual(first.grad_fn.weight_meta, old_shapes)
            self.assertIs(first.grad_fn.bridge.current_weights()[0], weights[0])

    def test_missing_native_capability_fails_before_publication(self):
        x, ids, prob, wg, wu, wd = inputs()
        pool = object.__new__(bridge_module.NativeFSDP2Pool)
        pool._save_gmm1_output = 'bf16'
        pool.shmem = SimpleNamespace(world_size=lambda: 4)
        pool.module = SimpleNamespace(_worker_native=SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, 'shared-enabled S9 ABI1'):
            pool.forward_shared(x, ids, prob, None, None, wg, wu, wd, 10)


def shared_planes(x, wg, wu, wd, limit):
    gate, up = torch.nn.functional.linear(x, wg), torch.nn.functional.linear(x, wu)
    g, u = gate.float(), up.float()
    if limit > 0:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    activation = (torch.nn.functional.silu(g) * u).to(x.dtype)
    return gate, up, activation


class FakeNative:
    """Numerical CPU double, not a replacement for device acceptance.

    Recompute reconstructs only shared planes; the routed saved state and plan
    are consumed exactly once through the real NativeFSDP2Pool method.
    """
    def __init__(self):
        self.calls, self.recomputes = [], 0
        self.outputs = []
        self.fail = False

    def backward_replica_shared_v2(self, *args):
        self.calls.append(args)
        if self.fail:
            raise RuntimeError('native sentinel failure')
        (_, dy, prob, w1, w2, counts, state, dispatch, capacity, limit,
         saved_mode, descriptor, x, wg, wu, wd, gate, up, activation,
         shared_limit, recompute) = args
        if recompute:
            assert all(value.numel() == 0 for value in (gate, up, activation))
            gate, up, activation = shared_planes(x, wg, wu, wd, shared_limit)
            self.recomputes += 1
        g, u = gate.float(), up.float()
        mg, mu = torch.ones_like(g), torch.ones_like(u)
        if shared_limit > 0:
            mg, mu = (g <= shared_limit), (u >= -shared_limit) & (u <= shared_limit)
            g, u = g.clamp(max=shared_limit), u.clamp(-shared_limit, shared_limit)
        da = torch.nn.functional.linear(dy, wd.T).float()
        sigmoid = g.sigmoid()
        dg = (da * u * sigmoid * (1 + g * (1 - sigmoid)) * mg).to(x.dtype)
        du = (da * torch.nn.functional.silu(g) * mu).to(x.dtype)
        dx = (torch.nn.functional.linear(dg, wg.T).float() +
              torch.nn.functional.linear(du, wu.T).float()).to(x.dtype)
        # Sentinel routed-direct contribution is added exactly once by native.
        dx = (dx.float() + .25).to(x.dtype)
        dwg = (dg.float().T @ x.float()).to(wg.dtype)
        dwu = (du.float().T @ x.float()).to(wu.dtype)
        dwd = (dy.float().T @ activation.float()).to(wd.dtype)
        result = (dx, torch.full_like(prob, 3), torch.full_like(w1, 11),
                  torch.full_like(w2, 13), dwg, dwu, dwd)
        self.outputs.append(result)
        return result

    def backward_replica_shared_v2_with_timer(self, *args):
        return (*self.backward_replica_shared_v2(*args), torch.tensor([123], dtype=torch.int64),
                torch.ones((10, 128, 2), dtype=torch.int64))


class NativePoolContracts(unittest.TestCase):
    def make_pool(self):
        pool = object.__new__(bridge_module.NativeFSDP2Pool)
        pool._save_gmm1_output = 'bf16'
        pool.home, pool.slots, pool.capacity = 2, 2, 3
        pool.busy = pool.failed = pool.shared_timer_enabled = False
        pool.backward_restore = 'panel'
        pool.shmem = object()
        pool.stats = dict(backwards=0, backward_restores=0)
        native = FakeNative()
        module = SimpleNamespace(home=2, slots=2, _plan_owner=object(), epoch=40,
                                 w1=torch.zeros((4, 512, 256), dtype=torch.bfloat16),
                                 w2=torch.zeros((4, 256, 256), dtype=torch.bfloat16),
                                 _worker_native=SimpleNamespace(megaflux_shared_expert_backward_v2_version=lambda: 2),
                                 op=native, finish_count=0, trace_timers={})
        def create_plan(ids, replay, capacity):
            module.epoch += 1
            plan = replay.plan(SimpleNamespace(executor=module, logical_ids=ids,
                                              capacity=capacity, epoch=module.epoch))
            module.last_plan = plan
            return plan
        def finish():
            module.finish_count += 1
        module.create_plan, module.finish = create_plan, finish
        module._descriptor = lambda plan: ('new-epoch', plan.epoch)
        pool.module = module
        pool.published = []
        pool._publish = lambda w1, w2: pool.published.append((w1, w2))
        return pool, native

    def run_backward(self, pool, mode='keep'):
        torch.manual_seed(42)
        x, ids, prob, wg, wu, wd = inputs()
        x = torch.randn_like(x) * .01
        wg, wu, wd = (torch.randn_like(value) * .01 for value in (wg, wu, wd))
        planes = shared_planes(x, wg, wu, wd, 10.)
        if mode == 'recompute':
            planes = tuple(x.new_empty((0,)) for _ in range(3))
        w1, w2 = pool.module.w1[:2], pool.module.w2[:2]
        status = [0, 3, 7, 0, 0, 0, 0, 3]
        placement = (ids, ids.clone(), torch.tensor([[-1]]), torch.tensor([3]))
        result = pool.backward_shared(x + .5, prob, w1, w2, status, placement,
                                      torch.tensor([3]), torch.tensor([7]), x,
                                      10., x, wg, wu, wd, *planes, shared_save_mode=mode)
        return result

    def test_keep_and_native_shared_only_recompute_match_with_fresh_replay_epochs(self):
        pool, native = self.make_pool()
        kept = self.run_backward(pool, 'keep')
        recomputed = self.run_backward(pool, 'recompute')
        for actual, expected in zip(recomputed, kept):
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(native.recomputes, 1)
        self.assertEqual([call[11] for call in native.calls], [('new-epoch', 41), ('new-epoch', 42)])
        self.assertEqual([call[-1] for call in native.calls], [False, True])
        self.assertEqual(pool.module.last_plan.header[2].item(), 42)
        self.assertEqual(pool.module.finish_count, 2)
        self.assertEqual(pool.stats, dict(backwards=2, backward_restores=2))
        self.assertEqual(kept[2].shape[0], pool.home)
        # Native allocates fresh output storage per call. The owner prefix
        # keeps that storage alive; subsequent calls must not mutate it.
        self.assertEqual(kept[2].data_ptr(), native.outputs[0][2].data_ptr())
        self.assertEqual(kept[3].data_ptr(), native.outputs[0][3].data_ptr())
        self.assertIsNotNone(kept[2]._base)
        self.assertNotEqual(kept[2].data_ptr(), recomputed[2].data_ptr())
        self.assertTrue(torch.equal(kept[2], torch.full_like(kept[2], 11)))
        self.assertFalse(pool.busy or pool.failed)

    def test_timer_method_keeps_seven_gradient_contract_and_records_both_timers(self):
        pool, native = self.make_pool()
        pool.shared_timer_enabled = True
        result = self.run_backward(pool, 'recompute')
        self.assertEqual(len(result), 7)
        self.assertEqual(native.recomputes, 1)
        self.assertEqual(pool.module.trace_timers['backward'].tolist(), [123])
        self.assertEqual(tuple(pool.last_shared_backward_timer.shape), (10, 128, 2))

    def test_missing_v2_capability_fails_before_publish_or_plan(self):
        pool, _ = self.make_pool()
        pool.module._worker_native.megaflux_shared_expert_backward_v2_version = lambda: 0
        with self.assertRaisesRegex(RuntimeError, 'S9 V2'):
            self.run_backward(pool)
        self.assertFalse(pool.published)
        self.assertEqual(pool.module.epoch, 40)
        self.assertFalse(pool.busy or pool.failed)

    def test_native_failure_poisons_pool_without_finish(self):
        pool, native = self.make_pool()
        native.fail = True
        with self.assertRaisesRegex(RuntimeError, 'native sentinel failure'):
            self.run_backward(pool)
        self.assertFalse(pool.busy)
        self.assertTrue(pool.failed)
        self.assertEqual(pool.module.finish_count, 0)
        with self.assertRaisesRegex(RuntimeError, 'failed or concurrently'):
            self.run_backward(pool)


if __name__ == '__main__':
    unittest.main()
