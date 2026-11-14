"""Saved-mode bridge contract checks using Python doubles, without torch/NPU.

These execute the production method bodies. They do not validate tensor math,
torch checkpoint replay, FSDP hooks or device performance.
"""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'saved_mode_cpu', ROOT / 'ascend_deepep/saved_gate_up_mode.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class Tensor:
    def __init__(self, shape=(), name='tensor', dtype='bf16'):
        self.shape, self.name, self.dtype = shape, name, dtype
        self.device = 'cpu-double'

    def contiguous(self): return self
    def detach(self): return self
    def clone(self): return Tensor(self.shape, self.name + '.clone', self.dtype)
    def to(self, *args): return self
    def copy_(self, other): return self
    def zero_(self): return self
    def data_ptr(self): return id(self)
    def __getitem__(self, key): return self


class Context:
    def save_for_backward(self, *values): self.saved_tensors = values
    def mark_non_differentiable(self, *values): pass


class Operator:
    def __init__(self): self.calls = []

    def compute_train_replica(self, *args):
        self.calls.append(('forward', args[-2]))
        return tuple(Tensor(name=n) for n in ('y', 'counts', 'state', 'dispatch', 'scratch'))

    def backward_replica(self, *args):
        self.calls.append(('backward', args[-2]))
        return tuple(Tensor(name=n) for n in ('dx', 'dp', 'dw1', 'dw2'))


identity = lambda f: f
torch_double = SimpleNamespace(
    autograd=SimpleNamespace(Function=object, function=SimpleNamespace(once_differentiable=identity)),
    bfloat16='bf16', int32='int32', int64='int64',
    is_tensor=lambda t: isinstance(t, Tensor), no_grad=contextlib.nullcontext,
    tensor=lambda *a, **kw: Tensor((8,), dtype=kw.get('dtype', 'int32')))
namespace = dict(torch=torch_double, once_differentiable=identity,
                 SimpleNamespace=SimpleNamespace, MegaFluxBF16MixedTrain=object,
                 validate_save_gmm1_output=policy.validate_save_gmm1_output,
                 resolve_save_gmm1_output=policy.resolve_save_gmm1_output,
                 MAGIC=1, PREFIX={'forward': 0, 'backward': 0}, HEADER=16, LANES=1,
                 layout=lambda *a: None)


def load_class(file, name):
    tree = ast.parse((ROOT / 'ascend_deepep' / file).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), file, 'exec'), namespace)
    return namespace[name]


Device = load_class('megaflux_device.py', 'MegaFluxBF16DeviceTrain')
Replica = load_class('megaflux_device.py', '_DeviceReplica')
namespace['_PLAN_TENSORS'] = ('logical_ids', 'physical_ids', 'copies', 'logical_counts')
load_class('fsdp2_bridge.py', '_SavedPlacementReplay')
External = load_class('fsdp2_bridge.py', '_ExternalExperts')
Pool = load_class('fsdp2_bridge.py', 'NativeFSDP2Pool')


def plan(ids=None):
    return SimpleNamespace(max_recv=20480, epoch=1,
        logical_ids=ids or Tensor((8192, 2)), physical_ids=Tensor((8192, 2)),
        copies=Tensor((4, 2)), logical_counts=Tensor((8,)),
        header=Tensor((8,), dtype='int32'), owner_counts=Tensor((2,)))


class Executor(Device):
    def __init__(self, mode):
        self._save_gmm1_output = mode
        self.shmem = SimpleNamespace(rank=lambda: 0, world_size=lambda: 4)
        self.home = self.slots = 2
        self.w1, self.w2 = Tensor((2, 6144, 7168)), Tensor((2, 7168, 3072))
        self.op = Operator()
        self.pipeline_trace = self.gradient_trace = None
        self._active = False
        self.create_plan = lambda ids, *a: plan(ids)
        self.release_pending = lambda: None

    def _check_device_plan(self, p): pass
    def _descriptor(self, p): return ['unchanged-descriptor']


def make_pool(mode):
    p = object.__new__(Pool)
    p._save_gmm1_output = mode
    p.module = Executor(mode)
    p.shmem, p.home, p.slots = p.module.shmem, 2, 2
    p.capacity, p.tokens, p.hidden, p.ffn, p.topk = 20480, 8192, 7168, 3072, 2
    p.policy = p.stats = p.audit = p.capture = None
    p.control_tape_device, p.backward_restore = 'npu', 'panel'
    p.busy = p.failed = False
    return p


class SavedModeContractTests(unittest.TestCase):
    def test_auto_matches_fp32_sentinel_and_explicit_modes_stay_explicit(self):
        for limit, resolved in ((0, 'mxfp8'), (10, 'bf16'),
                                (float.fromhex('0x1.fffffep+127'), 'mxfp8'),
                                (1e-50, 'bf16'), (1e99, 'bf16')):
            self.assertEqual(policy.resolve_save_gmm1_output('auto', limit), resolved)
        for mode in ('bf16', 'none', 'mxfp8'):
            self.assertEqual(policy.resolve_save_gmm1_output(mode, 10), mode)
        self.assertEqual(policy.saved_gate_up_metadata('mxfp8', 10)['layout_version'], 2)
        self.assertEqual(policy.saved_gate_up_metadata('auto', 0)['layout_version'], 1)
        with self.assertRaisesRegex(ValueError, 'save_gmm1_output'):
            policy.validate_save_gmm1_output('fp16')

    def test_all_forward_overrides_pass_the_selected_mode(self):
        variants = (
            ('megaflux_device.py', 'MegaFluxBF16DeviceTrain', 'MegaFluxBF16MixedTrain'),
            ('megaflux_pipeline.py', 'MegaFluxBF16PipelineTrain', 'MegaFluxBF16DeviceTrain'),
            ('megaflux_stream.py', 'MegaFluxBF16StreamTrain', 'MegaFluxBF16PipelineTrain'),
            ('megaflux_stripe.py', 'MegaFluxBF16StripeTrain', 'MegaFluxBF16PipelineTrain'),
            ('megaflux_full.py', 'MegaFluxBF16FullTrain', 'MegaFluxBF16StreamTrain'),
            ('megaflux_cached.py', '_CachedControl', None),
            ('megaflux_detail.py', 'MegaFluxBF16DetailTrain', 'MegaFluxBF16EarlyTrain'),
            ('megaflux_stage_timer.py', 'MegaFluxBF16StageTimerTrain', 'MegaFluxBF16EarlyTrain'))
        for file, name, base in variants:
            if base is not None: namespace[base] = object
            method = load_class(file, name)._forward_device
            for requested, limit in (('bf16', 10), ('none', 10), ('mxfp8', 10), ('auto', 0)):
                with self.subTest(file=file, mode=requested):
                    e, p = Executor(requested), plan()
                    e.stripe_ready = e.pipeline_ready = Tensor()
                    e.mode, e.MODES = 'w12_stripe', {'w12_stripe': 2}
                    e.stage_capacity = e.detail_capacity = p.max_recv
                    e.stage_layouts = {k: dict(total=0, version=6) for k in ('forward', 'backward')}
                    e.detail_specs = {k: dict(aic_capacity=0, aiv_capacity=0) for k in ('forward', 'backward')}
                    e.detail_enabled = False
                    if 'timer' in file or 'detail' in file:
                        e.pipeline_trace, e.gradient_trace = Tensor(), Tensor()
                    method(e, Tensor((8192, 7168)), Tensor((8192, 2)), p, limit)
                    resolved = policy.resolve_save_gmm1_output(requested, limit)
                    self.assertEqual(e.op.calls, [('forward', resolved)])
                    self.assertEqual(p._save_gmm1_output, resolved)

    def test_policy_is_validated_before_native_initialization_and_read_only(self):
        with self.assertRaisesRegex(ValueError, 'save_gmm1_output'):
            Pool(*([None] * 11), save_gmm1_output='fp16')
        with self.assertRaisesRegex(ValueError, 'save_gmm1_output'):
            Device(save_gmm1_output='fp16')
        for instance in (make_pool('none'), Executor('mxfp8')):
            with self.assertRaises(AttributeError):
                instance.save_gmm1_output = 'bf16'

    def test_device_autograd_uses_forward_mode_and_preserves_tensor_tape(self):
        for mode in ('bf16', 'none', 'mxfp8', 'auto'):
            e, p, ctx = Executor(mode), plan(), Context()
            Replica.forward(ctx, e, Tensor(), Tensor(), e.w1, e.w2, p, 0)
            expected = policy.resolve_save_gmm1_output(mode, 0)
            # The per-call context, not later module/plan metadata, is authoritative.
            p._save_gmm1_output = e.last_save_gmm1_output = 'different-call'
            Replica.backward(ctx, Tensor())
            self.assertEqual(e.op.calls, [('forward', expected), ('backward', expected)])
            self.assertEqual(len(ctx.saved_tensors), 6)

    def test_shared_pool_interleaved_auto_tapes_and_live_weights(self):
        pool = make_pool('auto')
        weights = [Tensor((2, 6144, 7168)), Tensor((2, 7168, 3072))]
        bridge = SimpleNamespace(pool=pool, current_weights=lambda: tuple(weights))
        contexts = []
        for limit in (0, 10):
            ctx = Context()
            External.forward(ctx, bridge, Tensor((8192, 7168)), Tensor((8192, 2)),
                             Tensor((8192, 2)), *weights, limit)
            self.assertFalse(any(t is w for t in ctx.saved_tensors for w in weights))
            self.assertEqual(len(ctx.saved_tensors), 9)
            contexts.append(ctx)
        weights[:] = [Tensor(w.shape, 'new-live-weight') for w in weights]
        for ctx in reversed(contexts): External.backward(ctx, Tensor())
        self.assertEqual(pool.module.op.calls, [('forward', 'mxfp8'), ('forward', 'bf16'),
                                               ('backward', 'bf16'), ('backward', 'mxfp8')])
        self.assertFalse(pool.busy or pool.failed)


if __name__ == '__main__':
    unittest.main()
