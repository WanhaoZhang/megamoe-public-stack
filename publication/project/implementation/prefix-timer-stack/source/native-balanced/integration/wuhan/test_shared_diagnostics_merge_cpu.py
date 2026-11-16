"""Real pool control flow with fake tensor/native objects; no numerical/NPU claim.

Exercises the merge between optional diagnostics and shared saved/redispatch
paths without importing PyTorch or a native library.
"""
from contextlib import nullcontext
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class Tensor:
    def __init__(self, shape, dtype='bf16', data=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, 'cpu'
        self.data = list(data) if data is not None else None

    def detach(self): return self
    def contiguous(self): return self
    def is_contiguous(self): return True
    def numel(self): return math.prod(self.shape)
    def to(self, *args): return self
    def clone(self): return Tensor(self.shape, self.dtype, self.data)
    def copy_(self, source): return self
    def __getitem__(self, key): return self.data[key] if self.data is not None else self
    def __setitem__(self, key, value): self.data[key] = value


class SharedDiagnosticsMerge(unittest.TestCase):
    def setUp(self):
        torch = ModuleType('torch')
        torch.bfloat16, torch.uint8, torch.int32 = 'bf16', 'u8', 'i32'
        torch.autograd = SimpleNamespace(Function=object,
            function=SimpleNamespace(once_differentiable=lambda function: function))
        torch.no_grad = nullcontext
        torch.is_tensor = lambda value: isinstance(value, Tensor)
        torch.empty = lambda n, dtype, device: Tensor((n,), dtype)
        torch.tensor = lambda data, dtype, device: Tensor((len(data),), dtype, data)
        package = '_shared_merge_cpu'
        parent = ModuleType(package)
        parent.__path__ = [str(HERE.parents[1]/'ascend_deepep')]
        timing = ModuleType(package+'.control_timing')
        timing.control_timing_scope = lambda *args: nullcontext()
        shared = ModuleType(package+'.shared_expert')
        shared.SHARED_FORWARD_ABI = 1
        shared.SHARED_BACKWARD_ABI = 2
        shared.SHARED_REDISPATCH_BACKWARD_ABI = 3
        shared.validate_shared_inputs = lambda *args: (3, 256, 256)
        shared.validate_shared_save_mode = lambda mode: int(mode == 'keep')
        shared.validate_routed_input_mode = lambda mode: int(mode == 'redispatch')
        shared.shared_saved_state = lambda *args: args
        shared.prepare_shared_timer_capture = lambda *args: None
        planners = ModuleType(package+'.megaflux_planners')
        planners._CHECKED_PLAN = object()
        self.modules = patch.dict(sys.modules, {'torch':torch, package:parent,
            package+'.control_timing':timing, package+'.shared_expert':shared,
            package+'.megaflux_planners':planners})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(package+'.fsdp2_bridge',
            HERE.parents[1]/'ascend_deepep/fsdp2_bridge.py')
        self.bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bridge)

    def exercise(self, diagnostics, routed_mode, save_mode):
        pool = self.bridge.NativeFSDP2Pool.__new__(self.bridge.NativeFSDP2Pool)
        pool._save_gmm1_output = 'bf16'
        pool.busy = pool.failed = False
        pool.home = pool.slots = 2
        pool.tokens, pool.hidden, pool.ffn, pool.topk, pool.capacity = 3, 256, 256, 2, 8
        pool.backward_restore = 'panel'
        pool.stats = dict(forwards=0, backwards=0, home_refresh_bytes=0, backward_restores=0) if diagnostics else None
        pool.shared_timer_enabled = False
        pool.shmem = SimpleNamespace(world_size=lambda:4)
        pool.policy = object()
        debug = []
        pool.x_redispatch_debug_sink = lambda *args: debug.append(args[0])
        x, ids, prob = Tensor((3,256)), Tensor((3,2),'i32'), Tensor((3,2))
        w1, w2 = Tensor((2,512,256)), Tensor((2,256,256))
        weights = (Tensor((256,256)),)*3
        status = [0,3,1,0,0,0,0,8]
        plan = SimpleNamespace(logical_ids=ids, physical_ids=ids, copies=Tensor((4,2),'i32'),
            logical_counts=Tensor((8,),'i32'), max_recv=8)
        calls = []
        def forward(*args):
            calls.append('forward')
            raw = tuple(Tensor((3,256)) for _ in range(3))
            return x,Tensor((4,),'i32'),Tensor((8,)),Tensor((4096,),'u8'),Tensor((0,)),*raw,Tensor((0,)),Tensor((0,))
        def backward(*args):
            calls.append('backward')
            return (Tensor((3,256)),Tensor((3,2)),w1,w2,*weights)
        def backward_redispatch(*args):
            return (*backward(*args),Tensor((4096,),'u8'))
        native = SimpleNamespace(compute_train_replica_shared=forward,
            backward_replica_shared_v2=backward, backward_replica_shared_v3=backward_redispatch)
        caps = SimpleNamespace(megaflux_shared_expert_forward_abi=lambda:1,
            megaflux_shared_expert_backward_v2_version=lambda:2,
            megaflux_shared_expert_backward_v3_version=lambda:3)
        pool.module = SimpleNamespace(w1=w1,w2=w2,op=native,_worker_native=caps,
            create_plan=lambda *args:plan,_check_device_plan=lambda *args:None,
            _descriptor=lambda *args:None,finish=lambda:status,trace_timers={})
        y, actual_status, saved, _ = pool.forward_shared(x,ids,prob,w1,w2,*weights,10,
            shared_save_mode=save_mode,routed_input_mode=routed_mode)
        self.assertIs(y,x)
        self.assertIs(actual_status,status)
        self.assertEqual(saved[2].numel(),0 if routed_mode=='redispatch' else 4096)
        planes = tuple(Tensor((3,256)) if save_mode=='keep' else Tensor((0,)) for _ in range(3))
        result = pool.backward_shared(x,prob,w1,w2,status,saved[3:],saved[0],saved[1],saved[2],10,
            x,*weights,*planes,shared_save_mode=save_mode,routed_input_mode=routed_mode)
        self.assertEqual(len(result),7)
        self.assertEqual(calls,['forward','backward'])
        self.assertFalse(pool.failed or pool.busy)
        self.assertEqual(debug,['forward','backward'] if routed_mode=='redispatch' else [])
        if diagnostics:
            self.assertEqual([pool.stats[key] for key in ('forwards','backwards','backward_restores')],[1,1,1])
            self.assertEqual(pool.stats['home_refresh_bytes'],2*(w1.numel()+w2.numel())*2)
            if routed_mode=='redispatch':
                self.assertEqual(pool.stats['forward_dispatch_bytes_dropped'],4096)
                self.assertEqual(pool.stats['native_x_redispatch_backwards'],1)
        else:
            self.assertIsNone(pool.stats)

    def test_shared_paths_with_diagnostics_on_and_off(self):
        for diagnostics in (False,True):
            for mode in ('saved','redispatch'):
                for save in ('keep','recompute'):
                    with self.subTest(diagnostics=diagnostics,mode=mode,save=save):
                        self.exercise(diagnostics,mode,save)

    def test_device_header_replay_clones_and_updates_only_new_epoch(self):
        pool = SimpleNamespace(home=2,slots=2,_plan_owner=object())
        ids = Tensor((3,2),'i32')
        header = Tensor((8,),'i32',[0,3,11,0,0,0,0,8])
        replay = self.bridge._SavedPlacementReplay(2,2,8,header,(ids,ids,ids,ids))
        result = replay.plan(SimpleNamespace(executor=pool,capacity=8,logical_ids=ids,epoch=19))
        self.assertEqual(header.data,[0,3,11,0,0,0,0,8])
        self.assertEqual(result.header.data,[0,3,19,0,0,0,0,8])


if __name__ == '__main__':
    unittest.main()
