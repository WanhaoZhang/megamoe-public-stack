"""Real CPU autograd/tape tests; fake native math is delegation evidence only."""
from types import SimpleNamespace
import unittest
import weakref

import torch

from test_shared_expert_cpu import shared, bridge_module, FakePool, inputs


class NativeFakePool(FakePool):
    def forward_shared(self,*args,shared_save_mode='recompute',routed_input_mode='saved'):
        y,status,saved,planes=super().forward_shared(*args,shared_save_mode=shared_save_mode)
        self.input_mode=routed_input_mode
        if routed_input_mode=='redispatch':
            saved=(*saved[:2],torch.empty(0,dtype=torch.uint8),*saved[3:])
            planes=(*planes[:-1],torch.cat((planes[-1],torch.tensor([1],dtype=torch.int32))))
        return y,status,saved,planes

    def backward_shared(self,*args,shared_save_mode='recompute',routed_input_mode='saved',save_gmm1_output=None):
        self.backward_input_mode=routed_input_mode
        if routed_input_mode=='redispatch':
            assert args[8].numel()==0
        return super().backward_shared(*args,shared_save_mode=shared_save_mode,save_gmm1_output=save_gmm1_output)


class NativeXBridgeTests(unittest.TestCase):
    def test_default_tape_stays_v2_and_mode_validation(self):
        self.assertEqual(shared.validate_routed_input_mode('saved'),0)
        self.assertEqual(shared.validate_routed_input_mode('redispatch'),1)
        with self.assertRaises(ValueError):shared.validate_routed_input_mode('external')
        x,_,_,wg,_,_=inputs()
        raw=tuple(torch.zeros((len(x),256),dtype=x.dtype) for _ in range(3))
        old=shared.shared_saved_state(x,*raw,'keep')
        new=shared.shared_saved_state(x,*raw,'keep','redispatch')
        self.assertEqual(old[-1].tolist(),[1,1,3,256,256])
        self.assertEqual(new[-1].tolist(),[1,1,3,256,256,1])
        self.assertIs(new[0],x)

    def test_native_keep_and_recompute_autograd_maps_twelve_inputs(self):
        for policy in ('keep','recompute'):
            x,ids,prob,wg,wu,wd=inputs()
            for t in (x,prob,wg,wu,wd):t.requires_grad_()
            weights=tuple(torch.zeros(1,dtype=x.dtype,requires_grad=True) for _ in range(2))
            pool=NativeFakePool()
            bridge=bridge_module.FSDP2ExpertsBridge(pool,lambda:weights,lambda:(wg,wu,wd))
            y=bridge.forward_shared(x,ids,prob,wg,wu,wd,10,policy,routed_input_mode='redispatch')
            tape=y.grad_fn.saved_tensors
            self.assertEqual(tape[3].numel(),0)
            self.assertEqual(tape[-1].tolist()[-1],1)
            self.assertEqual([v.numel() for v in tape[10:13]],[3*256]*3 if policy=='keep' else [0]*3)
            y.sum().backward()
            self.assertEqual(pool.backward_input_mode,'redispatch')
            for t,value in zip((x,prob,*weights,wg,wu,wd),(5,7,11,13,17,19,23)):
                self.assertTrue(torch.equal(t.grad,torch.full_like(t,value)))
            self.assertFalse(pool.old_backward_called)

    def test_native_metadata_cannot_silently_fall_back_to_saved(self):
        x,ids,prob,wg,wu,wd=inputs();x.requires_grad_()
        weights=tuple(torch.zeros(1,dtype=x.dtype,requires_grad=True) for _ in range(2))
        pool=NativeFakePool();bridge=bridge_module.FSDP2ExpertsBridge(pool,lambda:weights,lambda:(wg,wu,wd))
        def pack(t):
            if t.dtype==torch.int32 and t.numel()==6:
                value=t.clone();value[-1]=0;return value
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            y=bridge.forward_shared(x,ids,prob,wg,wu,wd,routed_input_mode='redispatch')
        with self.assertRaisesRegex(RuntimeError,'saved-state metadata'):y.sum().backward()
        self.assertEqual(pool.backward_calls,[])

    def test_real_pool_drops_dispatch_and_rejects_old_worker(self):
        x,ids,prob,wg,wu,wd=inputs()
        pool=bridge_module.NativeFSDP2Pool.__new__(bridge_module.NativeFSDP2Pool)
        pool._save_gmm1_output='bf16'
        pool.busy=pool.failed=False
        pool.tokens,pool.hidden,pool.ffn,pool.topk=3,256,256,2
        pool.home=pool.slots=2;pool.capacity=8
        pool.stats={'forwards':0};pool.shared_timer_enabled=False
        pool.shmem=SimpleNamespace(world_size=lambda:4)
        pool.policy=object();pool._publish=lambda *a:None
        pool.x_redispatch_debug_sink=None
        status=(0,3,1,0,0,0,0,8)
        plan=SimpleNamespace(logical_ids=ids,physical_ids=ids,copies=torch.tensor([-1]),logical_counts=torch.tensor([3]),max_recv=8)
        class Op:
            calls=0
            backward_replica_shared_v3=lambda *a:None
            def compute_train_replica_shared(inner,*args):
                inner.calls+=1
                dispatch=torch.empty(4096,dtype=torch.uint8)
                inner.dispatch_ref=weakref.ref(dispatch)
                raw=tuple(torch.zeros((3,256),dtype=x.dtype) for _ in range(3))
                return x.clone(),torch.tensor([3]),torch.tensor([1]),dispatch,torch.empty(0),*raw,torch.empty(0),torch.empty(0)
        op=Op()
        caps=SimpleNamespace(megaflux_shared_expert_forward_abi=lambda:1)
        pool.module=SimpleNamespace(_worker_native=caps,op=op,w1=torch.empty(0),w2=torch.empty(0),
            create_plan=lambda *a:plan,_check_device_plan=lambda p:None,_descriptor=lambda p:None,finish=lambda:status)
        args=(x,ids,prob,torch.empty(0),torch.empty(0),wg,wu,wd,10)
        with self.assertRaisesRegex(RuntimeError,'ABI3'):pool.forward_shared(*args,routed_input_mode='redispatch')
        self.assertEqual(op.calls,0)
        caps.megaflux_shared_expert_backward_v3_version=lambda:3
        result=pool.forward_shared(*args,routed_input_mode='redispatch')
        self.assertEqual(result[2][2].numel(),0)
        self.assertIsNone(op.dispatch_ref())
        self.assertEqual(pool.stats['forward_dispatch_bytes_dropped'],4096)
        self.assertIs(result[3][0],x)


if __name__=='__main__':unittest.main()
