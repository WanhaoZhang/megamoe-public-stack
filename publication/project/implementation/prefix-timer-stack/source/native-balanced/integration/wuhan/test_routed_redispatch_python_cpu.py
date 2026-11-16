"""Autograd original-X tape contract; device transport is separately validated."""
import unittest
import weakref
import torch
from test_bridge_cpu import MathPool, load_bridge

bridge_module=load_bridge()

class RedispatchMathPool(MathPool):
    routed_input_mode='redispatch'
    def forward(self,*args):
        y,header,saved=super().forward(*args)
        self.packed_ref=weakref.ref(saved[2])
        return y,header,saved
    def backward(self,grad,prob,w1,w2,status,placement,counts,state,dispatch,limit,
                 *,save_gmm1_output='bf16',original_x=None):
        assert dispatch.numel()==0
        assert original_x is not None
        self.restored_x=original_x.detach().clone()
        return super().backward(grad,prob,w1,w2,status,placement,counts,state,
                               original_x,limit,save_gmm1_output=save_gmm1_output)

class RoutedTapeTests(unittest.TestCase):
    def test_drops_packed_input_and_preserves_original_x_with_real_autograd(self):
        torch.manual_seed(42)
        x=torch.randn(5,4,requires_grad=True)
        p=torch.full((5,1),0.5,requires_grad=True)
        w1=torch.randn(1,6,4,requires_grad=True)
        w2=torch.randn(1,4,3,requires_grad=True)
        ids=torch.zeros((5,1),dtype=torch.int32)
        pool=RedispatchMathPool()
        bridge=bridge_module.FSDP2ExpertsBridge(pool,lambda:(w1,w2))
        y=bridge(x,ids,p)
        tape=y.grad_fn.saved_tensors
        self.assertEqual(tape[3].numel(),0)
        self.assertEqual(tape[-1].data_ptr(),x.data_ptr())
        self.assertIsNone(pool.packed_ref())
        y.sum().backward()
        self.assertTrue(torch.equal(pool.restored_x,x.detach()))
        for t in (x,p,w1,w2):
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.isfinite(t.grad).all())
        refs=[t.detach().clone().requires_grad_() for t in (x,p,w1,w2)]
        from test_bridge_cpu import math_forward
        math_forward(*refs).sum().backward()
        for t,reference in zip((x,p,w1,w2),refs):
            torch.testing.assert_close(t.grad,reference.grad,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
