"""CPU-only proxy tests: tape ownership, rebuilt Bwd inputs and audit boundary."""
from contextlib import nullcontext
import gc
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import shared_expert_redispatch_pool as adapter


class Payload:
    def __init__(self):
        self.calls = []

    def to(self, dtype):
        self.calls.append(('to', dtype))
        return self

    def contiguous(self):
        self.calls.append(('contiguous',))
        return self


class BasePool:
    home, slots, hidden, capacity = 2, 2, 7168, 20480
    group = object()
    marker = object()

    def __init__(self):
        self.counts, self.state = object(), object()
        self.placement = tuple(object() for _ in range(4))
        self.y, self.status, self.shared = object(), object(), object()
        self.backward = Mock(return_value=('routed-gradients',))
        self.backward_shared = Mock(return_value=('shared-gradients',))
        self.forward_modes = []

    def forward(self, *args):
        dispatch = Payload()
        self.dispatch_weakref = weakref.ref(dispatch)
        return self.y, self.status, (self.counts, self.state, dispatch, *self.placement)

    def forward_shared(self, *args, shared_save_mode='recompute'):
        self.forward_modes.append(shared_save_mode)
        return (*self.forward(*args), self.shared)


class RedispatchPoolTest(unittest.TestCase):
    def setUp(self):
        self.dist = ModuleType('torch.distributed')
        self.dist.get_world_size = Mock(return_value=8)
        self.dist.all_gather_object = Mock(side_effect=self.gather)
        self.torch = ModuleType('torch')
        self.torch.__path__ = []
        self.torch.bfloat16 = object()
        self.torch.distributed = self.dist
        timing = ModuleType('ascend_deepep.control_timing')
        timing.control_timing_scope = Mock(side_effect=lambda *args: nullcontext())
        parent = ModuleType('ascend_deepep')
        parent.__path__ = []
        self.modules = patch.dict(sys.modules, {
            'torch': self.torch, 'torch.distributed': self.dist,
            'ascend_deepep': parent, 'ascend_deepep.control_timing': timing})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.base = BasePool()
        self.proxy = adapter.BenchmarkRedispatchPool(self.base)
        self.x, self.prob = Payload(), Payload()
        self.buf = Payload()
        self.rebuild = patch.object(adapter, 'recompute_dispatch_segment', return_value=(self.buf, 37))
        self.recompute = self.rebuild.start()
        self.addCleanup(self.rebuild.stop)
        self.verifier_patch = patch.object(adapter, 'verify_dispatch_output', return_value=dict(
            pass_=True, actual_rows=37, token_prefix_sha256='tokens', probability_prefix_sha256='prob'))
        self.verifier = self.verifier_patch.start()
        self.addCleanup(self.verifier_patch.stop)

    @staticmethod
    def gather(outcomes, value):
        outcomes[:] = [value.copy() for _ in outcomes]

    def test_forward_tapes_original_reference_and_releases_forward_dispatch(self):
        y, status, saved = self.proxy.forward(self.x, 'ids', self.prob, 'w1', 'w2', 10.)
        self.assertIs(y,self.base.y)
        self.assertIs(status,self.base.status)
        self.assertIs(saved[2],self.x)
        self.assertEqual(saved[3:],self.base.placement)
        gc.collect()
        self.assertIsNone(self.base.dispatch_weakref())
        self.assertEqual(self.proxy.forward_dispatch_drops,1)
        self.assertIs(self.proxy.marker,self.base.marker)
        self.assertEqual(self.x.calls,[])
        self.recompute.assert_not_called()
        self.verifier.assert_not_called()
        self.dist.all_gather_object.assert_not_called()

    def test_both_backwards_consume_rebuilt_pack_and_cast_probabilities(self):
        placement = self.base.placement
        args = ('grad',self.prob,'w1','w2',self.base.status,placement,
                self.base.counts,self.base.state,self.x,10.)
        self.assertEqual(self.proxy.backward(*args),('routed-gradients',))
        forwarded = self.base.backward.call_args.args
        self.assertIs(forwarded[8],self.buf)
        self.assertIsNot(forwarded[8],self.x)
        for policy in ('keep','recompute'):
            self.assertEqual(self.proxy.backward_shared(*args,self.x,'wg','wu','wd',
                'rawg','rawu','activation',shared_save_mode=policy),('shared-gradients',))
            self.assertIs(self.base.backward_shared.call_args.args[8],self.buf)
            self.assertEqual(self.base.backward_shared.call_args.kwargs,
                             dict(shared_save_mode=policy, save_gmm1_output=None))
        self.assertEqual(self.proxy.backward_reconstructions,3)
        self.assertEqual(self.prob.calls,[('to',self.torch.bfloat16),('contiguous',)]*3)
        self.recompute.assert_called_with(self.x,placement[1],self.prob,
            local_experts=4,ep_group=self.base.group,max_recv=20480)
        self.verifier.assert_not_called()
        self.dist.all_gather_object.assert_not_called()

    def test_shared_forward_retains_policy_and_shared_state_without_private_x_clone(self):
        for policy in ('keep','recompute'):
            y,status,saved,shared=self.proxy.forward_shared(self.x,'ids',self.prob,
                'w1','w2','wg','wu','wd',10.,shared_save_mode=policy)
            self.assertIs(saved[2],self.x)
            self.assertIs(shared,self.base.shared)
            self.assertIs(y,self.base.y)
            self.assertIs(status,self.base.status)
            gc.collect()
            self.assertIsNone(self.base.dispatch_weakref())
        self.assertEqual(self.base.forward_modes,['keep','recompute'])
        self.assertEqual(self.proxy.forward_dispatch_drops,2)
        self.assertEqual(self.x.calls,[])

    def test_routed_backward_forwards_the_frozen_saved_format_to_base_pool(self):
        received = []
        def backward(*args, save_gmm1_output=None):
            self.assertIs(args[8], self.buf)
            received.append(save_gmm1_output)
            return ('routed-gradients',)
        self.base.backward = backward
        args = ('grad', self.prob, 'w1', 'w2', self.base.status, self.base.placement,
                self.base.counts, self.base.state, self.x, 10.)
        for mode in (None, 'bf16', 'none', 'mxfp8'):
            self.assertEqual(self.proxy.backward(*args, save_gmm1_output=mode),
                             ('routed-gradients',))
        self.assertEqual(received, [None, 'bf16', 'none', 'mxfp8'])
        self.assertEqual(self.proxy.backward_reconstructions, 4)

    def test_untimed_forward_audit_is_collective_and_disabled_for_measured_calls(self):
        self.proxy.audit_forward=True
        self.proxy.forward(self.x,'ids',self.prob,'w1','w2',10.)
        self.assertEqual(len(self.proxy.audits),1)
        self.assertEqual(self.proxy.audits[0]['mode'],'framework')
        self.assertTrue(self.proxy.audits[0]['pass_'])
        self.assertEqual(self.proxy.backward_reconstructions,0)
        self.dist.all_gather_object.assert_called_once()
        self.assertEqual(len(self.dist.all_gather_object.call_args.args[0]),8)
        self.verifier.assert_called_once_with(
            self.verifier.call_args.args[0],self.buf,self.base.counts,7168,20480)
        self.proxy.audit_forward=False
        self.recompute.reset_mock()
        self.verifier.reset_mock()
        self.dist.all_gather_object.reset_mock()
        self.proxy.forward(self.x,'ids',self.prob,'w1','w2',10.)
        self.recompute.assert_not_called()
        self.verifier.assert_not_called()
        self.dist.all_gather_object.assert_not_called()
        self.assertEqual(len(self.proxy.audits),1)

    def test_bad_prefix_or_remote_rank_failure_cannot_enter_native_backward(self):
        self.proxy.audit_forward=True
        self.verifier.side_effect=AssertionError('bad prefix')
        with self.assertRaisesRegex(AssertionError,'collective original-X'):
            self.proxy.forward(self.x,'ids',self.prob,'w1','w2',10.)
        self.base.backward.assert_not_called()
        self.assertEqual(self.proxy.forward_dispatch_drops,0)
        self.verifier.side_effect=None
        def remote_failure(outcomes,value):
            self.gather(outcomes,value)
            outcomes[7]=dict(pass_=False,mode='framework',error='other rank mismatch')
        self.dist.all_gather_object.side_effect=remote_failure
        with self.assertRaisesRegex(AssertionError,'other rank mismatch'):
            self.proxy.forward(self.x,'ids',self.prob,'w1','w2',10.)
        self.base.backward.assert_not_called()

    def test_receipt_states_external_reconstruction_not_kernel_redispatch(self):
        with patch.object(adapter,'reference_identity',return_value=dict(sha256='helper')):
            receipt=self.proxy.receipt()
        self.assertTrue(receipt['original_x_saved_as_reference'])
        self.assertFalse(receipt['external_reconstruction_in_both_modes'])
        self.assertEqual(receipt['external_reconstruction_modes'],[])
        self.proxy.forward(self.x,'ids',self.prob,'w1','w2',10.)
        self.assertEqual(self.proxy.receipt()['external_reconstruction_modes'],['framework'])
        self.proxy.forward_shared(self.x,'ids',self.prob,'w1','w2','wg','wu','wd',10.)
        self.assertTrue(self.proxy.receipt()['external_reconstruction_in_both_modes'])
        self.assertFalse(receipt['forward_dispatch_retained'])
        self.assertFalse(receipt['native_original_x_redispatch'])
        self.assertFalse(receipt['communication_overlap'])


if __name__ == '__main__':
    unittest.main()
