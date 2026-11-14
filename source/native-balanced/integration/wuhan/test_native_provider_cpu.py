"""Native mode keeps borrowed buffers and avoids audit-only allocations."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch, torch_npu
from ascend_deepep.execution_plan_device import DevicePlanMetadata, normalize_validation
from ascend_deepep.moonep_planner import MoonEPPlanner
from ascend_deepep.megaflux_planners import _CHECKED_PLAN

class Contract(unittest.TestCase):
    def test_metadata_has_no_device_scratch(self):
        def peers(out,signature,group=None): out[:]=[signature]*4
        with patch('torch.distributed.get_world_size',return_value=4), \
             patch('torch.distributed.get_rank',return_value=0), \
             patch('torch.distributed.get_backend',return_value='hccl'), \
             patch('torch.distributed.all_gather_object',side_effect=peers), \
             patch('torch.empty',side_effect=AssertionError('audit allocation')):
            metadata=DevicePlanMetadata(home=2,slots=2,tokens=257,topk=2,device='npu:0')
        self.assertFalse(hasattr(metadata,'scratch'))
        self.assertEqual(normalize_validation('native'),'native')
        self.assertEqual(normalize_validation('fused'),'fused')

    def test_native_plan_borrows_outputs(self):
        policy=MoonEPPlanner.__new__(MoonEPPlanner)
        policy.validation='native';policy._executor=None;policy.timer_backing=None
        policy.diagnostics=False
        policy.validator=SimpleNamespace(home=2,slots=2,rank=0,ep=4,tokens=257,topk=2,device=torch.device('cpu'))
        policy.physical=torch.empty((257,2),dtype=torch.int32)
        policy.copies=torch.empty((4,2),dtype=torch.int32)
        policy.logical=torch.empty(8,dtype=torch.int32)
        policy.assignment=torch.empty((8,4),dtype=torch.int32)
        policy.native_header=torch.empty(8,dtype=torch.int32)
        workspace=SimpleNamespace(online_plan_moonep=lambda *args:None)
        executor=SimpleNamespace(home=2,slots=2,workspace=workspace,_plan_owner=object(),
            shmem=SimpleNamespace(rank=lambda:0,world_size=lambda:4))
        ids=torch.zeros((257,2),dtype=torch.int32)
        context=SimpleNamespace(executor=executor,logical_ids=ids,epoch=7,capacity=20480)
        with patch.object(torch.Tensor,'clone',side_effect=AssertionError('duplicate snapshot')):
            result=policy.plan(context)
        self.assertIs(result.logical_ids,ids)
        for key,source in [('physical_ids','physical'),('copies','copies'),('logical_counts','logical'),('header','native_header')]:
            self.assertIs(getattr(result,key),getattr(policy,source))
        self.assertIs(result._certificate,_CHECKED_PLAN)
        self.assertEqual((result.epoch,result.max_recv),(7,20480))
        context.executor=SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError,'one executor'):policy.plan(context)

if __name__=='__main__': unittest.main()
