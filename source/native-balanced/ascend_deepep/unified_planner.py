"""Fixed-owner policy registry and the common online compiler/validation path.

Only this adapter binds the policy to runtime services. Neither a policy nor
its immutable body owns model parameters, communication epochs or stream state.
"""
from dataclasses import dataclass, field
import ctypes
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from .execution_plan_device import DevicePlanValidator
from .megaflux_planners import _CHECKED_PLAN
from .control_timing import control_timing_scope
from .runtime_limits import LOCAL_EXPERT_CAPACITY


@dataclass(frozen=True)
class PolicyDefinition:
    name: str
    code: int
    source_mode: str


POLICIES = {x.name:x for x in (
    PolicyDefinition('moonep',0,'global_interval'),
    PolicyDefinition('native',1,'global_interval'),
    PolicyDefinition('ultra_ep',2,'local_transportation'),
    PolicyDefinition('echo',3,'breadth_then_depth'),
    PolicyDefinition('eplb_fixed',4,'global_interval'),
    PolicyDefinition('step_time',5,'causal_cost_search'))}


@dataclass(frozen=True)
class DeviceProposal:
    copies: object
    quota: object


def _identity(tensor):
    return (tensor.data_ptr(),tensor._version,tuple(tensor.shape),tensor.dtype,tensor.device)


@dataclass(frozen=True)
class ExecutionPlanBody:
    logical_ids: object
    physical_ids: object
    copies: object
    logical_counts: object
    source_quota: object
    control_template: object
    home: int
    slots: int
    capacity: int
    _identities: tuple = field(init=False,repr=False)

    def __post_init__(self):
        object.__setattr__(self,'_identities',tuple(_identity(t) for t in self.tensors()))

    def tensors(self):
        return (self.logical_ids,self.physical_ids,self.copies,self.logical_counts,
                self.source_quota,self.control_template)

    def check(self):
        if tuple(_identity(t) for t in self.tensors())!=self._identities:
            raise RuntimeError('immutable execution plan body was mutated')

    @classmethod
    def freeze(cls,plan,quota,home,slots,capacity):
        values=[getattr(plan,n).detach().clone() for n in
                ('logical_ids','physical_ids','copies','logical_counts')]
        return cls(*values,quota.detach().clone(),plan.header.detach().clone(),home,slots,capacity)

    def plan(self,context):
        self.check()
        m=context.executor
        if (m.home,m.slots,context.capacity)!=(self.home,self.slots,self.capacity):
            raise RuntimeError('plan body/lease layout or receive capacity mismatch')
        if context.logical_ids is not self.logical_ids:
            raise RuntimeError('rebind must use the frozen logical assignments')
        header=self.control_template.clone();header[2]=int(context.epoch)
        return SimpleNamespace(reference=SimpleNamespace(home=self.home,slots=self.slots),
            logical_ids=self.logical_ids,physical_ids=self.physical_ids,copies=self.copies,
            logical_counts=self.logical_counts,source_quota=self.source_quota,header=header,
            epoch=context.epoch,owner=m._plan_owner,max_recv=self.capacity,
            body=self,_certificate=_CHECKED_PLAN)


class CommonOnlinePlanner:
    def __init__(self,*,name,home,slots,tokens,topk,device,group=None,cost_model=None):
        if name not in POLICIES:raise ValueError('unknown planner policy')
        self.policy=POLICIES[name]
        self.model=None;self.model_sha=None;self.decisions=[]
        if name=='step_time':
            if cost_model is None:raise RuntimeError('step_time requires an independently calibrated frozen cost model')
            payload=Path(cost_model).read_bytes();self.model_sha=hashlib.sha256(payload).hexdigest();self.model=json.loads(payload)
            shape=self.model.get('shape',{})
            semantic=self.model.get('semantic_only',False)
            if self.model.get('schema')!=1 or shape.get('ep')!=8 or shape.get('experts')!=256:
                raise ValueError('unsupported step_time model schema/shape')
            if semantic and os.environ.get('MEGAMOE_SEMANTIC_QUALIFICATION')!='1':
                raise ValueError('semantic-only costs are forbidden outside explicit qualification')
            if not semantic and shape!=dict(ep=8,experts=256,hidden=7168,ffn=2048,tokens=16384,topk=8):
                raise ValueError('measured cost model does not cover this executor shape')
            if (home,slots,tokens,topk)!=(32,32,shape['tokens'],shape['topk']):
                raise ValueError('step_time model/planner metadata mismatch')
        elif cost_model is not None:raise ValueError('a step-time cost model is only valid for step_time')
        self.validator=DevicePlanValidator(home=home,slots=slots,tokens=tokens,topk=topk,
            device=device,group=group,policy_signature=('fixed-owner-unified-v1',name,self.model_sha))
        v=self.validator
        if v.ep not in (2,4,8):raise ValueError('unified planner first version requires EP2/4/8')
        self.library=ctypes.CDLL('libascend_deepep_unified_planner.so')
        if int(self.library.megaflux_unified_planner_abi())!=1 or int(self.library.megaflux_max_local_experts())!=LOCAL_EXPERT_CAPACITY:
            raise RuntimeError('actual planner SO capability/ABI mismatch')
        alloc=lambda shape:torch.empty(shape,dtype=torch.int32,device=device)
        self.copies=alloc((v.ep,slots));self.physical=alloc((tokens,topk))
        self.logical=alloc(v.experts);self.destination=alloc((v.experts,v.ep));self.header=alloc(8)
        self.quota=alloc((v.ep,v.experts,v.ep))
        if self.model is None:self.costs=torch.zeros(1,dtype=torch.float32,device=device)
        else:
            import numpy as np
            values=np.asarray(self.model['device_vector'],dtype=np.float32)
            if values.shape!=(512,) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>=2**31) or np.any(values!=np.floor(values)):
                raise ValueError('invalid calibrated device model vector')
            expected={0:1,1:8,2:256,3:32,4:32,5:tokens,6:topk,7:shape['hidden'],8:shape['ffn'],
                      32:512,33:32768,35:int(semantic),55:2,63:16}
            if any(int(values[k])!=value for k,value in expected.items()) or not (0<=values[10]<=4 and 1<=values[11]<=4 and 4<=values[12]<=64):
                raise ValueError('cost model schema/workspace/search bounds mismatch')
            if self.model['variant'] not in ('full','no_gradient_service','linear_kernel','uniform_topology'):
                raise ValueError('unknown cost model ablation')
            self.extension=ctypes.CDLL('libascend_deepep_step_time_planner.so')
            if int(self.extension.megaflux_step_time_model_schema())!=1 or int(self.extension.megaflux_step_time_max_local_experts())!=LOCAL_EXPERT_CAPACITY:
                raise RuntimeError('step_time extension/model ABI mismatch')
            entry=ctypes.cast(self.extension.mega_moe_step_time_planner_do,ctypes.c_void_p)
            install=self.library.megaflux_install_policy_extension;install.argtypes=[ctypes.c_void_p];install.restype=None;install(entry)
            workspace=np.zeros(512+32768,dtype=np.float32);workspace[:512]=values
            self.costs=torch.from_numpy(workspace).to(device)
        self.calls=0

    def _compile(self,context,code,copies,quota):
        m=context.executor;v=self.validator;ids=context.logical_ids
        if not callable(getattr(m.workspace,'online_plan_unified',None)):
            raise RuntimeError('unified planner requires its rebuilt Host extension')
        if (m.home,m.slots,m.shmem.rank(),m.shmem.world_size())!=(v.home,v.slots,v.rank,v.ep):
            raise ValueError('unified planner/runtime topology mismatch')
        if ids.dtype!=torch.int32 or ids.device!=v.device or tuple(ids.shape)!=(v.tokens,v.topk) or not ids.is_contiguous():
            raise ValueError('unified planner input metadata mismatch')
        if self.model is not None:
            shape=self.model['shape']
            if (tuple(m.w1.shape),tuple(m.w2.shape),context.capacity)!=((32,2*shape['ffn'],shape['hidden']),
                    (32,shape['hidden'],shape['ffn']),int(self.model['device_vector'][9])):
                raise ValueError('actual executor shape/capacity is outside the frozen cost model')
        with control_timing_scope('planner.call'):
            m.workspace.online_plan_unified(m.shmem,ids,context.epoch,copies,self.physical,
                self.logical,self.destination,self.header,quota,self.costs,context.capacity,code)
        plan=v.validate(context,copies=copies,physical_ids=self.physical,native_header=self.header)
        plan.source_quota=quota
        if code==5:self.decisions.append(self.destination.flatten()[:8].detach().clone())
        self.calls+=1
        return plan

    def plan(self,context):
        return self._compile(context,self.policy.code,self.copies,self.quota)

    def compile_proposal(self,context,proposal):
        """Public route for another provider; same collective compiler/gate."""
        if not isinstance(proposal,DeviceProposal):raise TypeError('expected DeviceProposal(copies,quota)')
        return self._compile(context,6,proposal.copies,proposal.quota)
