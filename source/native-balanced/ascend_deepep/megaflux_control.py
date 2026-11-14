"""Optimized online control; current routes are replanned on every invocation.

The owner-only fast path uses logical IDs. Native pinned control read remains
blocking and is part of the call; this is not a device-only control ABI.
"""
from types import SimpleNamespace
import torch
from .megaflux_mixed import MegaFluxBF16MixedTrain

class ControlReplicaPlan:
    def __init__(self,module,ids,beta,penalty,gain,epoch):
        self.reference=SimpleNamespace(home=module.home,slots=module.slots)
        self.logical_ids=ids;self.epoch=epoch;self.owner_token=module._plan_owner
        shape=(tuple(ids.shape),ids.device)
        if module._buffer_shape!=shape:
            ep=module.shmem.world_size();e=module.home*ep
            make=lambda size:torch.empty(size,device=ids.device,dtype=torch.int32)
            module._plan_buffers=(make((ep,module.slots)),make(ids.shape),make((e,)),make((e,ep)),make((8,)))
            module._buffer_shape=shape
        self.copies,scratch,self.logical_counts,self.assignment,self.header=module._plan_buffers
        self.control=module.workspace.online_plan_control(module.shmem,ids,int(beta),int(penalty),int(gain),int(epoch),*module._plan_buffers)
        if self.control[3]:raise ValueError('online planner invalid routes or rank policy/epoch mismatch: '+str(self.control))
        if self.control[2]!=epoch:raise RuntimeError('stale optimized planner epoch')
        self.has_replicas=bool(self.control[0]);self.max_recv=int(self.control[1])
        self.physical_valid=bool(self.control[5])
        if self.physical_valid!=self.has_replicas:raise RuntimeError('planner physical validity mismatch')
        self.physical_ids=scratch if self.has_replicas else ids
        self.physical_stride=module.home+module.slots if self.has_replicas else module.home

class MegaFluxBF16ControlTrain(MegaFluxBF16MixedTrain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._plan_buffers=None;self._buffer_shape=None;self._plan_owner=object()
    def plan_online(self,logical_ids,beta=768,penalty_blocks=0,min_gain_blocks=0):
        if self._closed:raise RuntimeError('operator closed')
        if self._active:raise RuntimeError('cannot replan before the saved backward finishes')
        self._epoch+=1
        plan=ControlReplicaPlan(self,logical_ids,beta,penalty_blocks,min_gain_blocks,self._epoch)
        self.last_plan=plan
        return plan
    def _check(self,x,p,plan):
        super()._check(x,p,plan)
        if isinstance(plan,ControlReplicaPlan) and (plan.owner_token is not self._plan_owner or plan.epoch!=self._epoch):
            raise RuntimeError('optimized plan lease expired or belongs to another operator')
