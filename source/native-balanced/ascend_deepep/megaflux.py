"""S0--S4 expert replication using the unchanged BF16 compute kernels.

One active forward per instance. Online planning is a separate device stage;
persistent overlap is future work. Workspace aliases must not outlive this module.
"""
import torch
from torch.autograd.function import once_differentiable
from .megaflux_plan import ReplicaPlan

class DeviceReplicaPlan:
    def __init__(self, plan:ReplicaPlan, rank:int, device):
        self.reference=plan
        self.copies=torch.as_tensor(plan.copies.copy(),device=device,dtype=torch.int32)
        self.route_blocks=torch.as_tensor(plan.route_blocks[rank].copy(),device=device,dtype=torch.int32)
        self.block_exec=torch.as_tensor(plan.block_exec.copy(),device=device,dtype=torch.int32)
        self.max_recv=plan.max_recv
        # Decode physical slots using the immutable CPU plan, preserving the
        # original logical IDs even when spare slots change the rank stride.
        import numpy as np
        slotmap=np.concatenate((np.arange(plan.home*len(plan.copies)).reshape(-1,plan.home),plan.copies),axis=1)
        self.logical_ids=torch.as_tensor(slotmap.ravel()[plan.physical_ids[rank]],device=device,dtype=torch.int32)
        self.has_replicas=bool((plan.copies>=0).any())

class OnlineReplicaPlan:
    def __init__(self,module,ids,beta,penalty_blocks,min_gain_blocks,epoch):
        from types import SimpleNamespace
        self.reference=SimpleNamespace(home=module.home,slots=module.slots)
        self.logical_ids=ids
        self.copies,self.physical_ids,self.logical_counts,self.assignment,self.header=module.workspace.online_plan(
            module.shmem,ids,int(beta),int(penalty_blocks),int(min_gain_blocks),int(epoch))
        # Only 32 control bytes cross to host. The compute ABI takes host-side
        # dimensions; include this blocking read in every online measurement.
        control=self.header.cpu().tolist()
        if control[3]:raise ValueError('online planner invalid routes or rank policy/epoch mismatch: '+str(control))
        if control[2]!=epoch:raise RuntimeError('stale planner epoch')
        self.has_replicas=bool(control[0]);self.max_recv=int(control[1])
        self.epoch=epoch;self.control=control

class _StagedReplica(torch.autograd.Function):
    @staticmethod
    def forward(ctx,module,x,p,w1,w2,plan,limit,copy_weights):
        if module._active:raise RuntimeError('one unfinished forward per staged operator')
        if w1.data_ptr()!=module.w1.data_ptr() or w2.data_ptr()!=module.w2.data_ptr():
            raise ValueError('use owner views returned by this module; no implicit home staging')
        module._active=True
        try:
            fast=module.optimized and not plan.has_replicas
            physical=plan.logical_ids if fast else (plan.physical_ids if hasattr(plan,'physical_ids') else module.workspace.map_routes(plan.route_blocks,plan.block_exec))
            if copy_weights and not fast:module.workspace.copy(module.shmem,plan.copies)
            a,b=(module.w1,module.w2) if fast else (module.exec_w1,module.exec_w2)
            y,counts,state,dispatch,transient=module.op.compute_train(module.shmem,x,physical,p,
                a,b,plan.max_recv,float(limit),'bf16')
            logical=counts if fast else module.workspace.counts(module.shmem,plan.copies,counts)
            ctx.module=module;ctx.plan=plan;ctx.limit=float(limit);ctx.fast=fast
            ctx.save_for_backward(p,w1,w2,counts,state,dispatch)
            ctx.mark_non_differentiable(logical)
            module.last_exec_counts=counts
            return y,logical
        except Exception:
            module._active=False;raise
    @staticmethod
    @once_differentiable
    def backward(ctx,dy,unused=None):
        module=ctx.module;p,w1,w2,counts,state,dispatch=ctx.saved_tensors
        try:
            a,b=(w1,w2) if ctx.fast else (module.exec_w1,module.exec_w2)
            dx,dp,dw1,dw2=module.op.backward(module.shmem,dy.contiguous(),p,
                a,b,counts,state,dispatch,
                ctx.plan.max_recv,ctx.limit,'bf16')
            if ctx.fast:owner1,owner2=dw1,dw2
            else:
                reduce=module.workspace.reduce_sparse if module.optimized else module.workspace.reduce
                owner1,owner2=reduce(module.shmem,ctx.plan.copies,dw1,dw2)
            return None,dx,dp,owner1,owner2,None,None,None
        finally:module._active=False

class MegaFluxBF16Train:
    def __init__(self,shmem,home_experts,slots,hidden,ffn,local_mem_size,device,optimized=True):
        self.shmem=shmem;self.home=home_experts;self.slots=slots
        self.optimized=optimized
        self.workspace=torch.classes.AscendDeepEP.MegaMoeReplicaWorkspace(home_experts,slots,hidden,ffn)
        template=torch.empty(0,device=device,dtype=torch.bfloat16)
        self.exec_w1,self.exec_w2=self.workspace.weights(template)
        self.w1=self.exec_w1[:home_experts].detach().requires_grad_()
        self.w2=self.exec_w2[:home_experts].detach().requires_grad_()
        self.op=torch.classes.AscendDeepEP.MegaMoeBF16(local_mem_size)
        self.op.prepare_context(shmem,template)
        self._active=False;self._closed=False;self.last_exec_counts=None
        self._epoch=0;self.last_plan=None
    def plan_online(self,logical_ids,beta=768,penalty_blocks=0,min_gain_blocks=0):
        if self._closed:raise RuntimeError('staged operator closed')
        if self._active:raise RuntimeError('cannot replan before the saved backward finishes')
        self._epoch+=1
        result=OnlineReplicaPlan(self,logical_ids,beta,penalty_blocks,min_gain_blocks,self._epoch)
        self.last_plan=result
        return result
    def forward_online(self,x,logical_ids,p,limit=0.0,beta=768,penalty_blocks=0,min_gain_blocks=0):
        plan=self.plan_online(logical_ids,beta,penalty_blocks,min_gain_blocks)
        return self(x,p,plan,limit)
    def forward_cpu_online(self,x,logical_ids,p,limit=0.0,beta=768,penalty_blocks=0,min_gain_blocks=0):
        """Diagnostic host policy: all route D2H/gather/CPU/H2D costs are real."""
        import numpy as np
        import torch.distributed as dist
        from .megaflux_plan import make_plan
        if self._active:raise RuntimeError('cannot replan before the saved backward finishes')
        local=logical_ids.detach().cpu().numpy()
        gathered=[None]*dist.get_world_size();dist.all_gather_object(gathered,local)
        fixed=make_plan(np.stack(gathered),self.home*len(gathered),self.slots,beta,
                        penalty_blocks=penalty_blocks,min_gain_blocks=min_gain_blocks)
        plan=DeviceReplicaPlan(fixed,dist.get_rank(),logical_ids.device)
        self.last_plan=plan
        return self(x,p,plan,limit)
    def _check(self,x,p,plan):
        if self._closed:raise RuntimeError('staged operator closed')
        if plan.reference.home!=self.home or plan.reference.slots!=self.slots:
            raise ValueError('plan/workspace expert capacity mismatch')
        if tuple(p.shape)!=tuple(plan.logical_ids.shape) or x.shape[0]!=p.shape[0]:
            raise ValueError('fixed plan must match this call token/TopK shape')
    def __call__(self,x,p,plan,limit=0.0,copy_weights=True):
        self._check(x,p,plan)
        if not torch.is_grad_enabled():return self.forward_only(x,p,plan,limit,copy_weights)
        return _StagedReplica.apply(self,x,p,self.w1,self.w2,plan,limit,copy_weights)
    def forward_only(self,x,p,plan,limit=0.0,copy_weights=True):
        self._check(x,p,plan)
        if self._active:raise RuntimeError('unfinished backward')
        fast=self.optimized and not plan.has_replicas
        physical=plan.logical_ids if fast else (plan.physical_ids if hasattr(plan,'physical_ids') else self.workspace.map_routes(plan.route_blocks,plan.block_exec))
        if copy_weights and not fast:self.workspace.copy(self.shmem,plan.copies)
        a,b=(self.w1,self.w2) if fast else (self.exec_w1,self.exec_w2)
        y,counts,*_=self.op.compute_train(self.shmem,x,physical,p,a,b,
            plan.max_recv,float(limit),'bf16')
        self.last_exec_counts=counts
        logical=counts if fast else self.workspace.counts(self.shmem,plan.copies,counts)
        return y,logical
    def close(self):
        if self._closed:return
        if self._active:raise RuntimeError('cannot close before backward')
        torch.npu.synchronize()
        self.op.destroy()
        self._closed=True
