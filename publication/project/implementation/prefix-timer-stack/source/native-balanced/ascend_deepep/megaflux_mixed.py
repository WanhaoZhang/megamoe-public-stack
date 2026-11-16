"""S5: coarse copy/compute and compute/owner-SUM in mixed BF16 kernels.

The S4 planner and 32-byte control read are retained. There is no overlap yet.
"""
import torch
from torch.autograd.function import once_differentiable
from .megaflux import MegaFluxBF16Train

class _MixedReplica(torch.autograd.Function):
    @staticmethod
    def forward(ctx,module,x,p,w1,w2,plan,limit):
        if module._active:raise RuntimeError('one unfinished forward per mixed operator')
        module._active=True
        try:
            y,logical,counts,state,dispatch=module._forward(x,p,plan,float(limit))
            ctx.module=module;ctx.plan=plan;ctx.limit=float(limit)
            ctx.save_for_backward(p,w1,w2,counts,state,dispatch)
            ctx.mark_non_differentiable(logical)
            return y,logical
        except Exception:
            module._active=False;raise
    @staticmethod
    @once_differentiable
    def backward(ctx,dy,unused=None):
        m=ctx.module;p,w1,w2,counts,state,dispatch=ctx.saved_tensors
        try:
            args=(m.shmem,dy.contiguous(),p,w1,w2,counts,state,dispatch,
                  ctx.plan.max_recv,ctx.limit,'bf16')
            if ctx.plan.has_replicas:
                descriptor=m.workspace.descriptor(m.shmem,ctx.plan.copies)
                dx,dp,a,b=m.op.backward_replica(*args,descriptor)
                dw1,dw2=a[:m.home],b[:m.home]
            else:dx,dp,dw1,dw2=m.op.backward(*args)
            return None,dx,dp,dw1,dw2,None,None
        finally:m._active=False

class MegaFluxBF16MixedTrain(MegaFluxBF16Train):
    def __init__(self,shmem,home_experts,slots,hidden,ffn,local_mem_size,device,optimized=True):
        if not optimized:raise ValueError('mixed executor requires optimized sparse policy')
        self.shmem=shmem;self.home=home_experts;self.slots=slots;self.optimized=True
        self.workspace=torch.classes.AscendDeepEP.MegaMoeReplicaWorkspace.split(home_experts,slots,hidden,ffn)
        template=torch.empty(0,device=device,dtype=torch.bfloat16)
        a,b=self.workspace.weights(template)
        self.w1=a.detach().requires_grad_();self.w2=b.detach().requires_grad_()
        self.replica_w1,self.replica_w2=self.workspace.replicas(template)
        self.op=torch.classes.AscendDeepEP.MegaMoeBF16(local_mem_size)
        self.op.prepare_context(shmem,template)
        self._active=False;self._closed=False;self.last_exec_counts=None
        self._epoch=0;self.last_plan=None
    def _forward(self,x,p,plan,limit):
        if plan.has_replicas:
            physical=plan.physical_ids if hasattr(plan,'physical_ids') else self.workspace.map_routes(plan.route_blocks,plan.block_exec)
            descriptor=self.workspace.descriptor(self.shmem,plan.copies)
            y,counts,state,dispatch,_=self.op.compute_train_replica(self.shmem,x,physical,p,
                self.w1,self.w2,plan.max_recv,limit,'bf16',descriptor)
            logical=self.workspace.counts(self.shmem,plan.copies,counts)
        else:
            y,counts,state,dispatch,_=self.op.compute_train(self.shmem,x,plan.logical_ids,p,
                self.w1,self.w2,plan.max_recv,limit,'bf16')
            logical=counts
        self.last_exec_counts=counts
        return y,logical,counts,state,dispatch
    def __call__(self,x,p,plan,limit=0.0,copy_weights=True):
        self._check(x,p,plan)
        if not copy_weights:raise ValueError('S5 refreshes replica weights on every forward')
        if not torch.is_grad_enabled():return self.forward_only(x,p,plan,limit)
        return _MixedReplica.apply(self,x,p,self.w1,self.w2,plan,limit)
    def forward_only(self,x,p,plan,limit=0.0,copy_weights=True):
        self._check(x,p,plan)
        if self._active:raise RuntimeError('unfinished backward')
        if not copy_weights:raise ValueError('S5 refreshes replica weights on every forward')
        y,logical,*_=self._forward(x,p,plan,float(limit))
        return y,logical
