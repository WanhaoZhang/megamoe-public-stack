"""External-parameter, checkpoint-aware MoE bridge and bounded native pool.

Native imports are lazy so the autograd/checkpoint contract can be tested on
CPU with a mathematical backend before running Ascend kernels.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import os
import torch
from .sft_checkpoint import claim


class ResidentPlacement:
    """Private replay of an already checked plan, with a new execution epoch."""
    def __init__(self, plan, host_control=None, host_copies=None,cache_identity=False,slim_zero=False):
        from .megaflux_planners import _CHECKED_PLAN
        if plan._certificate is not _CHECKED_PLAN:
            raise ValueError('only an already checked plan may be frozen')
        self.capacity_hint = plan.max_recv
        # Optional host snapshot is obtained from the actual checked header
        # by the zero fastpath. Legacy freezing adds no host synchronization.
        self.host_control=None if host_control is None else tuple(host_control)
        self.host_copies=host_copies
        # Only the opt-in path caches identity of an already checked input.
        # New checkpoint/router tensors and any normal in-place mutation still
        # require a real equality check against the immutable ID snapshot.
        self._cache_identity=host_control is not None or cache_identity
        self._checked_ids=(plan.logical_ids,plan.logical_ids._version) if self._cache_identity else None
        self._slim_zero=slim_zero
        self._unsealed_header=None
        self.tensors = {n: getattr(plan,n).detach().clone() for n in
            ('logical_ids','physical_ids','copies','logical_counts')}
        if slim_zero:
            # The first forward's status is not known yet. Do not synchronize
            # to inspect it, or clone its header on the NPU speculatively.
            h=plan.header
            self._unsealed_header=(h,h._version,h.data_ptr())
        else:self.tensors['header']=plan.header.detach().clone()
        self._stamps=tuple((n,t,t._version,t.data_ptr()) for n,t in self.tensors.items())

    def seal(self,status):
        """Called only after the first forward's successful checked finish."""
        if not self._slim_zero:
            self.host_control=tuple(status);return
        if self._unsealed_header is None:raise RuntimeError('checkpoint placement already sealed')
        h,v,p=self._unsealed_header
        if h._version!=v or h.data_ptr()!=p:raise RuntimeError('checkpoint header was mutated before finish')
        if len(status)!=8 or status[3] or status[1]>self.capacity_hint or status[7]!=self.capacity_hint:
            raise RuntimeError('cannot seal an invalid checkpoint plan')
        # Zero replay calls original F/B: their native APIs never receive this
        # header. CPU metadata supports observers without clone/fill NPU ops.
        header=torch.tensor(status,dtype=h.dtype,device='cpu') if status[0]==0 else h.detach().clone()
        self.tensors['header']=header
        self._stamps+=(("header",header,header._version,header.data_ptr()),)
        self.host_control=tuple(status);self._unsealed_header=None

    def plan(self, context):
        from .megaflux_planners import _CHECKED_PLAN
        m=context.executor
        if self._unsealed_header is not None:raise RuntimeError('finish before replaying checkpoint placement')
        if context.capacity != self.capacity_hint:
            raise ValueError('checkpoint replay changed receive capacity')
        if any(self.tensors[n] is not t or t._version!=v or t.data_ptr()!=p for n,t,v,p in self._stamps):
            raise RuntimeError('frozen checkpoint placement was mutated')
        known=self._checked_ids
        if known is None or context.logical_ids is not known[0] or context.logical_ids._version!=known[1]:
            if not torch.equal(context.logical_ids,self.tensors['logical_ids']):
                raise RuntimeError('checkpoint replay changed natural routing')
            if self._cache_identity:self._checked_ids=(context.logical_ids,context.logical_ids._version)
        host=None if self.host_control is None else list(self.host_control)
        if host is not None:host[2]=context.epoch
        host_zero=self._slim_zero and host is not None and host[0]==0
        if host_zero:header=torch.tensor(host,dtype=self.tensors['header'].dtype,device='cpu')
        else:header=self.tensors['header'].clone();header[2]=context.epoch
        return SimpleNamespace(**{k:v for k,v in self.tensors.items() if k not in ('header','logical_ids')},
            logical_ids=context.logical_ids,header=header,epoch=context.epoch,
            owner=m._plan_owner,max_recv=context.capacity,
            _host_control=host,
            _sft_host_only_zero=host_zero,
            _host_copies=self.host_copies,
            reference=SimpleNamespace(home=m.home,slots=m.slots),_certificate=_CHECKED_PLAN)


class _External(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge, x, ids, p, w1, w2):
        call,replay=claim(bridge.layer)
        lease=bridge.pool.acquire()
        try:
            y,counts,state,dispatch=lease.forward(x,ids,p,w1,w2,call.placement if replay else None)
            call.bind(lease,replay)
            if not replay:
                call.placement=lease.freeze_placement(ids)
            if not call.checkpointed or replay:
                lease.hold_backward()
            ctx.call=call;ctx.bridge=bridge
            ctx.lease_state=getattr(bridge.pool,'fastpath',None)=='device'
            if ctx.lease_state:
                # Expanded first-F and home-only zero-replay buffers may have
                # different shapes. Native state is lease-owned; p/W saves
                # still trigger checkpoint recompute and version checks.
                lease._sft_saved=(counts,state,dispatch)
                ctx.save_for_backward(p,w1,w2)
            else:ctx.save_for_backward(p,w1,w2,counts,state,dispatch)
            if bridge.observer is not None:
                bridge.observer.forward(bridge.layer,call,replay,x,ids,p,w1,w2,y,counts,lease)
            return y
        except BaseException:
            bridge.pool.failed=True
            raise

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,dy):
        # Accessing saved tensors causes non-reentrant checkpoint replay first.
        saved=ctx.saved_tensors
        call=ctx.call;lease=call.take_backward()
        if ctx.lease_state:
            p,w1,w2=saved
            if getattr(lease,'_sft_saved',None) is None:
                raise RuntimeError('live SFT recompute has no native saved state')
            counts,state,dispatch=lease._sft_saved
        else:p,w1,w2,counts,state,dispatch=saved
        try:
            result=lease.backward(dy,p,w1,w2,counts,state,dispatch)
            if ctx.bridge.observer is not None:
                ctx.bridge.observer.backward(ctx.bridge.layer,call,dy,result)
            call.finish_backward()
            return None,result[0],None,result[1],result[2],result[3]
        except BaseException:
            ctx.bridge.pool.failed=True
            raise


class SftMoEBridge:
    def __init__(self,pool,layer,observer=None):
        self.pool,self.layer,self.observer=pool,str(layer),observer

    def __call__(self,x,ids,p,w1,w2):
        if self.pool.failed:
            raise RuntimeError('failed SFT native pool cannot be reused')
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x,p,w1,w2)):
            return _External.apply(self,x,ids,p,w1,w2)
        lease=self.pool.acquire()
        try:
            y,*_=lease.forward(x,ids,p,w1,w2,None)
            lease.finish_forward()
            return y
        except BaseException:
            self.pool.failed=True
            raise


class NativePool:
    """One runtime per EP group; native slots grow only for truly live calls.

    With whole-layer checkpoint, original forwards retire at the frame exit;
    each recompute/backward then reuses the same slot. This avoids 48 persistent
    copies of home/replica weights and communication windows.
    """
    def __init__(self,shmem,home,slots,hidden,ffn,window_bytes,device,publish_fence,
                 backend='s9',max_tokens=2048,topk=8,beta=768,penalty=2,gain=4,
                 max_slots=2,worker_flags=3,timer=False,fastpath=None):
        if backend not in ('baseline','s9','worker'):
            raise ValueError('unknown SFT backend')
        self.shmem,self.home,self.slots,self.hidden,self.ffn=shmem,home,slots,hidden,ffn
        self.window_bytes,self.device,self.publish_fence=window_bytes,device,publish_fence
        self.backend,self.max_tokens,self.topk=backend,max_tokens,topk
        if worker_flags not in (0,1,2,3):raise ValueError('invalid worker feature flags')
        self.worker_flags=worker_flags
        if timer and backend!='worker':raise ValueError('detailed SFT Timer requires worker backend')
        self.timer=timer
        self.fastpath=os.environ.get('SFT_EXECUTOR_FASTPATH','legacy') if fastpath is None else fastpath
        self.slim_zero=os.environ.get('SFT_ZERO_REPLAY_SLIM','0')=='1'
        if self.fastpath not in ('legacy','zero','sparse','device'):
            raise ValueError('unknown SFT executor fastpath')
        if timer and self.fastpath!='legacy':
            raise ValueError('fastpath must be qualified without diagnostic Timer first')
        if self.slim_zero and self.fastpath!='device':raise ValueError('slim zero replay requires device fastpath')
        self.zero_replica_calls=0;self.home_refresh_calls=0
        self.home_refresh_bytes=0;self.exported_expert_copies=0
        self.home_refresh_bytes_measured=self.fastpath!='device'
        self.capacity=(shmem.world_size()*max_tokens*topk+127)//128*128
        self.policy=(beta,penalty,gain);self.max_slots=max_slots
        self.handles=[];self.failed=False;self.calls=0;self.backward_calls=0

    def acquire(self):
        if self.failed:raise RuntimeError('native pool failed')
        for h in self.handles:
            if not h.leased:
                h.leased=True;return h
        if len(self.handles)>=self.max_slots:
            raise RuntimeError('SFT live lease budget exceeded; do not overwrite pending buffers')
        h=_NativeLease(self);h.leased=True;self.handles.append(h);return h

    def close(self):
        if any(h.leased for h in self.handles):raise RuntimeError('SFT native leases remain live')
        for h in self.handles:h.module.close()
        self.handles.clear()


class _NativeLease:
    def __init__(self,pool):
        self.pool=pool;self.leased=False;self.stream=None;self.plan=None;self.ids=None
        self.use_original=False;self.host_control=None;self.host_copies=None
        if pool.backend=='baseline':
            from . import MegaMoeBF16Train
            self.module=MegaMoeBF16Train(pool.shmem,pool.window_bytes)
        else:
            # The frozen control-v2 SO also uses worker ABI3, with flags=0.
            # Passing the old Early descriptor to it makes the opaque trace
            # word look like an envelope pointer and is not ABI compatible.
            if pool.timer:
                from .megaflux_worker_timer import MegaFluxBF16WorkerTimerTrain as Executor
            else:
                from .megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain as Executor
            self.module=Executor(pool.shmem,pool.home,pool.slots,pool.hidden,pool.ffn,pool.window_bytes,pool.device)
            expected=0 if pool.backend=='s9' else pool.worker_flags
            if self.module.worker_flags!=expected:
                raise RuntimeError('SFT backend/native worker flag mismatch')
            if pool.fastpath=='device' and self.module.worker_external_refresh!=1:
                raise RuntimeError('device export requires an explicitly qualified external-refresh SO')

    @contextmanager
    def on_stream(self):
        current=torch.npu.current_stream()
        if current.npu_stream!=self.stream.npu_stream:self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):yield
        if current.npu_stream!=self.stream.npu_stream:current.wait_stream(self.stream)

    def _execution_descriptor(self):
        if getattr(self.plan,'_sft_host_only_zero',False):
            raise RuntimeError('CPU-only zero replay header must never enter a replica native API')
        descriptor=self.module._descriptor(self.plan)
        if self.pool.fastpath in ('sparse','device'):
            if len(descriptor)!=26 or descriptor[-1]!=1:
                raise RuntimeError('external framework weights require the full early descriptor')
            descriptor[-1]=3
        return descriptor

    def forward(self,x,ids,p,w1,w2,placement):
        q=self.pool;self.stream=torch.npu.current_stream()
        if x.shape[0]>q.max_tokens or ids.shape!=(x.shape[0],q.topk):
            raise ValueError('SFT token/route dimensions exceed collective capacity')
        if tuple(w1.shape)!=(q.home,2*q.ffn,q.hidden) or tuple(w2.shape)!=(q.home,q.hidden,q.ffn):
            raise ValueError('Qwen weights must be transformed to kernel layout')
        if any(t.dtype!=torch.bfloat16 for t in (x,p,w1,w2)):
            raise ValueError('SFT kernel inputs must be BF16')
        if placement is not None and q.backend=='baseline' and not torch.equal(placement,ids):
            raise RuntimeError('checkpoint replay changed baseline routing')
        self.ids=ids.detach().clone() if q.backend=='baseline' else None
        self.use_original=q.backend=='baseline';self.host_control=None;self.host_copies=None;self._frozen_placement=None
        if q.backend!='baseline' and q.fastpath=='legacy':
            self.module.w1.copy_(w1);self.module.w2.copy_(w2)
            q.home_refresh_calls+=1
            q.home_refresh_bytes+=(w1.numel()+w2.numel())*w1.element_size()
        # Device mode relies on the original kernel's existing entry
        # rendezvous too. Permit the same host-fence removal for the original
        # backend, so experiments can separate generic bridge savings from
        # gains due to replication/pipelining. Legacy denominators stay fixed.
        if q.fastpath=='legacy' or (q.backend=='baseline' and q.fastpath!='device'):
            self.stream.synchronize();q.publish_fence();self.stream.synchronize()
        q.calls+=1
        if q.backend=='baseline':
            y,counts,state,dispatch,_=self.module._op.compute_train(q.shmem,x,ids,p,w1,w2,q.capacity,0.,'bf16')
        else:
            from .megaflux_planners import NativeOnlinePlanner
            policy=placement if placement is not None else NativeOnlinePlanner(*q.policy)
            self.plan=self.module.create_plan(ids,policy,q.capacity)
            self.plan._consumed=True
            if q.fastpath=='device':
                self.module._check_device_plan(self.plan)
                proof=getattr(self.plan,'_host_control',None)
                if proof is not None and proof[0]==0:
                    if proof[3] or proof[2]!=self.plan.epoch or proof[1]>q.capacity or proof[7]!=q.capacity:
                        raise RuntimeError('frozen zero device plan has invalid status')
                    self.host_control=tuple(proof);self.use_original=True
                    # Original kernel owns its own device entry rendezvous.
                    # This is a replay of a finished, checked zero placement.
                    y,counts,state,dispatch,_=self.module.op.compute_train(
                        q.shmem,x,ids,p,w1,w2,q.capacity,0.,'bf16')
                else:
                    y,counts,state,dispatch,_=self.module.op.compute_train_replica(
                        q.shmem,x,self.plan.physical_ids,p,w1,w2,q.capacity,0.,'bf16',self._execution_descriptor())
                self.module.last_exec_counts=counts
            elif q.fastpath!='legacy':
                # Read the actual collective header, including errors, before
                # choosing a collective kernel. A bad plan cannot enter the
                # original kernel and bypass its capacity/error gate.
                header=getattr(self.plan,'_host_control',None)
                if header is None:header=self.plan.header.cpu().tolist()
                if header[3] or header[2]!=self.plan.epoch or header[1]>q.capacity or header[7]!=q.capacity:
                    raise RuntimeError('fastpath rejected planner error/epoch/capacity')
                self.host_control=tuple(header);self.use_original=header[0]==0
                if self.use_original:
                    self.stream.synchronize();q.publish_fence();self.stream.synchronize()
                    y,counts,state,dispatch,_=self.module.op.compute_train(
                        q.shmem,x,ids,p,w1,w2,q.capacity,0.,'bf16')
                    self.module.last_exec_counts=counts;q.zero_replica_calls+=1
                else:
                    if q.fastpath=='sparse':
                        # Local GEMMs use the actual framework weights. Only
                        # experts exported to remote replicas need SHMEM homes;
                        # include exported experts even if owner has zero rows.
                        copies=getattr(self.plan,'_host_copies',None)
                        if copies is None:copies=tuple(tuple(r) for r in self.plan.copies.cpu().tolist())
                        self.host_copies=copies;rank=q.shmem.rank()
                        exported=sorted({e%q.home for row in copies for e in row if e>=0 and e//q.home==rank})
                        for e in exported:
                            self.module.w1[e].copy_(w1[e]);self.module.w2[e].copy_(w2[e])
                        q.exported_expert_copies+=len(exported)
                        q.home_refresh_bytes+=len(exported)*(w1[0].numel()+w2[0].numel())*w1.element_size()
                    else:
                        self.module.w1.copy_(w1);self.module.w2.copy_(w2)
                        q.home_refresh_bytes+=(w1.numel()+w2.numel())*w1.element_size()
                    q.home_refresh_calls+=1
                    self.stream.synchronize();q.publish_fence();self.stream.synchronize()
                    if q.fastpath=='sparse':
                        self.module._check_device_plan(self.plan)
                        y,counts,state,dispatch,_=self.module.op.compute_train_replica(
                            q.shmem,x,self.plan.physical_ids,p,w1,w2,q.capacity,0.,'bf16',self._execution_descriptor())
                        self.module.last_exec_counts=counts
                    else:
                        y,_,counts,state,dispatch=self.module._forward_device(x,p,self.plan,0.)
            else:
                y,_,counts,state,dispatch=self.module._forward_device(x,p,self.plan,0.)
        return y,counts,state,dispatch

    def freeze_placement(self,ids):
        if self.pool.backend=='baseline':return self.ids
        self._frozen_placement=ResidentPlacement(self.plan,self.host_control,self.host_copies,
            cache_identity=self.pool.fastpath=='device',slim_zero=self.pool.slim_zero)
        return self._frozen_placement

    def hold_backward(self):
        if self.pool.backend!='baseline':self.module._active=True

    def backward(self,dy,p,w1,w2,counts,state,dispatch):
        q=self.pool
        with self.on_stream():
            if self.use_original:
                if q.backend!='baseline':self.module._check_device_plan(self.plan)
                op=self.module._op if q.backend=='baseline' else self.module.op
                out=op.backward(q.shmem,dy.contiguous(),p,w1,w2,counts,state,dispatch,q.capacity,0.,'bf16')
                if q.backend!='baseline':self.module._active=False
            else:
                self.module._check_device_plan(self.plan)
                a,b=(w1,w2) if q.fastpath in ('sparse','device') else (self.module.w1,self.module.w2)
                dx,dp,a,b=self.module.op.backward_replica(q.shmem,dy.contiguous(),p,
                    a,b,counts,state,dispatch,q.capacity,0.,'bf16',self._execution_descriptor())
                out=(dx,dp,a[:q.home],b[:q.home]);self.module._active=False
            q.backward_calls+=1
        return out

    def finish_forward(self):
        if self.pool.backend!='baseline' and self.module._active:
            raise RuntimeError('cannot discard a non-checkpoint training lease')
        self._finish()

    def finish_backward(self):self._finish()

    def _finish(self):
        if not self.leased:raise RuntimeError('native SFT lease already released')
        with self.on_stream():
            if self.pool.backend=='baseline':self.stream.synchronize()
            elif self.use_original and self.pool.fastpath!='legacy':
                # The actual planner header was already copied/validated.
                # Original F/B never writes it. Keep lease/version checks and
                # the execution-stream drain, but do not DMA the same control
                # header a second time. All replica paths still use finish().
                self.module._check_device_plan(self.plan)
                self.stream.synchronize()
                self.module.last_status=list(self.host_control)
                self.module._pending=False
                if self.pool.fastpath=='device':self.pool.zero_replica_calls+=1
            else:
                status=self.module.finish()
                if self.pool.fastpath=='device':
                    if status[0]:self.pool.home_refresh_calls+=1
                    else:self.pool.zero_replica_calls+=1
            if self.pool.backend!='baseline' and self.pool.fastpath=='device' and self._frozen_placement is not None:
                # Attach only status returned by the real checked finish (or a
                # validated replay of it). Original checkpoint F has retired;
                # a later zero replay may use the original kernel directly.
                self._frozen_placement.seal(self.module.last_status)
        self.plan=None;self.ids=None;self.leased=False
