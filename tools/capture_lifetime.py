"""Lifecycle dependency scopes using the existing rank-local Event timeline.

These measurements include diagnostic overhead. Host envelopes are recorded
separately; nested intervals are never added to infer critical-path length.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import json
import time
import torch
import torch.distributed as dist


def install(out,rank,scopes,invocation):
    import lifetime_device as module
    cls=module.LifetimeDevice
    active=ContextVar('lifetime_diagnostic_context',default=None)
    group_active=ContextVar('group_manifest_context',default=None)
    records=[];edges=[];sequence=0

    def identity(instance,method,args):
        request=invocation()
        if request is not None:return request
        version=args[0] if method=='begin' else instance.context.version
        return ((-1 if version is None else version,-1,
                 -1 if instance.layer is None else instance.layer,-1),False)

    def record(instance,name,key,replay,start,finish,**extra):
        nonlocal sequence
        records.append(dict(rank=rank,sequence=sequence,name=name,mode=instance.mode,
            invocation=list(key),recompute=replay,version=instance.context.version,
            logical_layer=instance.layer,host_begin_ns=start,host_end_ns=finish,
            host_elapsed_ms=(finish-start)/1e6,diagnostic_only=True,
            nested_scopes_nonadditive=True,**extra))
        sequence+=1

    for name in ('begin','prepare','commit','gradients','flush','_drain','window_boundary','finish'):
        original=getattr(cls,name)
        def decorate(original=original,name=name):
            @functools.wraps(original)
            def call(self,*args,**kwargs):
                key,replay=identity(self,name,args)
                token=active.set((self,key,replay))
                event=scopes['begin']('lifetime.'+name,key=key,recompute=replay)
                before=dict(self.stats);pending_before=len(self.pending)
                start=time.monotonic_ns();failed=False
                try:return original(self,*args,**kwargs)
                except BaseException:failed=True;raise
                finally:
                    finish=time.monotonic_ns();scopes['end'](event,failed)
                    record(self,name,key,replay,start,finish,event_sequence=event[0]['sequence'],
                        pending_before=pending_before,pending_after=len(self.pending),failed=failed,
                        payload_delta={k:self.stats[k]-v for k,v in before.items()})
                    active.reset(token)
            return call
        setattr(cls,name,decorate())

    original_observe=module.observe_tensor
    def observe(tensor,label):
        state=active.get()
        if state is None:return original_observe(tensor,label)
        instance,key,replay=state;start=time.monotonic_ns()
        try:return original_observe(tensor,label)
        finally:record(instance,'host_observe.'+label,key,replay,start,time.monotonic_ns(),
                       tensor_shape=list(tensor.shape),bytes=tensor.numel()*tensor.element_size())
    module.observe_tensor=observe

    original_gather=dist.all_gather_object
    @functools.wraps(original_gather)
    def gather(*args,**kwargs):
        state=active.get()
        if state is None:
            group=group_active.get()
            if group is None:return original_gather(*args,**kwargs)
            key,replay,layers=group;start=time.monotonic_ns()
            try:return original_gather(*args,**kwargs)
            finally:
                records.append(dict(rank=rank,name='host_group_manifest_all_gather',mode='framework',
                    invocation=list(key),recompute=replay,logical_layers=list(layers),
                    host_begin_ns=start,host_end_ns=time.monotonic_ns(),diagnostic_only=True,
                    semantics='common grouped-framework identity check, not lifecycle-specific'))
        instance,key,replay=state;start=time.monotonic_ns()
        try:return original_gather(*args,**kwargs)
        finally:record(instance,'host_object_all_gather',key,replay,start,time.monotonic_ns())
    dist.all_gather_object=gather

    from grouped_experts import GroupedExpertsBridge
    original_group=GroupedExpertsBridge.__call__
    def grouped(self,tasks,key,*,recompute=False,limit=10.):
        token=group_active.set((key,recompute,self.domain.layers))
        try:return original_group(self,tasks,key,recompute=recompute,limit=limit)
        finally:group_active.reset(token)
    GroupedExpertsBridge.__call__=grouped
    original_weights=GroupedExpertsBridge._weights
    def weights(self):
        group=group_active.get();request=invocation()
        key,replay=(group[:2] if group is not None else request) if (group is not None or request is not None) else ((-1,-1,self.domain.layers[0],-2),False)
        event=scopes['begin']('framework.owner_compute_materialization',key=key,recompute=replay)
        start=time.monotonic_ns()
        try:
            result=original_weights(self)
            records.append(dict(rank=rank,name='owner_compute_materialization_enqueue',mode='framework',
                invocation=list(key),recompute=replay,logical_layers=list(self.domain.layers),
                host_begin_ns=start,host_end_ns=time.monotonic_ns(),diagnostic_only=True,
                output_dtypes=[str(value.dtype) for value in result],
                output_bytes=sum(value.numel()*value.element_size() for value in result),
                event_sequence=event[0]['sequence']))
            return result
        finally:scopes['end'](event)
    GroupedExpertsBridge._weights=weights

    original_add=cls._add_owner
    def add_owner(self,layer,expert,first,second):
        state=active.get();start=time.monotonic_ns()
        try:return original_add(self,layer,expert,first,second)
        finally:
            if state is not None:
                _,key,replay=state
                record(self,'owner_publish_enqueue',key,replay,start,time.monotonic_ns(),
                       owner_layer=layer,local_expert=expert,
                       logical_bytes=(first.numel()+second.numel())*4,
                       semantics='host enqueue only; device completion belongs to service-work Event')
    cls._add_owner=add_owner

    original_batch=dist.batch_isend_irecv
    def batch(operations,*args,**kwargs):
        state=active.get()
        if state is None:return original_batch(operations,*args,**kwargs)
        instance,key,replay=state;packet=getattr(instance,'_diagnostic_packet',None)
        start=time.monotonic_ns()
        works=original_batch(operations,*args,**kwargs)
        record(instance,'p2p_submit_enqueue',key,replay,start,time.monotonic_ns(),
               packet=packet,operations=[dict(peer=op.peer,bytes=op.tensor.numel()*op.tensor.element_size(),
                                             kind=op.op.__name__) for op in operations])
        class Work:
            def __init__(self,work):self.original=work
            def __getattr__(self,name):return getattr(self.original,name)
            def wait(self,*a,**kw):
                began=time.monotonic_ns()
                try:return self.original.wait(*a,**kw)
                finally:record(instance,'p2p_work_wait',key,replay,began,time.monotonic_ns(),packet=packet,
                    semantics='host wait-return envelope; service Event records device completion')
        return [Work(work) for work in works]
    dist.batch_isend_irecv=batch

    original_scope=cls._delivery_scope
    @contextmanager
    def delivery(self):
        state=active.get()
        if state is None or self.service is None:
            with original_scope(self):yield
            return
        _,key,replay=state
        producer=scopes['begin']('lifetime.producer_ready',key=key,recompute=replay)
        scopes['end'](producer)
        waiting=scopes['begin']('lifetime.service_wait_for_producer',stream=self.service,key=key,recompute=replay)
        with original_scope(self):
            scopes['end'](waiting)
            service=scopes['begin']('lifetime.service_work',stream=self.service,key=key,recompute=replay)
            edges.append(dict(rank=rank,source_event=producer[0]['sequence'],
                target_event=waiting[0]['sequence'],next_event=service[0]['sequence'],
                reason='existing service.wait_event(current-stream ready)',
                invocation=list(key),diagnostic_only=True,clock='one-device only'))
            try:yield
            finally:scopes['end'](service)
    cls._delivery_scope=delivery

    def export():
        for path,values in ((out/f'lifetime-events-rank{rank}.jsonl',records),
                            (out/f'lifetime-dependencies-rank{rank}.jsonl',edges)):
            if values:
                with path.open('a') as handle:
                    for row in values:handle.write(json.dumps(row)+'\n')
                values.clear()
    scopes['register_flusher'](export)
