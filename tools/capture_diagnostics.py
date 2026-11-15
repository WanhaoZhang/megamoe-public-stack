"""Original TimerV2 snapshots, queued on-stream and exported at step retirement.

There is no per-pool synchronize or CPU read. Snapshot copies and Event markers
are diagnostic overhead; neither these runs nor their wall times are formal
performance scores. Each device retains its own original clock domain.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch


def install(out:Path,rank:int):
    from capture_framework import install as install_framework_scopes
    scopes=install_framework_scopes(out,rank)
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    from ascend_deepep.unified_planner import CommonOnlinePlanner
    from ascend_deepep.megaflux_stage_layout import PREFIX,HEADER,LANES,WORDS
    from ascend_deepep.plan_replay import invocation as plan_invocation
    original_forward=NativeFSDP2Pool.forward;original_backward=NativeFSDP2Pool.backward
    original_compile=CommonOnlinePlanner._compile;original_publish=NativeFSDP2Pool._publish
    pending=[];controls=[];contexts=[];counter=0;control_counter=0

    def invocation():
        return contexts[-1] if contexts else plan_invocation()

    def control(phase,fn,args,kwargs):
        nonlocal control_counter
        request=invocation()
        if request is None:return fn(*args,**kwargs)
        key,recompute=request;token=scopes['begin'](phase,key=key,recompute=recompute)
        start=time.monotonic_ns()
        try:return fn(*args,**kwargs)
        finally:
            finish=time.monotonic_ns();scopes['end'](token)
            controls.append(dict(rank=rank,sequence=control_counter,phase=phase,invocation=list(key),
                recompute=recompute,call_enqueue_ms=(finish-start)/1e6,host_begin_ns=start,host_return_ns=finish,
                event_sequence=token[0]['sequence'],diagnostic_only=True,
                semantics='host enqueue/return envelope; device completion is the same-stream Event end'))
            control_counter+=1

    def compile(self,*args,**kwargs):return control('online_plan_and_common_validation',original_compile,(self,*args),kwargs)
    def publish(self,*args,**kwargs):return control('framework_owner_to_bf16_mirror',original_publish,(self,*args),kwargs)

    class OperatorScopes:
        def __init__(self,native,pool):self.native,self.pool=native,pool
        def __getattr__(self,name):
            value=getattr(self.native,name)
            if name not in ('compute_train_replica','backward_replica','backward_replica_redispatch_with_timer'):return value
            def invoke(*args,**kwargs):
                key,recompute=invocation()
                phase='forward' if name=='compute_train_replica' else 'backward'
                token=scopes['begin']('operator.'+phase,key=key,recompute=recompute)
                self.pool._diagnostic_operator_event=(tuple(key),phase,token[0]['sequence'])
                try:return value(*args,**kwargs)
                finally:scopes['end'](token)
            return invoke

    def snapshot(pool,phase,key,recompute,token,start,finish,body):
        nonlocal counter
        m=pool.module;plan=m.last_plan
        # Clones follow this invocation on the consuming stream, before the
        # reusable phase/worker buffers can be overwritten by another layer.
        before=torch.npu.Event(enable_timing=True);after=torch.npu.Event(enable_timing=True);before.record()
        arrays={'original_'+phase:m.trace_timers[phase].detach().clone()}
        if phase=='backward':
            transport=getattr(pool,'last_x_redispatch_backward_timer',None)
            if transport is not None:arrays['original_x_redispatch_backward']=transport.detach().clone()
        raw=getattr(m,'pipeline_trace' if phase=='forward' else 'gradient_trace')
        arrays['stage_raw']=raw[PREFIX[phase]:].detach().clone()
        for name in ('logical_ids','physical_ids','copies','logical_counts'):
            arrays[name]=getattr(plan,name).detach().clone()
        if body is None:raise RuntimeError('diagnostic invocation has no frozen source-aware plan body')
        arrays['source_quota']=body.source_quota.detach().clone()
        if m.last_exec_counts is None:raise RuntimeError('missing actual execution counts')
        arrays['execution_counts']=m.last_exec_counts.detach().clone()
        after.record()
        metadata=dict(rank=rank,sequence=counter,phase=phase,invocation=list(key),recompute=recompute,
            call_enqueue_ms=(finish-start)/1e6,layout=dict(m.stage_layouts[phase]),epoch=plan.epoch,
            host_begin_ns=start,host_return_ns=finish,event_sequence=token[0]['sequence'],
            host_clock='same-host monotonic_ns enqueue/return envelopes; not device completion timestamps',
            device_clock='rank-local GetSystemCycle, not merged with host or other ranks',
            asynchronous_snapshot=True,per_call_synchronize=False,diagnostic_only=True)
        operator_key,operator_phase,operator_event=pool._diagnostic_operator_event
        if operator_key!=tuple(key) or operator_phase!=phase:raise RuntimeError('operator Event identity mismatch')
        metadata['operator_event_sequence']=operator_event
        pending.append((metadata,arrays,before,after));counter+=1

    def export():
        # capture_framework.flush has completed all queued work once per step.
        directory=out/'timer';directory.mkdir(exist_ok=True)
        for metadata,tensors,before,after in pending:
            start=time.monotonic_ns();arrays={k:v.cpu().numpy() for k,v in tensors.items()}
            raw=arrays.pop('stage_raw');head=raw[:HEADER+LANES*8];counters=head[HEADER:].reshape(LANES,8)
            if counters[:,1:3].any():raise RuntimeError('duplicate or out-of-bounds original stage record')
            pairs=raw[HEADER+LANES*8:].reshape(-1,WORDS);indices=np.flatnonzero(pairs[:,0]);used=pairs[indices].copy()
            if int(counters[:,0].sum())!=len(indices) or not (np.all(used[:,1]>=used[:,0]) and np.all(used[:,3]==1)):
                raise RuntimeError('invalid original coordinate stage records')
            arrays.update(stage_header=head.copy(),stage_indices=indices,stage_pairs=used)
            metadata['snapshot_device_ms']=before.elapsed_time(after)
            arrays['metadata']=np.array(json.dumps(metadata))
            np.savez_compressed(directory/f"rank{rank}-{metadata['sequence']:04d}-{metadata['phase']}.npz",**arrays)
            with (out/f'capture-overhead-rank{rank}.jsonl').open('a') as f:
                f.write(json.dumps(dict(sequence=metadata['sequence'],invocation=metadata['invocation'],phase=metadata['phase'],
                    snapshot_device_ms=metadata['snapshot_device_ms'],deferred_cpu_export_ms=(time.monotonic_ns()-start)/1e6))+'\n')
        pending.clear()
        if controls:
            with (out/f'control-rank{rank}.jsonl').open('a') as f:
                for row in controls:f.write(json.dumps(row)+'\n')
            controls.clear()

    def forward(pool,*args,**kwargs):
        request=plan_invocation()
        if request is None:return original_forward(pool,*args,**kwargs)
        if not isinstance(pool.module.op,OperatorScopes):pool.module.op=OperatorScopes(pool.module.op,pool)
        key,recompute=request;contexts.append((key,recompute))
        token=scopes['begin']('MegaMoE.forward',key=key,recompute=recompute);start=time.monotonic_ns()
        try:
            result=original_forward(pool,*args,**kwargs);finish=time.monotonic_ns();scopes['end'](token)
            snapshot(pool,'forward',key,recompute,token,start,finish,pool._plan_bodies[key].body)
            return result
        finally:contexts.pop()

    def backward(pool,*args,**kwargs):
        placement=args[5] if len(args)>5 else kwargs['placement']
        matches=[(key,ticket) for key,ticket in pool._plan_bodies.items() if placement[0].data_ptr()==ticket.logical_ids.data_ptr()]
        if len(matches)!=1:raise RuntimeError('saved backward placement has no unique diagnostic plan identity')
        key,ticket=matches[0];contexts.append((key,False))
        token=scopes['begin']('MegaMoE.backward',key=key,recompute=False);start=time.monotonic_ns()
        original_descriptor=pool.module._descriptor
        def descriptor(plan):
            # Prefix checkpoints can replay two waves before either backward.
            # Reset this backward's reusable coordinate slots on its consuming
            # stream; forward-only reset otherwise merges the two invocations.
            m=pool.module;trace=m.gradient_trace;offset=PREFIX['backward']
            trace[:offset].zero_();trace[offset+HEADER:].zero_()
            trace[offset+7].fill_(plan.epoch)
            return original_descriptor(plan)
        pool.module._descriptor=descriptor
        try:
            result=original_backward(pool,*args,**kwargs);finish=time.monotonic_ns();scopes['end'](token)
            snapshot(pool,'backward',key,False,token,start,finish,ticket.body)
            return result
        finally:
            pool.module._descriptor=original_descriptor
            contexts.pop()

    scopes['register_flusher'](export)
    NativeFSDP2Pool.forward=forward;NativeFSDP2Pool.backward=backward
    NativeFSDP2Pool._publish=publish;CommonOnlinePlanner._compile=compile
    return dict(scopes=scopes,invocation=invocation)
