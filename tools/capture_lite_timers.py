"""Original per-expert TimerV2 on the final update; export after training exits.

No per-step CPU export, new framework Events, coordinate clone or synchronize.
All timings remain rank-local diagnostics; not performance scores.
"""
import atexit
import json
import os
import time
from pathlib import Path
import numpy as np


def install(out:Path,rank:int):
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    from mindspeed_llm.fsdp2.features.ep8_imbalance import plan_invocation
    selected=int(os.environ.get('EP8_LIGHT_CAPTURE_STEP','24'))
    old_forward=NativeFSDP2Pool.forward;old_backward=NativeFSDP2Pool.backward
    pending=[]

    def snapshot(pool,phase,key,recompute,begin,end,body):
        if key[0]!=selected:return
        m=pool.module
        if int(m._worker_native.megaflux_dispatch_stage_schema())!=10:
            raise RuntimeError('Light capture requires original-only TimerV2 schema10')
        if phase not in m.trace_timers:raise RuntimeError('Original timer missing')
        arrays={'original_'+phase:m.trace_timers[phase].detach().clone()}
        if phase=='backward':
            xp=pool.last_x_redispatch_backward_timer
            if xp is None:raise RuntimeError('XP phase timer missing')
            arrays['redispatch_transport']=xp.detach().clone()
        plan=m.last_plan
        for name in ('logical_ids','physical_ids','copies','logical_counts'):
            arrays[name]=getattr(plan,name).detach().clone()
        arrays['source_quota']=body.source_quota.detach().clone()
        arrays['execution_counts']=m.last_exec_counts.detach().clone()
        pending.append((dict(rank=rank,phase=phase,invocation=list(key),recompute=bool(recompute),
            host_begin_ns=begin,host_return_ns=end,coordinate_records=False,diagnostic_only=True,
            clock='rank-local GetSystemCycle; host bounds are enqueue envelopes only',
            capture='final selected update only; clones on consuming stream; export after training'),arrays))

    def forward(pool,*args,**kwargs):
        key,recompute=plan_invocation();begin=time.monotonic_ns()
        result=old_forward(pool,*args,**kwargs)
        if key[0]==selected:
            snapshot(pool,'forward',key,recompute,begin,time.monotonic_ns(),pool._plan_bodies[key].body)
        return result

    def backward(pool,*args,**kwargs):
        placement=args[5] if len(args)>5 else kwargs['placement']
        matches=[(key,ticket) for key,ticket in pool._plan_bodies.items()
                 if placement[0].data_ptr()==ticket.logical_ids.data_ptr()]
        if len(matches)!=1:raise RuntimeError('Ambiguous frozen backward plan')
        key,ticket=matches[0];begin=time.monotonic_ns()
        result=old_backward(pool,*args,**kwargs)
        snapshot(pool,'backward',key,False,begin,time.monotonic_ns(),ticket.body)
        return result

    def export():
        if not pending:return
        directory=out/'timer-light';directory.mkdir(exist_ok=True)
        files=[]
        for i,(metadata,tensors) in enumerate(pending):
            arrays={name:value.cpu().numpy() for name,value in tensors.items()}
            arrays['metadata']=np.array(json.dumps(metadata))
            name=f"rank{rank}-{i:02d}-{metadata['phase']}.npz"
            np.savez_compressed(directory/name,**arrays);files.append(name)
        (out/f'light-capture-rank{rank}.json').write_text(json.dumps(dict(
            selected_step=selected,files=files,diagnostic_only=True,training_performance_score=False),indent=2))
        pending.clear()

    atexit.register(export)
    NativeFSDP2Pool.forward=forward;NativeFSDP2Pool.backward=backward
