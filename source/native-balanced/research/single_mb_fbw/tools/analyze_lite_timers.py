"""Rank-local original TimerV2 scopes, without inventing synchronized clocks.

Default units are device ticks. A new-node clock calibration is required for
milliseconds; old-machine calibration is never silently reused.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from verify_training_run import verify,verify_loaded_worker

ROOT=Path(__file__).resolve().parents[1]


def original_dependency_waits(raw,counts,units=1.):
    """Original Overwrite slots, not coordinate records or accumulated ticks.

    Bases21/85/149 are verified against backward_catlass_impl.h. The frozen
    reader only decodes fixed0..20 and dynamic kinds11..19, missing these.
    """
    counters=((10+20*64)*2+15)//16*16
    values=np.asarray(raw).reshape(-1)
    if values.size<counters*96:raise ValueError('Truncated original TimerV2')
    values=values[:counters*96].reshape(96,counters)
    rows=[];totals={}
    for name,base,parity in (('aic_dz_ready_wait',21,0),('aiv_dx_ready_wait',85,1),('aic_do_reuse_wait',149,0)):
        by_lane={}
        for lane in range(96):
            if lane%3!=parity:continue
            spans=[]
            for expert in range(min(64,len(counts))):
                if counts[expert]<=0:continue
                begin,end=map(int,values[lane,(base+expert)*2:(base+expert)*2+2])
                if not begin:continue
                if end<begin:raise ValueError('Invalid original dependency wait')
                spans.append((begin,end));rows.append(dict(name=name,lane=lane,expert=expert,
                    begin_tick=begin,end_tick=end,duration=(end-begin)/units))
            spans.sort()
            if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
                raise ValueError('Sequential waits overlap on one lane')
            by_lane[str(lane)]=sum(end-begin for begin,end in spans)/units
        totals[name]=dict(per_lane=by_lane,mean_lane_sum=float(np.mean(list(by_lane.values()))),
                         max_lane_sum=max(by_lane.values()))
    return dict(intervals=rows,sequential_lane_totals=totals,
                scope='Nonoverlapping original dependency waits summed only within one hardware lane')


def bounds(events,name,units=1.):
    rows=[r for r in events if r['name']==name and not r.get('accumulated',False)]
    if not rows:return None
    begin=min(r['start'] for r in rows);end=max(r['end'] for r in rows)
    return dict(begin_tick=begin,end_tick=end,envelope=(end-begin)/units,
        mean_lane_interval=float(np.mean([(r['end']-r['start'])/units for r in rows])),lanes=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen-baseline',required=True,type=Path)
    p.add_argument('--clocks',type=Path);p.add_argument('--qualification',type=Path)
    p.add_argument('--partial',action='store_true');a=p.parse_args()
    code=a.frozen_baseline/'tools/reduce_diagnostics.py'
    spec=importlib.util.spec_from_file_location('original_timer_reader',code);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    clocks=json.loads(a.clocks.read_text())['clocks'] if a.clocks else None
    rows=json.loads((a.qualification or ROOT/'evidence/light-timer-qualification-v1.json').read_text())
    if (not a.partial and len(rows)!=6) or not 1<=len(rows)<=6:
        raise RuntimeError('Expected matched0/1 three-policy diagnostics, or explicit partial audit')
    caps=json.loads((ROOT/'evidence/fast-redispatch-capabilities-v1.json').read_text())
    shas={r['variant']:r['so_sha256'] for r in caps};records=[];identities=None;snapshots={}
    for row in rows:
        root=ROOT/'evidence'/row['tag'];checked=verify(root)
        current=[r['identities'] for r in checked]
        if identities is None:identities=current
        if current!=identities:raise RuntimeError('Diagnostic initial conditions differ')
        loaded=next(iter(checked[0]['workers']));worker=next(n for n in shas if f'worker-variants/{n}/' in loaded)
        verify_loaded_worker(checked,worker,shas[worker]);cache=int(worker[-1]);policy=row['tag'].split('vio20-',1)[1].rsplit('-cache',1)[0]
        paths=sorted((root/'timer-light').glob('*.npz'))
        if len(paths)!=48:raise RuntimeError(f'Expected six snapshots per rank: {root}')
        for path in paths:
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']));rank=meta['rank'];phase=meta['phase'];units=float(clocks[str(rank)]) if clocks else 1.
                if meta['invocation'][0]!=24 or meta['coordinate_records']:raise RuntimeError('Unexpected capture scope')
                events=reader.original_records(z['original_'+phase],phase,1.)
                names=('kernel','aic_init','aiv_init','dh','swiglu_grad','dx','dw1','dw2','unpermute','topk_unpermute') if phase=='backward' else ('kernel','init_param','gmm1','activation','gmm2','unpermute')
                phases={n:bounds(events,n,units) for n in names}
                plan={n:hashlib.sha256(z[n].tobytes()).hexdigest() for n in ('copies','source_quota','physical_ids','execution_counts')}
                key=(policy,rank,tuple(meta['invocation']),phase,bool(meta['recompute']))
                if plan!=snapshots.setdefault(key,plan):raise RuntimeError('Metadata intervention changed the captured plan')
                xp={}
                if phase=='backward':
                    dependency_waits=original_dependency_waits(z['original_'+phase],z['execution_counts'].reshape(-1),units)
                    raw=z['redispatch_transport']
                    if raw.shape!=(4,128,2):raise RuntimeError('XP timer ABI changed')
                    for i,label in enumerate(('source_prepare','source_publish','xp_transport','xp_retirement')):
                        valid=raw[i][(raw[i,:,0]>0)&(raw[i,:,1]>=raw[i,:,0])]
                        if not len(valid):raise RuntimeError('Missing original XP phase')
                        xp[label]=dict(lanes=len(valid),mean_lane_interval=float(np.mean((valid[:,1]-valid[:,0])/units)),
                            begin_tick=int(valid[:,0].min()),end_tick=int(valid[:,1].max()))
                rec=dict(tag=row['tag'],rank=rank,cache=cache,policy=policy,invocation=meta['invocation'],phase=phase,
                    recompute=meta['recompute'],scopes=phases,transport=xp,snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if phase=='backward':
                    rec['dependency_waits']=dependency_waits
                    end=phases['kernel']['end_tick']
                    rec['after_dx_gemm']=(end-phases['dx']['end_tick'])/units
                    rec['after_unpermuted_dx']=(end-phases['unpermute']['end_tick'])/units
                    rec['dw1_envelope']=phases['dw1']['envelope']
                records.append(rec)
    result=dict(diagnostic_only=True,partial=a.partial,diagnostic_processes=len(rows),
        units='milliseconds' if clocks else 'GetSystemCycle ticks',records=records,
        reader_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
        scope='Rank-local original TimerV2 scopes; no coordinate records; final update only',
        caveats=['Interval envelopes include waits and are not arithmetic latency','Do not sum overlapping phases or concatenate independent rank clocks',
                 'After-dX-GEMM interval is not final input-gradient readiness or removable whole-step time',
                 'Cross-layer deferral additionally needs legal graph, memory leases and hardware resource scheduling'])
    output='light-timer-scopes-partial-v1.json' if a.partial else 'light-timer-scopes-v1.json'
    (ROOT/'analysis'/output).write_text(json.dumps(result,indent=2));print('LIGHT_TIMER_SCOPE_AUDIT_PASS',len(records),result['units'])


if __name__=='__main__':main()
