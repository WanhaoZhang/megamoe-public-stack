"""Reduce same-device Event scopes, without summing nested or replay spans.

Autograd module backward scopes can include checkpoint recomputation. Export
both the complete dependency envelope and its measured non-replay remainder.
Neither quantity is labelled kernel-active time or a global cross-rank clock.
"""
import argparse
import csv
import json
from pathlib import Path
import re


def union_length(intervals,lo,hi):
    pairs=sorted((max(lo,a),min(hi,b)) for a,b in intervals if b>lo and a<hi)
    end=lo;total=0.
    for a,b in pairs:
        if b>a:total+=max(0.,b-max(a,end));end=max(end,b)
    return total


def category(name):
    if name=='online_plan_and_common_validation':return 'control.online_planning'
    if name=='framework_owner_to_bf16_mirror':return 'control.owner_publish'
    if name.startswith('operator.'):return name
    if name.startswith('MegaMoE.'):return name
    if name.startswith('FSDP.'):return 'FSDP.'+name.split('.',2)[1].split(':',1)[0]
    if name.startswith('optimizer.'):return name
    if '.self_attn.' in name:return 'attention.'+name.rsplit('.',1)[1]
    if '.shared_experts.' in name:return 'shared.'+name.rsplit('.',1)[1]
    if re.search(r'\.layers\.\d+\.(forward|backward)$',name):return 'layer.'+name.rsplit('.',1)[1]
    return 'framework_other.'+name.rsplit('.',1)[-1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--tag',action='append',required=True);a=ap.parse_args();output=a.root/'calibration/framework';output.mkdir(parents=True,exist_ok=True)
    rows=[];calls=[];controls=[];synchronization=[]
    for tag in a.tag:
        directory=a.root/'evidence'/tag;cfg=json.loads((directory/'effective-config.json').read_text())
        for rank in range(8):
            events=[json.loads(x) for x in (directory/f'framework-events-rank{rank}.jsonl').read_text().splitlines()]
            by_sequence={r['sequence']:r for r in events}
            if len(events)!=len(by_sequence):raise ValueError('duplicate diagnostic Event identity')
            trace=[]
            for r in events:
                if r['end_ms']<r['start_ms'] or r['failed']:raise ValueError('failed or invalid measured Event scope')
                peers=[x for x in events if x['step']==r['step'] and x['stream']==r['stream'] and x['recompute']]
                replay_intersection=union_length([(x['start_ms'],x['end_ms']) for x in peers],r['start_ms'],r['end_ms'])
                children=[x for x in events if x['sequence']!=r['sequence'] and x['stream']==r['stream'] and
                    x['step']==r['step'] and x['host_begin_ns']>=r['host_begin_ns'] and x['host_end_ns']<=r['host_end_ns']]
                child_union=union_length([(x['start_ms'],x['end_ms']) for x in children],r['start_ms'],r['end_ms'])
                record=dict(tag=tag,planner=cfg['planner'],maxvio=cfg['maxvio'],rank=rank,step=r['step'],layer=r['layer'],
                    recompute=r['recompute'],name=r['name'],category=category(r['name']),sequence=r['sequence'],stream=r['stream'],
                    start_ms=r['start_ms'],end_ms=r['end_ms'],envelope_ms=r['elapsed_ms'],
                    same_stream_nested_union_ms=child_union,exclusive_envelope_ms=max(0.,r['elapsed_ms']-child_union),
                    checkpoint_replay_intersection_ms=replay_intersection,
                    non_replay_envelope_ms=max(0.,r['elapsed_ms']-replay_intersection),
                    checkpoint_early_stop=r.get('checkpoint_early_stop',False),diagnostic_only=True)
                rows.append(record)
                trace.append(dict(ph='X',pid=rank,tid=r['stream'],name=r['name'],ts=r['start_ms']*1000,
                    dur=r['elapsed_ms']*1000,args=dict(step=r['step'],layer=r['layer'],recompute=r['recompute'],
                    semantics='same-device Event envelope, includes dependencies and host supply gaps; not kernel active time')))
            for control in [json.loads(x) for x in (directory/f'control-rank{rank}.jsonl').read_text().splitlines()]:
                event=by_sequence[control['event_sequence']]
                controls.append(dict(tag=tag,planner=cfg['planner'],maxvio=cfg['maxvio'],**control,
                    current_stream_ms=event['elapsed_ms'],event_start_ms=event['start_ms'],event_end_ms=event['end_ms']))
            for phase in ('forward','backward'):
                for step in range(4):
                    for layer in range(2):
                        for recompute in ([False,True] if phase=='forward' else [False]):
                            pool=[r for r in events if r['name']=='MegaMoE.'+phase and r['step']==step and r['layer']==layer and r['recompute']==recompute]
                            native=[r for r in events if r['name']=='operator.'+phase and r['step']==step and r['layer']==layer and r['recompute']==recompute]
                            if len(pool)!=1 or len(native)!=1:raise ValueError('missing/duplicate operator invocation Event scope')
                            calls.append(dict(tag=tag,rank=rank,step=step,layer=layer,phase=phase,recompute=recompute,
                                pool_sequence=pool[0]['sequence'],operator_sequence=native[0]['sequence'],
                                pool_event_ms=pool[0]['elapsed_ms'],operator_event_ms=native[0]['elapsed_ms']))
            (output/f'{tag}-rank{rank}-framework-trace.json').write_text(json.dumps(dict(traceEvents=trace,displayTimeUnit='ms')))
    for stem,data in [('events',rows),('controls',controls),('invocations',calls)]:
        keys=list(dict.fromkeys(key for row in data for key in row))
        with (output/(stem+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
    meta=dict(schema=1,tags=a.tag,ranks=8,records=len(rows),invocations=len(calls),control_records=len(controls),
        clock='independent same-device Event origin per rank; no synchronized cross-rank timeline',
        module_backward='includes saved-tensor recomputation; measured replay union is exported separately',
        nested_events='union subtraction, never sum of nested intervals',diagnostic_only=True)
    (output/'semantics.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))


if __name__=='__main__':
    assert union_length([(1,3),(2,4),(5,8)],0,6)==4
    main()
