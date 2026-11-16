"""Map original assignments to sampled producer/owner edges, without clock stitching."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
_reader=None


def load_calls(baseline,policy,step,layer):
    global _reader
    if _reader is None:
        spec=importlib.util.spec_from_file_location('frozen_timer_reader',baseline/'tools/reduce_diagnostics.py')
        _reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(_reader)
    sys.path.insert(0,str(baseline/'source/native-balanced/ascend_deepep'))
    from megaflux_stage_layout import summary_decode
    with (baseline/'calibration/reduced/calls.csv').open(encoding='utf-8-sig') as f:
        tags={r['tag'] for r in csv.DictReader(f) if r['planner']==policy and float(r['maxvio'])==7
            and int(r['step'])==step and int(r['layer'])==layer and r['phase']=='backward'}
    if len(tags)!=1:raise ValueError((policy,tags))
    matches=[baseline/'evidence'/next(iter(tags))]
    calls={}
    for path in sorted((matches[0]/'timer').glob('rank*-*-backward.npz')):
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            if metadata['invocation']!=[step,layer]:continue
            layout=metadata['layout'];layout['extra_ranges']={int(k):v for k,v in layout['extra_ranges'].items()}
            assert (layout['ep'],layout['home'],layout['slots'],layout['h'],layout['f'])==(8,32,32,7168,2048)
            events=[]
            for index,pair in zip(archive['stage_indices'],archive['stage_pairs']):
                kind,expert,projection,row,column,peer,slot=summary_decode(layout,'backward',int(index))
                if kind in (6,15,16,19,20):
                    events.append(dict(kind=kind,expert=expert,projection=projection,row=row,column=column,
                        begin_tick=int(pair[0]),end_tick=int(pair[1]),lane=int(pair[2])))
            clocks=json.loads((baseline/'calibration/reduced/clock-semantics.json').read_text())['clocks']
            frequency=clocks[str(metadata['rank'])]
            original=_reader.original_records(archive['original_backward'],'backward',frequency)
            origin=min(e['start'] for e in original if e['name']=='kernel')
            end=max(e['end'] for e in original if e['name']=='kernel')
            dw_end=max(e['end'] for e in original if e['name'] in ('dw1','dw2'))
            dx_end=max(e['end'] for e in original if e['name']=='dx')
            owner=[e['end_tick'] for e in events if e['kind']==16]
            timing=dict(rank=metadata['rank'],ticks_per_ms=frequency,local_dw_end_ms=(dw_end-origin)/frequency,
                local_dx_end_ms=(dx_end-origin)/frequency,kernel_end_ms=(end-origin)/frequency,
                owner_complete_ms=(max(owner)-origin)/frequency if owner else None,
                owner_after_last_local_dw_ms=max(0,max(owner)-dw_end)/frequency if owner else None,
                scope='same-rank intervals only; overlap observation, not removable whole-step time')
            calls[metadata['rank']]=dict(path=str(path),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),timing=timing,
                metadata=metadata,events=events,**{name:archive[name].copy() for name in (
                    'logical_ids','physical_ids','copies','source_quota','execution_counts')})
    if set(calls)!=set(range(8)):raise ValueError('Incomplete rank snapshots')
    return calls


def reconstruct(baseline,policy,step,layer):
    calls=load_calls(baseline,policy,step,layer);cp=calls[0]['copies'];q=calls[0]['source_quota']
    home=32;destination=q.sum(axis=0);source=q.sum(axis=2)
    for rank,c in calls.items():
        assert np.array_equal(c['copies'],cp) and np.array_equal(c['source_quota'],q)
        assert np.array_equal(np.bincount(c['logical_ids'].ravel(),minlength=256),source[rank])
        physical=c['physical_ids'].ravel();logical=c['logical_ids'].ravel()
        assert np.all((physical>=0)&(physical<8*64))
        destinations=physical//64;slots=physical%64
        resolved=np.where(slots<home,destinations*home+slots,cp[destinations,np.maximum(slots-home,0)])
        assert np.array_equal(resolved,logical),'Physical destination resolves to a different logical expert'
        actual_quota=np.bincount(logical.astype(np.int64)*8+destinations,minlength=256*8).reshape(256,8)
        assert np.array_equal(actual_quota,q[rank]),'Per-source physical assignments disagree with quota'
        for physical,rows in enumerate(c['execution_counts'].reshape(-1)):
            expert=rank*home+physical if physical<home else int(cp[rank,physical-home])
            assert int(rows)==(0 if expert<0 else int(destination[expert,rank]))
    # The worst sampled owner wait is observable on the owner's own clock.
    # Selecting it does not identify which remote contributor was last.
    waits=[((e['end_tick']-e['begin_tick'])/c['timing']['ticks_per_ms'],rank,e)
        for rank,c in calls.items() for e in c['events'] if e['kind']==19]
    if not waits:raise ValueError('No sampled owner waits')
    _,owner,wait=max(waits,key=lambda x:x[0]);logical=owner*home+wait['expert']
    producers=[]
    for rank in range(8):
        if not destination[logical,rank]:continue
        slot=logical%home if rank==owner else home+int(np.flatnonzero(cp[rank]==logical)[0])
        samples=[e for e in calls[rank]['events'] if e['kind']==6 and e['expert']==slot and
            e['projection']==wait['projection'] and e['row']==wait['row'] and e['column']==wait['column']]
        producers.append(dict(rank=rank,physical_slot=slot,rows=int(destination[logical,rank]),
            source_rows=q[:,logical,rank].tolist(),partial_release_samples=samples,
            snapshot=calls[rank]['path'],edge_to_owner=[owner,logical,wait['projection'],wait['row'],wait['column']]))
    return dict(policy=policy,step=step,layer=layer,plan_assignment_conservation_verified=True,
        active_remote=int((cp>=0).sum()),full_remote_gradient_mib=int((cp>=0).sum())*84,
        active_physical_experts=int((destination>0).sum()),
        nonzero_gradient_output_tiles=int((destination>0).sum())*(896+448),
        row_tiles128_total=int(((destination+127)//128).sum()),
        row_tiles192_total=int(((destination+191)//192).sum()),
        selected_sample=dict(selection='maximum observed owner tile wait, not a certified whole-step critical path',
            owner_rank=owner,logical_expert=logical,owner_wait=wait,contributors=producers),
        clock_rule='Raw ticks belong only to the named rank. Edges express dependencies, not synchronized timestamps. No max across producer clocks.',
        last_remote_contributor_identified=False,
        rank_local_tail_observations=[c['timing'] for c in calls.values()],
        limitation='Sparse column0 samples; no per-peer wait span here. Full contributor attribution requires additional measured owner probes.',
        snapshots={str(r):dict(path=c['path'],sha256=c['file_sha256']) for r,c in calls.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--step',type=int,default=1);p.add_argument('--layer',type=int,default=0)
    p.add_argument('--all-postwarmup',action='store_true');a=p.parse_args()
    points=[(s,l) for s in (1,2,3) for l in (0,1)] if a.all_postwarmup else [(a.step,a.layer)]
    records=[reconstruct(a.baseline,policy,s,l) for s,l in points for policy in ('native','ultra_ep','moonep','echo','eplb_fixed')]
    (ROOT/'analysis/original-contributor-dependencies.json').write_text(json.dumps(dict(diagnostic_only=True,records=records),indent=2))
    print(json.dumps([dict(policy=r['policy'],active_remote=r['active_remote'],
        owner=r['selected_sample']['owner_rank'],logical=r['selected_sample']['logical_expert'],
        contributors=len(r['selected_sample']['contributors']),last_contributor_identified=False) for r in records],indent=2))


if __name__=='__main__':main()
