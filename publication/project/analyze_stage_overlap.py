"""Same-device schema9 evidence of sampled owner service during dW production.

No cross-rank clock alignment; no conversion of uncalibrated cycles to ms.
Sampled column0 row-bin records do not cover every owner tile. This diagnoses
the original pipeline mechanism, not full-update exposure or a speedup.
"""
from pathlib import Path
import argparse,json,statistics
import numpy as np


def union(values):
    result=[]
    for first,last in sorted(values):
        if result and first<=result[-1][1]:result[-1][1]=max(last,result[-1][1])
        else:result.append([first,last])
    return result


def analyze(folders):
    records=[]
    for folder in folders:
        for path in sorted((folder/'timer').glob('*backward.npz')):
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']));layout=meta['layout']
                if layout.get('version')!=9:continue
                indices=data['stage_indices'];pairs=data['stage_pairs']
                def kind(number):
                    start,size=layout['extra_ranges'][str(number)]
                    return pairs[(indices>=start)&(indices<start+size),:2]
                release,service=kind(6),kind(20)
                if not len(release):continue
                last=int(release[:,1].max());intervals=union(service.tolist())
                duration=sum(end-start for start,end in intervals)
                overlap=sum(max(0,min(end,last)-start) for start,end in intervals)
                records.append(dict(file=str(path),rank=meta['rank'],invocation=meta['invocation'],
                    service_sample_count=len(service),release_sample_count=len(release),
                    sampled_owner_service_before_last_local_dW_release_cycles=overlap,
                    sampled_owner_service_union_cycles=duration,
                    sampled_service_before_last_local_release_fraction=overlap/duration if duration else None))
    active=[row for row in records if row['sampled_owner_service_union_cycles']]
    return dict(records=records,backward_records_with_dW_release=len(records),
        records_with_sampled_owner_service=len(active),
        service_starts_before_last_local_release_records=sum(row['sampled_owner_service_before_last_local_dW_release_cycles']>0 for row in active),
        median_sampled_service_before_last_local_release_fraction=statistics.median(
            row['sampled_service_before_last_local_release_fraction'] for row in active) if active else None,
        scope='rank-local cycles only; column0 sampled owner service overlaps ongoing local dW publication; not full kernel coverage or complete-update exposed time')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();result=analyze(args.folders);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='records'}))
