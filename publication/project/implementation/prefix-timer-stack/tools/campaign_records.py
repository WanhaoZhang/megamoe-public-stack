"""Resolve explicitly linked pre-training infrastructure retries, retaining failures.

No valid successful/slow timed run is replaceable. A recorded zero-step HCCL
initializer failure or an independently evidenced foreign-GPU activity window
can be retried under unchanged sources, retaining all original measurements.
"""
import json
import hashlib
from pathlib import Path

def rows_for(root:Path,names):
    return [row for name in names if (path:=root/f'evidence/campaign-{name}.json').exists() for row in json.loads(path.read_text())]

def init_failure(root,row):
    directory=root/'evidence'/row['tag']
    text=(directory/'stdout.log').read_text(errors='replace')
    steps=list(directory.glob('steps-rank*.jsonl'))
    return (not row['qualified'] and row.get('rank_completions',0)==0 and not any(p.stat().st_size for p in steps)
        and row.get('source_unchanged',False) and 'HcclBroadcast' in text and 'RunAicpuIndOpThreadInit' in text
        and 'WeightLoader.load' in text)

def verified_contention(root,row):
    direct=row.get('foreign_gpu_observation')
    if direct:
        line=(root/direct['file']).read_text().splitlines()[direct['line']-1]
        if hashlib.sha256(line.encode()).hexdigest()!=direct['sha256']:
            raise RuntimeError('direct foreign-process evidence changed')
        observation=json.loads(line)
        if not row['start']<=observation['timestamp']<=row['finish']:
            raise RuntimeError('direct foreign observation is outside the run interval')
        matches=[p for p in observation['foreign_or_unidentified'] if p['pid']==direct['pid']
                 and p.get('cgroup')==direct['cgroup'] and not p.get('task_container')
                 and not p.get('task_script') and not p.get('process_disappeared')]
        if len(matches)!=1:raise RuntimeError('no actual foreign GPU PID in direct observation')
        return True
    path=root/'evidence/EXTERNAL-CONTENTION.json'
    if not path.exists():return False
    ledger=json.loads(path.read_text())
    matching=[x for x in ledger['intervals'] if row['start']<x['end'] and row['finish']>x['begin']]
    if not matching:return False
    for interval in matching:
        proof=interval['proof'];line=(root/proof['file']).read_text().splitlines()[proof['line']-1]
        if hashlib.sha256(line.encode()).hexdigest()!=proof['sha256']:raise RuntimeError('foreign process evidence changed')
        observed=json.loads(line)
        if not any(interval['container'] in p.get('cgroup','') for p in observed['foreign_or_unidentified']):raise RuntimeError('no foreign-container GPU PID in the cited proof')
    return True

def resolved_campaign(root:Path,name,expected):
    original=rows_for(root,[name])
    if len(original)!=expected:raise RuntimeError(f'not all {expected} original {name} attempts are complete')
    recoveries=rows_for(root,[name+'-recovery']);resolved=[]
    for row in original:
        contaminated=verified_contention(root,row)
        if row['state']=='pass' and not contaminated:
            if any(r.get('retry_of')==row['tag'] and r['qualified'] for r in recoveries):raise RuntimeError('a successful run was improperly retried')
            resolved.append(row);continue
        if not contaminated and not init_failure(root,row):raise RuntimeError('a policy/training failure is not replaceable by infrastructure recovery: '+row['tag'])
        candidates=[r for r in recoveries if r.get('retry_of')==row['tag'] and r['state']=='pass' and not verified_contention(root,r)]
        if len(candidates)!=1:raise RuntimeError('initialization failure needs exactly one explicit successful linked retry: '+row['tag'])
        chosen=candidates[0]
        if any(chosen[k]!=row[k] for k in ('planner','repeat','maxvio')):raise RuntimeError('retry changed planned conditions')
        if json.loads((root/f'evidence/{chosen["tag"]}/source-identity.json').read_text())!=json.loads((root/f'evidence/{row["tag"]}/source-identity.json').read_text()):raise RuntimeError('retry source differs from the original attempt')
        resolved.append(chosen)
    if len(resolved)!=expected:raise RuntimeError('not all planned cells resolved')
    return resolved

def resolved_existing(root:Path):return resolved_campaign(root,'primary-existing',90)
