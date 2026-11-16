"""Audit full-update records and immutable initial conditions before comparisons."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    status=json.loads((root/'status.json').read_text())
    if not status['qualified'] or not status['source_unchanged']:raise ValueError('Unqualified run')
    result=[]
    for rank in range(8):
        done=json.loads((root/f'complete-rank{rank}.json').read_text())
        steps=[json.loads(l) for l in (root/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        if done['exit']!=0 or done['steps']!=25 or [r['step'] for r in steps]!=list(range(25)):
            raise ValueError('Incomplete or reordered full updates')
        audit=json.loads((root/f'planning-audit-rank{rank}.json').read_text())
        if sum(a['calls'] for a in audit)!=50 or any(a['live'] for a in audit):raise ValueError('Plan lifetime audit')
        identities={}
        for label,name in [('parameters','initial'),('inputs','input-identity'),('routes','route-identity')]:
            data=json.loads((root/f'{name}-rank{rank}.json').read_text())
            identities[label]=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
        libraries=json.loads((root/f'libraries-rank{rank}.json').read_text())['libraries']
        workers={k:v for k,v in libraries.items() if 'worker-variants/' in k}
        if len(workers)!=1:raise ValueError(f'Loaded worker identity not recorded: {list(workers)}')
        result.append(dict(rank=rank,identities=identities,workers=workers))
    return result


def verify_loaded_worker(checked,worker,expected_sha):
    """Compare the measured process mapping, not just its launch environment."""
    for rank in checked:
        loaded=rank['workers']
        if len(loaded)!=1:raise ValueError('Expected one loaded worker per rank')
        for path,metadata in loaded.items():
            if f'worker-variants/{worker}/' not in path or metadata.get('sha256')!=expected_sha:
                raise ValueError('Actually loaded worker differs from qualified build')


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);a=p.parse_args();reference=None
    for path in a.runs:
        result=verify(path)
        if reference is None:reference=result
        for left,right in zip(reference,result):
            if left['identities']!=right['identities']:raise ValueError(f'Initial condition mismatch: {path}')
        (path/'comparison-audit.json').write_text(json.dumps(dict(pass_=True,ranks=result),indent=2))
        print(path.name,'FULL_UPDATE_IDENTITY_PASS',flush=True)


if __name__=='__main__':main()
