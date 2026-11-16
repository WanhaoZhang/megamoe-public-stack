"""Keep small-gradient conditioning separate from model correctness claims."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    records=[];inputs={}
    for tag in ('diagnostic-native-native-pair-vio5-v1','diagnostic-moonep-native-pair-vio5-v1',
                'diagnostic-incoming-native-self-vio5-v1'):
        rows=[]
        for path in sorted((ROOT/'evidence'/tag).glob('paired-rank*.jsonl')):
            inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            rank=int(path.stem.split('rank')[1])
            for line in path.read_text().splitlines():
                step=json.loads(line)
                for name,metric in step.get('parameter_gradient_metrics',{}).items():
                    rows.append(dict(rank=rank,step=step['step'],parameter=name,
                        group='HC' if ('_hc.' in name or 'hc_head.' in name) else 'other',**metric))
        if not rows:raise RuntimeError('Missing per-parameter diagnostic')
        groups=[]
        for group in ('HC','other'):
            items=[r for r in rows if r['group']==group]
            groups.append(dict(group=group,comparisons=len(items),
                above_relative_002=sum(r['relative_l2']>.02 for r in items),
                max_relative=max(r['relative_l2'] for r in items),
                reference_energy=sum(r['reference_energy'] for r in items),
                difference_energy=sum(r['difference_energy'] for r in items)))
        worst=sorted(rows,key=lambda r:r['relative_l2'],reverse=True)[:10]
        for row in worst:
            row['reference_norm']=math.sqrt(row['reference_energy'])
            row['difference_norm']=math.sqrt(row['difference_energy'])
        records.append(dict(tag=tag,groups=groups,worst=worst))
    result=dict(scope='Existing diagnostic per-parameter energies only; no changed acceptance gate',
        records=records,input_sha256=inputs,
        caveats=['Relative errors of near-zero gradients can be ill-conditioned',
                 'Small absolute gradients do not establish small Adam update error',
                 'Nonzero incoming dY before earlier MoE excludes an entirely local MoE explanation, not every MoE contribution',
                 'Source-level attention nondeterminism is a hypothesis pending controlled NPU diagnostics'])
    (ROOT/'analysis/numerical-error-provenance.json').write_text(json.dumps(result,indent=2))
    print('NUMERICAL_ERROR_PROVENANCE_AUDIT',len(records))

if __name__=='__main__':main()
