"""Keep old gate scope separate from stricter per-parameter evidence."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    rows=[]
    for path in sorted(a.run.glob('paired-rank*.jsonl')):
        rank=int(path.stem.split('rank')[1])
        for line in path.read_text().splitlines():
            row=json.loads(line)
            for name,metric in row.get('parameter_gradient_metrics',{}).items():
                rows.append(dict(rank=rank,step=row['step'],parameter=name,
                    relative_l2=metric['relative_l2'],reference_norm=math.sqrt(metric['reference_energy']),
                    difference_norm=math.sqrt(metric['difference_energy'])))
    if not rows:raise ValueError('No per-parameter energy evidence')
    rows.sort(key=lambda r:r['relative_l2'],reverse=True)
    result=dict(scope='Measured same-weight comparison, not convergence or a relaxed numerical gate',
        strict_all_parameter_relative_l2_le_002=all(r['relative_l2']<=.02 for r in rows),
        comparisons=len(rows),largest_relative_errors=rows[:30])
    (a.run/'parameter-error-analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
