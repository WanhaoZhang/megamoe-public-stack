"""Compact, read-only observation; never creates or admits a device job."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
directory=ROOT/'evidence/diagnostic-fixed-state-pair-aligned-vio5'
rows=[];counts=[]
for rank in range(8):
    p=directory/f'paired-rank{rank}.jsonl'
    values=[json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    counts.append(len(values));rows+=values
status=directory/'status.json'
record=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),paired_steps_by_rank=counts,
    complete=all(x==25 for x in counts),status=json.loads(status.read_text()) if status.exists() else 'running')
if rows:
    record['maxima_so_far']={key:max(x[key] for x in rows) for key in ('loss_relative','fixed_weight_global_norm_relative',
        'complete_parameter_gradient_relative_l2','critical_max_gradient_relative_l2','all_parameter_max_gradient_relative_l2')}
print(json.dumps(record),flush=True)
