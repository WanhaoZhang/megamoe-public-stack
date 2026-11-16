"""Describe observed ready loads; a linear bound is never a speedup result."""
from pathlib import Path
import json
import statistics

root=Path(__file__).resolve().parent
tag='frontier-primary-easy-w2-r4'
rows=[json.loads(line) for line in (root/'evidence'/tag/'temporal-groups-rank0.jsonl').read_text().splitlines()]
records=[]
for row in rows:
    loads=row['loads']
    aggregate=[sum(task[rank] for task in loads) for rank in range(8)]
    serial=sum(max(task) for task in loads)
    records.append(dict(step=row['step'],group=row['group'],ready_tasks=len(row['ready_keys']),
        selected_tasks=len(loads),selected_layers=len({key[2] for key in row['keys']}),
        native_waves=len(row['native_waves']),linear_complementarity=1-max(aggregate)/serial if serial else 0))
result=dict(source_tag=tag,source_version='historical_cyclic_rows; selection metadata only',
    updates=3,decisions=len(records),mixed_layer_decisions=sum(row['selected_layers']>1 for row in records),
    multi_task_decisions=sum(row['selected_tasks']>1 for row in records),
    native_waves=sum(row['native_waves'] for row in records),
    mean_linear_complementarity=statistics.mean(row['linear_complementarity'] for row in records),
    max_linear_complementarity=max(row['linear_complementarity'] for row in records),
    measured_training_speedup=None,device_ready_timeline_verified=False,
    excluded_costs=['prefix probing','weights','token communication','barriers','backward','gradient delivery'],
    records=records)
(root/'opportunity-reference.json').write_text(json.dumps(result,indent=2))
print(json.dumps({key:value for key,value in result.items() if key!='records'}))
