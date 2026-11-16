"""Compare numerical isolations and rank-local backward readiness order."""
from pathlib import Path
import json

root=Path(__file__).resolve().parent
state=json.loads((root/'prefix-failure-diagnostics.json').read_text())
records=[]
for case in state['cases']:
    record={key:case[key] for key in ('label','tag','sync','offline','status')}
    comparison=root/'evidence'/(case['tag']+'-comparison.json')
    if comparison.exists():
        value=json.loads(comparison.read_text())
        record.update(identities_equal=value['identities_equal'],
            category_max_relative_l2=max(row['relative_l2'] for row in value['categories'].values()),
            parameter_max_relative_l2=max(row['relative_l2'] for row in value['parameters'].values()),
            all_gradients_pass=all(row['passed'] for row in value['categories'].values()) and
                               all(row['passed'] for row in value['parameters'].values()))
    path=root/'evidence'/case['tag']
    if (path/'temporal-audit-rank0.json').exists():
        windows=[];replay=[];forward=[]
        for rank in range(8):
            audit=json.loads((path/f'temporal-audit-rank{rank}.json').read_text())
            ready=[];recompute=[];creation=[]
            for window in audit['diagnostic_stage_windows']:
                key=(window['step'],tuple(window['microbatches']))
                seen=set();first=[]
                for row in window['ready']:
                    if row['index'] not in seen:seen.add(row['index']);first.append(row['index'])
                ready.append((key,first));recompute.append((key,window['recompute']))
                creation.append((key,window['forward']))
            windows.append(ready);replay.append(recompute);forward.append(creation)
        record.update(stage_creation_same_all_ranks=all(value==forward[0] for value in forward),
            first_ready_stage_order_same_all_ranks=all(value==windows[0] for value in windows),
            recompute_entry_order_same_all_ranks=all(value==replay[0] for value in replay),
            rank_first_ready_order=windows,rank_recompute_entry_order=replay,
            scope='Tensor readiness and recompute entry, not actual communication issue order')
    records.append(record)
result=dict(role='small numerical diagnostics, excluded from all performance results',records=records,
    new_planner=False,root_cause_confirmed=False,qualified_runtime_fix=False,
    complete=all(case['status'] in ('passed_numeric','failed_numeric') for case in state['cases']))
(root/'prefix-failure-diagnostics-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{key:value for key,value in row.items() if not key.startswith('rank_')} for row in records]))
