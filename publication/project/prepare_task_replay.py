"""Strip observed traces down to an explicitly offline task-selection control."""
from pathlib import Path
import json

root=Path(__file__).resolve().parent
out=root/'diagnostics';out.mkdir(exist_ok=True)
for name,tag in [('small','ordered-shared-easy_home-w2'),('primary','pilot-clean-timing-random-easy_home-w2')]:
    run=root/'evidence'/tag
    state=json.loads((run/'status.json').read_text())
    if not state['qualified'] or not state['source_unchanged']:raise RuntimeError('Unqualified source schedule')
    config=json.loads((run/'effective-config.json').read_text())
    sources=[[json.loads(line) for line in (run/f'temporal-groups-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
    groups=[{key:row[key] for key in ('step','group','keys','ready_keys')} for row in sources[0]]
    if any([{key:row[key] for key in ('step','group','keys','ready_keys')} for row in rows]!=groups for rows in sources[1:]):
        raise RuntimeError('Source ranks did not agree on task schedule')
    model=config['model']
    data=dict(version=1,role='offline_task_selection_diagnostic',source_tag=tag,
        model_shape=dict(layers=config['layers'],tokens=config['tokens'],hidden=model['hidden_size'],
                         ffn=model['moe_intermediate_size'],topk=model['num_experts_per_tok']),
        steps=config['steps'],window=config['forward_window'],microbatches=config['microbatches'],
        uses_future_derived_schedule=True,groups=groups)
    path=out/f'offline-{name}-schedule.json';path.write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(path=str(path),groups=len(groups),online_score_allowed=False)))
