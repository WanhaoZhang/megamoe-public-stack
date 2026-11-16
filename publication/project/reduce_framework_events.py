"""Read stream-local Event envelopes without adding nested or cross-rank scopes."""
from pathlib import Path
import csv
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
out=root/'diagnostics';out.mkdir(exist_ok=True)
tags=('timer-v2-matched-primary','timer-v2-matched-legacy-primary')

def kind(row):
    name=row['name']
    for exact in ('ready_frontier.prefix_probe','ready_frontier.device_selection',
                  'online_plan_and_common_validation','framework_owner_to_bf16_mirror',
                  'optimizer.global_clip','optimizer.AdamW'):
        if name==exact:return exact
    if name in ('operator.forward','operator.backward'):
        return name+('.replay' if row['recompute'] else '.original')
    if name.endswith('.self_attn.forward'):
        return 'Attention.forward'+('.replay' if row['recompute'] else '.original_and_probe')
    if name.endswith('.shared_experts.forward'):
        return 'Shared.forward'+('.replay' if row['recompute'] else '.original')
    if name.startswith('FSDP.unshard') and name.endswith('.all_gather_stream'):
        return 'FSDP.all_gather_stream_envelope'
    if name.startswith('FSDP.post_backward') and name.endswith('.reduce_scatter_stream'):
        return 'FSDP.reduce_scatter_stream_envelope'

def union_ms(rows):
    intervals=sorted((row['start_ms'],row['end_ms']) for row in rows)
    total=0.;end=float('-inf')
    for begin,finish in intervals:
        total+=max(0.,finish-max(begin,end));end=max(end,finish)
    return total

summaries=[]
for tag in tags:
    run=root/'evidence'/tag
    if not (run/'status.json').exists():continue
    status=json.loads((run/'status.json').read_text())
    if not status['qualified'] or not status['source_unchanged']:continue
    policy=json.loads((run/'temporal-audit-rank0.json').read_text())['mode']
    for rank in range(8):
        rows=[json.loads(line) for line in (run/f'framework-events-rank{rank}.jsonl').read_text().splitlines()]
        selected=[row for row in rows if row['step']==2 and not row['initialization'] and not row['failed'] and kind(row)]
        for label in sorted({kind(row) for row in selected}):
            subset=[row for row in selected if kind(row)==label]
            summaries.append(dict(tag=tag,policy=policy,rank=rank,step=2,category=label,
                calls=len(subset),union_ms=union_ms(subset),sum_envelopes_ms=sum(row['elapsed_ms'] for row in subset),
                semantics='not additive across categories; stream envelopes, not kernel active time'))
        if selected:
            categories=sorted({kind(row) for row in selected})
            ids={name:index for index,name in enumerate(categories)}
            origin=min(row['start_ms'] for row in selected)
            trace=[dict(ph='M',pid=rank,tid=index,name='thread_name',args=dict(name=name)) for name,index in ids.items()]
            trace.extend(dict(ph='X',pid=rank,tid=ids[kind(row)],name=row['name'],
                ts=(row['start_ms']-origin)*1000,dur=(row['end_ms']-row['start_ms'])*1000,
                args={key:row[key] for key in ('invocation','layer','recompute','stream','semantics')}) for row in selected)
            (out/f'{tag}-rank{rank}-framework.json').write_text(json.dumps(dict(traceEvents=trace,
                displayTimeUnit='ms',clock_note='single rank local Event origin only; do not merge rank zeros')))
        if rank==0 and selected:
            fig,axis=plt.subplots(figsize=(15,6.5))
            categories=sorted({kind(row) for row in selected})
            origin=min(row['start_ms'] for row in selected)
            for index,label in enumerate(categories):
                subset=[row for row in selected if kind(row)==label]
                axis.broken_barh([(row['start_ms']-origin,row['end_ms']-row['start_ms']) for row in subset],
                                 (index-.35,.7),facecolors=plt.get_cmap('tab20')(index%20))
            axis.set(yticks=range(len(categories)),yticklabels=categories,xlabel='Rank0 local Event time from first selected scope (ms)',
                title=f'{policy}: update2 diagnostic stream envelopes | one rank only\nNested categories overlap; intervals include dependencies and host supply gaps')
            axis.grid(axis='x',alpha=.2);fig.tight_layout()
            for suffix in ('png','svg','pdf'):fig.savefig(out/f'{tag}-rank0-framework.{suffix}',dpi=180,bbox_inches='tight')
            plt.close(fig)
if summaries:
    with (out/'framework-summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    aggregate=[]
    for policy in sorted({row['policy'] for row in summaries}):
        for category in sorted({row['category'] for row in summaries if row['policy']==policy}):
            rows=[row for row in summaries if row['policy']==policy and row['category']==category]
            aggregate.append(dict(policy=policy,category=category,ranks=len(rows),calls_per_rank=[row['calls'] for row in rows],
                median_union_ms=statistics.median(row['union_ms'] for row in rows),
                max_union_ms=max(row['union_ms'] for row in rows)))
    (out/'framework-summary.json').write_text(json.dumps(dict(diagnostic_only=True,update=2,
        no_cross_rank_clock_alignment=True,categories_not_additive=True,aggregates=aggregate),indent=2))
print(json.dumps(dict(rows=len(summaries),output=str(out))))
