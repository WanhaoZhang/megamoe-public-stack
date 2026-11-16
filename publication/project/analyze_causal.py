"""Read completed diagnostic runs; keep per-rank clock and scope semantics."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,json,csv,statistics

root=Path(__file__).resolve().parent


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def union(intervals):
    result=[]
    for first,last in sorted(intervals):
        if result and first<=result[-1][1]:result[-1][1]=max(result[-1][1],last)
        else:result.append([first,last])
    return result


def category(row):
    name=row['name']
    if name=='operator.forward':return 'MoE recompute' if row['recompute'] else 'MoE forward'
    if name.endswith('.self_attn.forward'):return 'Attention recompute' if row['recompute'] else 'Attention forward'
    if name.endswith('.self_attn.backward'):return 'Attention backward'
    if name.endswith('.lm_head.forward'):return 'LM head forward'
    if name.endswith('.lm_head.backward'):return 'LM head backward'
    return {'operator.backward':'MoE backward','framework.owner_compute_materialization':'owner BF16 materialization',
        'framework_owner_to_bf16_mirror':'publish home image','lifetime.gradients':'raw gradient adapter',
        'lifetime.service_work':'gradient service','lifetime.service_wait_for_producer':'service producer wait',
        'lifetime._drain':'gradient drain','optimizer.global_clip':'global clip',
        'optimizer.AdamW':'AdamW','online_plan_and_common_validation':'online plan + validation'}.get(name)


def analyze(prefix):
    state=json.loads((root/(prefix+'.json')).read_text())
    target=root/'analysis'/prefix;target.mkdir(parents=True,exist_ok=True)
    summaries=[];allmetrics=[]
    for item in state['runs']:
        if not item['passed']:continue
        folder=root/'evidence'/item['tag'];ranks=[];cache=Counter();reasons=Counter()
        task_groups={(row['step'],row['group']):{tuple(key) for key in row['keys']}
                     for row in load(folder/'temporal-groups-rank0.jsonl')}
        for rank in range(8):
            events=[row for row in load(folder/f'framework-events-rank{rank}.jsonl')
                    if row['step']==2 and not row.get('initialization')]
            host=[row for row in load(folder/f'lifetime-events-rank{rank}.jsonl') if row['invocation'][0]==2]
            if not events:raise RuntimeError('No last-update device events: '+item['tag'])
            groups=defaultdict(list)
            for row in events:
                name=category(row)
                if name:groups[name].append((row['start_ms'],row['end_ms']))
            metrics={name:sum(last-first for first,last in union(values)) for name,values in groups.items()}
            host_sums=defaultdict(float)
            for row in host:
                duration=(row['host_end_ns']-row['host_begin_ns'])/1e6
                host_sums[row['name']]+=duration
                if row['name']=='prepare':
                    for entry in row['cache_uses']:
                        previous=entry['previous'];current=tuple(row['invocation'])
                        if not entry['hit']:kind='miss'
                        elif previous is None:kind='unknown'
                        elif tuple(previous[0])==current:kind='same_invocation'
                        elif tuple(previous[0])[:2]==current[:2]:kind='same_group_other_wave'
                        else:
                            before=task_groups.get(tuple(previous[0])[:2]);after=task_groups.get(current[:2])
                            kind=('different_microbatch_tasks' if before and after and before.isdisjoint(after)
                                  else 'different_group_shared_or_unknown_tasks')
                        cache[kind]+=1
                if row['name']=='flush' and row['pending_before']:
                    reasons[row.get('flush_reason','unknown')]+=1
            native_end=max((row['end_ms'] for row in events if row['name']=='operator.backward'),default=0)
            service_end=max((row['end_ms'] for row in events if row['name']=='lifetime.service_work'),default=0)
            metrics['service_after_last_moe_ms']=max(0,service_end-native_end)
            ranks.append(dict(rank=rank,device_interval_union_ms=metrics,host_envelope_sums_ms=dict(host_sums)))
            for name,value in metrics.items():allmetrics.append(dict(tag=item['tag'],small=item['small'],mode=item['mode'],rank=rank,kind='device interval union (includes waits/supply gaps)',metric=name,milliseconds=value))
            for name,value in host_sums.items():allmetrics.append(dict(tag=item['tag'],small=item['small'],mode=item['mode'],rank=rank,kind='host envelope (nested/nonadditive)',metric=name,milliseconds=value))
            if rank==0 and not item['small']:plot(events,groups,item,target)
        summaries.append(dict(tag=item['tag'],mode=item['mode'],small=item['small'],step=2,
            ranks=ranks,cache_reuse_counts_all_ranks=dict(cache),flush_reasons_all_ranks=dict(reasons)))
    result=dict(stage=state['stage'],summaries=summaries,performance=False,
        cache_identity='offline interpretation of actual prefix key=(step,group,wave,1), joined with temporal-groups; raw kind labels are retained but not trusted',
        semantics='last diagnostic update; per-rank Event origins; nested categories not additive; service_after_last_moe is not complete-update exposed tail')
    (target/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if allmetrics:
        with (target/'scope-metrics.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=allmetrics[0].keys());writer.writeheader();writer.writerows(allmetrics)
    lines=['# Causal diagnostics progress','',f"Stage: {state['stage']}; completed qualified captures: {len(summaries)}. No diagnostic wall time is a performance score.",'',
        'Each table shows median over8rank-local interval totals for one diagnostic update. This is not an independent-repeat confidence interval. Parent/child scopes overlap and are not added.','',
        '|Mode|Small|MoE forward/replay/back intervals(ms)|BF16 materialization(ms)|Home publish(ms)|Host copies/counts reads(ms)|Host active-mask gather(ms)|Host finish envelope(ms)|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for item in summaries:
        def med(kind,name):return statistics.median(rank[kind].get(name,0) for rank in item['ranks'])
        native=sum(med('device_interval_union_ms',name) for name in ('MoE forward','MoE recompute','MoE backward'))
        reads=sum(med('host_envelope_sums_ms',name) for name in ('host_observe.copies','host_observe.execution_counts'))
        lines.append(f"|{item['mode']}|{item['small']}|{native:.3f}|{med('device_interval_union_ms','owner BF16 materialization'):.3f}|{med('device_interval_union_ms','publish home image'):.3f}|{reads:.3f}|{med('host_envelope_sums_ms','host_object_all_gather'):.3f}|{med('host_envelope_sums_ms','finish'):.3f}|")
    lines+=['','Raw original AscendTimerV2 arrays, coordinate stages and producer/service edges remain in each evidence directory. The NPU figure uses rank0 only and an explicitly local update origin. CPU envelopes are separate from kernel active time.','',
        'service_after_last_moe is only relative to the last native MoE backward; Attention/FSDP and other work may still overlap. It is not claimed as complete training-step exposed tail.','']
    (target/'REPORT.md').write_text('\n'.join(lines))
    return result


def plot(events,groups,item,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin=min(row['start_ms'] for row in events)
    labels=('Attention forward','Attention recompute','Attention backward','LM head forward','LM head backward',
            'owner BF16 materialization','publish home image','online plan + validation','MoE forward','MoE recompute','MoE backward','raw gradient adapter','gradient service','gradient drain','global clip','AdamW')
    fig,axis=plt.subplots(figsize=(14,7))
    for index,name in enumerate(labels):
        values=union(groups.get(name,[]))
        axis.broken_barh([(first-origin,last-first) for first,last in values],(index-.35,.7),
                        facecolors='tab:orange' if name=='gradient service' else 'tab:green' if name.startswith(('Attention','LM head')) else 'tab:blue')
    axis.set_yticks(range(len(labels)),labels);axis.invert_yaxis()
    axis.set_xlabel('Rank0 Event time since first observed update2 event (ms); not aligned across ranks')
    axis.set_title(item['mode']+' / diagnostic intervals including waits and host-supply gaps; blank space is not proven idle')
    axis.grid(axis='x',alpha=.2);fig.tight_layout()
    for extension in ('png','svg','pdf'):fig.savefig(target/(item['mode']+'-rank0.'+extension),dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('prefix',default='causal-v1',nargs='?');args=parser.parse_args()
    result=analyze(args.prefix);print(json.dumps(dict(stage=result['stage'],captures=len(result['summaries']))))
