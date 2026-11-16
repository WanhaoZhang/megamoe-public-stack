"""Audit actual complete runs and reduce eight-rank full-step timing to CSV."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from campaign_records import init_failure,verified_contention

VIO=(.2,1,5,10,15,20)

def world_max_steps(ranks,steps=25):
    if len(ranks)!=8:raise ValueError('eight rank streams are required')
    by_rank=[]
    for rows in ranks:
        mapped={r['step']:r for r in rows}
        if len(rows)!=steps or set(mapped)!=set(range(steps)):raise ValueError('incomplete/duplicate training step stream')
        by_rank.append(mapped)
    return [max(rank[step]['milliseconds'] for rank in by_rank) for step in range(steps)]

def digest_json(path):
    data=json.loads(path.read_text())
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def audit_run(directory,freeze):
    status=json.loads((directory/'status.json').read_text())
    if not status['qualified'] or status['exit']!=0:raise ValueError('run did not complete successfully')
    if not status['source_unchanged']:raise ValueError('source changed during a timed process')
    actual_source=json.loads((directory/'source-identity.json').read_text())
    if actual_source!=freeze['source']:
        final_freeze=directory.parent/'STEP-TIME-FREEZE.json'
        if not final_freeze.exists() or actual_source!=json.loads(final_freeze.read_text())['source']:
            raise ValueError('formal sources match neither the original baseline nor qualified final source')
    if status['peak_driver_MiB']>90*1024:raise ValueError('whole-card memory gate exceeded')
    if '--diagnostic-timer' in status['command']:raise ValueError('diagnostic run cannot enter the formal matrix')
    effective=json.loads((directory/'effective-config.json').read_text())
    if effective['steps']!=25 or effective['layers']!=2 or effective['tokens']!=16384:raise ValueError('formal training shape/length mismatch')
    model=effective['model']
    for key,value in dict(hidden_size=7168,moe_intermediate_size=2048,n_routed_experts=256,num_experts_per_tok=8,vocab_size=32768,num_hidden_layers=2).items():
        if model[key]!=value:raise ValueError(f'formal model mismatch: {key}')
    streams=[];identities=[]
    for rank in range(8):
        complete=json.loads((directory/f'complete-rank{rank}.json').read_text())
        if complete['steps']!=25 or complete['exit']!=0:raise ValueError('missing rank completion')
        rows=[json.loads(line) for line in (directory/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        if any(not math.isfinite(row[k]) for row in rows for k in ('milliseconds','loss','grad_norm')):raise ValueError('NaN/Inf timing/loss/norm')
        streams.append(rows)
        plan=json.loads((directory/f'planning-audit-rank{rank}.json').read_text())
        if sum(x['calls'] for x in plan)!=50 or any(x['live'] for x in plan):raise ValueError('planner replay/call-count violation')
        if any(x['replay']!=dict(original=50,recompute=50,backward=50) for x in plan):raise ValueError('checkpoint/retirement mismatch')
        updates=json.loads((directory/f'updates-rank{rank}.json').read_text())
        if len(updates)!=3 or any(not x['changed'] for x in updates):raise ValueError('three real updates were not verified')
        libraries=json.loads((directory/f'libraries-rank{rank}.json').read_text())
        if len(libraries['shmem'])!=1:raise ValueError('mixed SHMEM runtimes')
        selected=[(name,item['sha256']) for name,item in libraries['libraries'].items() if any(key in name for key in ('libascend_deepep_unified_planner.so','_C.cpython','libascend_deepep_megaflux_early_bf16.so'))]
        if len(selected)!=3:raise ValueError('could not identify actual Host/planner/worker SOs')
        for name,sha in selected:
            relative='source/'+name.split('/source/',1)[1]
            if freeze['binaries'].get(relative)!=sha:raise ValueError(f'loaded binary differs from baseline freeze: {name}')
        if libraries.get('sitecustomize') is not None:raise ValueError('hidden sitecustomize dependency')
        if effective['planner']=='step_time':
            plugins=[v['sha256'] for k,v in libraries['libraries'].items() if k.endswith('/libascend_deepep_step_time_planner.so')]
            if len(plugins)!=1 or not plan[0]['model_sha']:raise ValueError('missing actual step-time plugin/model identity')
            policy_freeze=json.loads((directory.parent/'STEP-TIME-FREEZE.json').read_text())
            if plugins[0]!=policy_freeze['plugin_sha256'] or plan[0]['model_sha'] not in policy_freeze['approved_model_shas']:
                raise ValueError('actual policy plugin/model differs from the independent calibration freeze')
            if json.loads((directory/'source-identity.json').read_text())!=policy_freeze['source']:
                raise ValueError('step_time sources differ from its independent qualification freeze')
        identities.append(dict(rank=rank,parameters=digest_json(directory/f'initial-rank{rank}.json'),
            inputs=digest_json(directory/f'input-identity-rank{rank}.json'),routes=digest_json(directory/f'route-identity-rank{rank}.json'),
            binaries=selected))
    times=world_max_steps(streams)[5:]
    return dict(mean_ms=statistics.mean(times),p50_ms=statistics.median(times),p90_ms=sorted(times)[math.ceil(.9*len(times))-1],
        min_ms=min(times),max_ms=max(times),std_ms=statistics.stdev(times),
        peak_hbm_gib=status['peak_driver_MiB']/1024,measured_world_max_ms=times,identities=identities,
        receive_capacity=effective['receive_capacity'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--campaign',action='append',required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    destination=a.out or a.root/'analysis';destination.mkdir(exist_ok=True,parents=True)
    freeze=json.loads((a.root/'evidence/BASELINE-FREEZE.json').read_text())
    records=[];common_identity={};global_identity={}
    for name in a.campaign:
        for row in json.loads((a.root/'evidence'/f'campaign-{name}.json').read_text()):
            entry={k:row[k] for k in ('tag','planner','maxvio','repeat','state')}
            if row.get('retry_of'):entry['retry_of']=row['retry_of']
            contaminated=verified_contention(a.root,row)
            if contaminated:
                entry['original_state']=row['state'];entry['state']='excluded_external_contention'
                entry['excluded_external_contention']=True
            if not row['qualified']:entry['pretraining_hccl_init_failure']=init_failure(a.root,row)
            if row['qualified'] and not contaminated:
                metrics=audit_run(a.root/'evidence'/row['tag'],freeze);entry.update(metrics)
                for identity in metrics['identities']:
                    key=identity['rank'],row['maxvio']
                    comparison=(identity['parameters'],identity['inputs'],identity['routes'])
                    if key in common_identity and common_identity[key]!=comparison:raise ValueError(f'parameter/input/route identity changed: {row["tag"]},rank{key[0]}')
                    common_identity[key]=comparison
                    pair=(identity['parameters'],identity['inputs'])
                    if identity['rank'] in global_identity and global_identity[identity['rank']]!=pair:raise ValueError('initial logical parameters or input changed across MaxVio')
                    global_identity[identity['rank']]=pair
            records.append(entry)
    points=[]
    for planner in dict.fromkeys(row['planner'] for row in records):
        for vio in VIO:
            rows=[x for x in records if x['planner']==planner and x['maxvio']==vio]
            good=[x for x in rows if x['state']=='pass' and 'mean_ms' in x]
            successful_retries={x['retry_of'] for x in good if x.get('retry_of')}
            init_recovered=[x for x in rows if x.get('pretraining_hccl_init_failure') and x['tag'] in successful_retries and not x.get('excluded_external_contention')]
            contention_recovered=[x for x in rows if x.get('excluded_external_contention') and x['tag'] in successful_retries]
            unresolved=len(rows)-len(good)-len(init_recovered)-len(contention_recovered)
            repeats=[x['repeat'] for x in good]
            if len(repeats)!=len(set(repeats)):raise ValueError('successful/slow repeated run was improperly duplicated')
            point=dict(planner=planner,maxvio=vio,attempts=len(rows),completed=len(good),failed=len(rows)-len(good),
                recovered_initialization_failures=len(init_recovered),unresolved_failures=unresolved,
                excluded_contention_attempts=sum(bool(x.get('excluded_external_contention')) for x in rows),
                recovered_contention_attempts=len(contention_recovered),
                state=('pass_with_infrastructure_retry' if contention_recovered else 'pass_with_init_retry' if init_recovered else 'pass') if rows and unresolved==0 else 'failed' if rows else 'not_run')
            if good:
                values=[x['mean_ms'] for x in good]
                point.update(mean_ms=statistics.mean(values),repeat_min_ms=min(values),repeat_max_ms=max(values),p50_ms=statistics.median(x['p50_ms'] for x in good),peak_hbm_gib=max(x['peak_hbm_gib'] for x in good))
            points.append(point)
    baseline={(x['maxvio']):x for x in points if x['planner']=='moonep' and x.get('mean_ms')}
    first=baseline.get(.2,{})
    denominator=first.get('mean_ms') if first.get('completed')==3 and first.get('unresolved_failures')==0 else None
    for point in points:
        if 'mean_ms' in point and denominator:point['normalized_ms']=point['mean_ms']/denominator
        if 'mean_ms' in point and point['maxvio'] in baseline:point['speedup_vs_moonep']=baseline[point['maxvio']]['mean_ms']/point['mean_ms']
    (destination/'runs.json').write_text(json.dumps(records,indent=2))
    (destination/'points.json').write_text(json.dumps(dict(points=points,common_denominator_ms=denominator),indent=2))
    for filename,data in [('runs.csv',[{k:v for k,v in r.items() if k not in ('identities','measured_world_max_ms')} for r in records]),('points.csv',points)]:
        fields=list(dict.fromkeys(key for row in data for key in row))
        with (destination/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(data)
    print(json.dumps(dict(runs=len(records),passed=sum(x['state']=='pass' for x in records),points=len(points),output=str(destination))))

if __name__=='__main__':
    # A changing straggler distinguishes max-before-reduction from the
    # incorrect max-of-rank-mean. No hardware or implementation mirroring.
    test=[[dict(step=s,milliseconds=2 if r==s else 1) for s in range(2)] for r in range(8)]
    assert world_max_steps(test,2)==[2,2]
    main()
