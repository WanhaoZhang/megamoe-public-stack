"""Audit actual current runs; do not infer comparability from launch intent."""
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parent
load=lambda path:json.loads(path.read_text())
state=load(ROOT/'study-v1.json')
if not state.get('passed'):raise SystemExit('Bounded study is not complete')
runs=[row for row in state['runs'] if row['phase']=='pilot']
if len(runs)!=2:raise RuntimeError('Need exactly one actual process per scope')
old,new=[ROOT/'evidence'/row['tag'] for row in runs]
checks={}
checks['same_actual_sources']=load(old/'source-identity.json')==load(new/'source-identity.json')
checks['same_effective_configuration']=load(old/'effective-config.json')==load(new/'effective-config.json')
stats=('weight_remote_bytes','weight_hit_bytes','cache_local_copy_bytes','gradient_remote_bytes','gradient_deliveries','flushes')
for rank in range(8):
    for stem in ('initial','persistent-initial','input-identity','route-identity','route-calls'):
        checks[f'rank{rank}:same_{stem}']=load(old/f'{stem}-rank{rank}.json')==load(new/f'{stem}-rank{rank}.json')
    def groups(folder):
        return [{name:row[name] for name in ('step','group','keys','ready_keys','native_waves')} for row in
                (json.loads(line) for line in (folder/f'temporal-groups-rank{rank}.jsonl').read_text().splitlines())]
    checks[f'rank{rank}:same_task_groups']=groups(old)==groups(new)
    a,b=[load(folder/f'lifetime-audit-rank{rank}.json') for folder in (old,new)]
    checks[f'rank{rank}:same_traffic_and_delivery_counts']=all(a[name]==b[name] for name in stats)
    checks[f'rank{rank}:no_debt']=a['pending']==b['pending']==0
    checks[f'rank{rank}:declared_scope']=a['gradient_ready_scope']=='call_stream' and b['gradient_ready_scope']=='bank_write'
    checks[f'rank{rank}:baseline_no_extra_events']=a['ready_bank_writes']==a['ready_bank_waits']==0
    for folder in (old,new):
        tag=folder.name
        route_calls=load(folder/f'route-calls-rank{rank}.json')
        checks[f'{tag}:rank{rank}:replay']=len(route_calls)==400 and set(route_calls.values())=={2}
        updates=load(folder/f'updates-rank{rank}.json')
        checks[f'{tag}:rank{rank}:parameter_updates']=len(updates)==3 and all(row['changed'] for row in updates)
        checks[f'{tag}:rank{rank}:complete']=load(folder/f'complete-rank{rank}.json')==dict(rank=rank,steps=25,exit=0)
        config=load(folder/f'lifetime-config-rank{rank}.json')
        checks[f'{tag}:rank{rank}:abi']=config['worker_capability']==2 and config['raw_schedule']=='preserve' and config['mode']=='b3'
        libraries=load(folder/f'libraries-rank{rank}.json')
        workers=[value['sha256'] for name,value in libraries['libraries'].items() if '/worker-variants/' in name and name.endswith('.so')]
        checks[f'{tag}:rank{rank}:worker']=workers==['e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed'] and len(libraries['shmem'])==1
        values=[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        checks[f'{tag}:rank{rank}:finite']=len(values)==25 and all(all(math.isfinite(row[key]) for key in ('loss','grad_norm','milliseconds')) for row in values)
    checks[f'rank{rank}:nonempty_bank_event_records']=b['ready_bank_writes']>=b['ready_bank_waits']
checks['new_bank_events_exercised']=sum(load(new/f'lifetime-audit-rank{rank}.json')['ready_bank_waits'] for rank in range(8))>0
for row in runs:
    status=load(ROOT/'evidence'/row['tag']/'status.json')
    checks[row['tag']+':admission_source_memory']=all(status.get(name) for name in ('qualified','source_unchanged','exclusive_devices')) and status['peak_driver_MiB']<=90*1024
output=dict(passed=all(checks.values()),checks=checks,failed=[name for name,value in checks.items() if not value],
    measured_scope='Initial parameters, all25 inputs/routes/replays/task groups, actual SO and traffic counters. No extra full per-step plan tensor dump in timed runs.',
    independent_processes_per_scope=1,persistent_goal='paused')
(ROOT/'diagnostics/pair-identity-audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(passed=output['passed'],failed=output['failed'])))
raise SystemExit(0 if output['passed'] else 1)
