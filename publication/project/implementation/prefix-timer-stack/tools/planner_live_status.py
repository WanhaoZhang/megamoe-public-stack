"""Compact read-only status; not an acceptance or performance reducer."""
import json
from pathlib import Path
import time

root=Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
read=lambda path,default=None:json.loads(path.read_text()) if path.exists() else default
ledger=read(root/'evidence/EXTERNAL-CONTENTION.json',dict(intervals=[]))
rows=read(root/'evidence/campaign-primary-existing.json',[])
excluded=lambda row:any(row['start']<x['end'] and row['finish']>x['begin'] for x in ledger['intervals'])
valid=[r for r in rows if r['qualified'] and r['state']=='pass' and not excluded(r)]
contract=read(root/'configs/campaign-primary-existing.json')
order=[];planners=contract['planners']
for repeat in range(contract['first_repeat'],contract['first_repeat']+contract['repeats']):
    offset=(repeat-1)%len(planners)
    for target in contract['targets']:
        for policy in planners[offset:]+planners[:offset]:
            order.append(f'primary-existing-vio{target:g}-{policy}-r{repeat}')
next_tag=next((t for t in order if t not in {r['tag'] for r in rows}),None)
steps={}
if next_tag:
    for rank in range(8):
        path=root/f'evidence/{next_tag}/steps-rank{rank}.jsonl'
        steps[rank]=len(path.read_text().splitlines()) if path.exists() else 0
holdfiles=list((root/'evidence').glob('OWN-QUEUE-INFRASTRUCTURE-HOLD*.json'))
held=any(not p.with_name(p.name.replace('INFRASTRUCTURE-HOLD','RESUMED')).exists() for p in holdfiles)
monitor=root/'evidence/driver-process-observations.jsonl'
with monitor.open('rb') as stream:
    stream.seek(max(0,monitor.stat().st_size-65536));lines=stream.read().splitlines()
last=json.loads(lines[-1]) if lines else {}
print(json.dumps(dict(time=time.time(),raw_existing=len(rows),valid_existing=len(valid),
    excluded_existing=sum(excluded(r) for r in rows),failed_exits=sum(not r['qualified'] for r in rows),
    latest_completed=rows[-1]['tag'] if rows else None,next_original=next_tag,step_counts=steps,
    queue_held=held,monitor_age_seconds=time.time()-last.get('timestamp',time.time()),
    observed_foreign_gpu_pids=len(last.get('foreign_or_unidentified',[])),
    driver_no_process_cards=last.get('no_process_cards'))),flush=True)
