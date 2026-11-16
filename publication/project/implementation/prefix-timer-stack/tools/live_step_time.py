"""Read-only live stage status, including current foreign GPU observations."""
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def tail(path):
    if not path.exists() or not path.stat().st_size:return {}
    with path.open('rb') as f:f.seek(max(0,path.stat().st_size-65536));lines=f.read().splitlines()
    return json.loads(lines[-1]) if lines else {}
stages={};latest=None
for name in ('primary-step-time','fourth-paired','heldout-a','heldout-b','ablation-no_gradient_service','ablation-linear_kernel','ablation-uniform_topology'):
    p=ROOT/f'evidence/campaign-{name}.json'
    if not p.exists():continue
    rows=json.loads(p.read_text());stages[name]=dict(recorded=len(rows),qualified=sum(x['qualified'] for x in rows),last=rows[-1]['tag'] if rows else None)
    if rows and (latest is None or rows[-1]['finish']>latest['finish']):latest=rows[-1]
monitor=tail(ROOT/'evidence/driver-process-observations-pod6-1-resume.jsonl')
print(json.dumps(dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stages=stages,
    driver=tail(ROOT/'evidence/step-time-stage-driver.jsonl'),last_status=({k:latest[k] for k in ('tag','state','exit','peak_driver_MiB')} if latest else None),
    monitor_age_seconds=time.time()-monitor.get('timestamp',time.time()),foreign_gpu_processes=monitor.get('foreign_or_unidentified',[]),
    no_process_cards=monitor.get('no_process_cards'))),flush=True)
