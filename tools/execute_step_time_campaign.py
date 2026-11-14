"""Execute already-approved stages; stop on a real failure for root review."""
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LOG=ROOT/'evidence/step-time-stage-driver.jsonl'
def record(**kw):
    row=dict(time_utc=datetime.now(timezone.utc).isoformat(),**kw)
    with LOG.open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def run(stage,variant=None):
    name='full' if variant is None else variant.replace('_','-')
    model=ROOT/f'configs/step-time-{name}.json'
    command=[sys.executable,'-u',str(ROOT/'tools/campaign_search.py'),stage,'--model',str(model)]
    if variant:command+=['--ablation',variant]
    record(state='start',stage=stage,variant=variant,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest())
    subprocess.run(command,check=True)
    record(state='complete',stage=stage,variant=variant)
    subprocess.run([sys.executable,str(ROOT/'tools/analyze_all_stages.py')],check=True)
def main():
    assert not (ROOT/'evidence/USER-CLOSEOUT-REQUEST.json').exists(),'task closed at user request; no automatic continuation'
    started=time.monotonic();announced=False
    while True:
        p=ROOT/'evidence/step-time-qualification.json'
        if p.exists() and json.loads(p.read_text()).get('passed'):break
        if not announced:record(state='waiting_for_final_qualification');announced=True
        if time.monotonic()-started>7200:raise RuntimeError('qualification not complete after two hours; review before running')
        time.sleep(10)
    for stage in ('primary','fourth','heldout'):run(stage)
    stable=json.loads((ROOT/'analysis/stability-four-rounds.json').read_text())
    if any(x['stable'] for x in stable):
        for variant in ('no_gradient_service','linear_kernel','uniform_topology'):run('ablation',variant)
    else:record(state='conditional_ablations_not_required',reason='no stable point at the frozen four-round threshold')
    record(state='all_measurements_complete')
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        record(state='needs_review',error=repr(exc));raise
