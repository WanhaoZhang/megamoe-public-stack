"""Compare real Native copies/quota across B modes at the frozen main shape."""
from pathlib import Path
import json
import subprocess
import sys
from run_diagnostic_gates import publish, LANE

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'diagnostics/main-plan-gates.json'


def save(state):
    text=json.dumps(state,indent=2)+'\n'
    STATE.write_text(text)
    (ROOT/'diagnostics/diagnostic-gates.json').write_text(text)


def rows(tag,rank):
    path=ROOT/'evidence'/tag/f'actual-plan-hashes-rank{rank}.jsonl'
    result=[json.loads(line) for line in path.read_text().splitlines()]
    return [row for row in result if row['invocation'] is not None and row['invocation'][0]>=0]


def main():
    if STATE.exists():raise RuntimeError('Preserve main plan audit registration')
    state=dict(stage='publishing_main_plan_audit',runs=[],performance=False)
    save(state)
    state['published']=publish();state['stage']='main_plan_gates';save(state)
    for mode in ('anchor','b0','b1','b2','b3'):
        tag='lifetime-main-plan-hash-'+mode+'-v1'
        extra=['--layers','4','--steps','3','--tokens','16384','--hidden','4096','--ffn','1024',
            '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
            '--planner','native','--temporal-pattern','random','--temporal-policy','fixed_native',
            '--lifetime-mode',mode,'--worker-variant','ep8-cap64-lifetime-v1',
            '--temporal-scheduler','ascendc','--temporal-scheduler-library',
            LANE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so',
            '--grad-sync','--hold-fsdp-forward','--temporal-prefix-reuse','--plan-hash-qualification']
        with (ROOT/'diagnostics'/(tag+'-driver.log')).open('wb') as handle:
            code=subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'remote_timer.py'),
                'gate','--tag',tag,*extra],stdout=handle,stderr=subprocess.STDOUT).returncode
        item=dict(tag=tag,mode=mode,exit=code);state['runs'].append(item);save(state)
        if code:raise RuntimeError('Main plan audit execution failed: '+tag)
        folder=ROOT/'evidence'/tag
        status=json.loads((folder/'status.json').read_text())
        if not all(status.get(key) for key in ('qualified','source_unchanged','exclusive_devices')) or status['peak_driver_MiB']>90*1024:
            raise RuntimeError('Main plan audit admission/memory/source gate failed')
        item['replica_max']=max(row['replicas'] for rank in range(8) for row in rows(tag,rank))
        if item['replica_max']==0:raise RuntimeError('Main shape produced no replicas; do not qualify empty path')
        save(state)
    checks={}
    for rank in range(8):
        signatures=[]
        for item in state['runs']:
            records=rows(item['tag'],rank)
            signatures.append([(row['invocation'],row['recompute'],row['hashes']) for row in records])
        checks[str(rank)]=dict(calls=len(signatures[0]),equal=all(value==signatures[0] for value in signatures[1:]))
    result=dict(passed=all(value['equal'] for value in checks.values()),ranks=checks,
        tags=[item['tag'] for item in state['runs']],performance=False,
        scope='main L4 H4096 F1024 S16384 GA4 W2 random routes, three parameter versions; actual compiled Native plans with positive replicas',
        fields=['copies','source_quota','logical_ids','physical_ids','logical_counts'])
    (ROOT/'diagnostics/lifetime-plan-equality.json').write_text(json.dumps(result,indent=2))
    state.update(stage='complete' if result['passed'] else 'failed',passed=result['passed'],plan_equality=result)
    save(state)
    if not result['passed']:raise RuntimeError('Compiled plans differ across lifecycle modes')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        state.update(stage='failed',error=str(error),performance=False);save(state)
        raise
