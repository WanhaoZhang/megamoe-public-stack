"""Finite Track A campaign; every arm must pass matched gradient and memory gates."""
from pathlib import Path
import json,subprocess,sys
import remote
ROOT=Path(__file__).resolve().parent
remote.REMOTE='/home/z00956592/megaflux/step-lifetime-kernel-20261115'
remote.CONTAINER='step-lifetime-kernel-20261115'
LANE=remote.REMOTE+'/stack'
LIB=LANE+'/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so'
ARMS=('anchor','b0','b1','b2','b3')
PREFIX=sys.argv[1] if len(sys.argv)>1 else 'lifetime-b2'
STATE=ROOT/(PREFIX+'.json')

def save(state):
    STATE.write_text(json.dumps(state,indent=2)+'\n')

def job(tag,arm,small=False,steps=3,pattern='random'):
    args=['--layers','2' if small else '4','--steps',str(steps),'--tokens','256' if small else '16384',
        '--hidden','512' if small else '4096','--ffn','512' if small else '1024','--microbatches','4',
        '--forward-window','2','--recv-capacity','4096' if small else '262144','--maxvio','10',
        '--planner','native','--temporal-pattern',pattern,'--temporal-policy','fixed_native','--lifetime-mode',arm,'--worker-variant','ep8-cap64-lifetime-v1',
        '--temporal-scheduler','ascendc','--temporal-scheduler-library',LIB,'--grad-sync',
        '--hold-fsdp-forward','--temporal-prefix-reuse']
    if small:args+=['--route-bank',LANE+'/small-routes','--dump-grad-step0']
    code=subprocess.run([sys.executable,str(ROOT/'remote_b.py'),'gate','--metadata-only','--tag',tag,*args]).returncode
    folder=ROOT/'evidence'/tag
    receipt=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    passed=(code==0 and receipt.get('qualified') and receipt.get('source_unchanged') and
            receipt.get('exclusive_devices') and receipt.get('peak_driver_MiB',1e9)<=90*1024)
    return dict(tag=tag,arm=arm,small=small,steps=steps,pattern=pattern,passed=passed,receipt=receipt)

def main():
    if STATE.exists():raise RuntimeError('Registration already exists; preserve tags')
    cpu=json.loads((ROOT/'diagnostics/delivery-cpu.json').read_text())
    if not cpu['passed']:raise RuntimeError('CPU delivery oracle failed')
    state=dict(stage='gates',gates=[],runs=[],independent_repeats=1,formal=False,qualification_path=True,
               control='CPU observed current placement; all observation and service costs counted')
    save(state)
    for arm in ARMS:
        item=job(PREFIX+'-math-'+arm,arm,small=True)
        state['gates'].append(item);save(state)
        if not item['passed']:raise RuntimeError('Numerical execution failed: '+arm)
        if arm!='anchor':
            target=LANE+'/evidence/'+item['tag']+'-comparison.json'
            command=['ssh',remote.HOST,'docker','exec',remote.CONTAINER,
                '/usr/local/python3.12.13/bin/python3',LANE+'/tools/compare_temporal_gradients.py',
                LANE+'/evidence/'+PREFIX+'-math-anchor',LANE+'/evidence/'+item['tag'],
                '--step0-only','--out',target]
            item['gradient_comparison_exit']=subprocess.run(command).returncode
            subprocess.run(['scp',remote.HOST+':'+target,str(ROOT/'evidence')],check=True)
            save(state)
            if item['gradient_comparison_exit']:raise RuntimeError('Per-parameter gradient gate failed: '+arm)
    state['stage']='main_memory_gate';save(state)
    item=job(PREFIX+'-main-adam3','b3')
    state['gates'].append(item);save(state)
    if not item['passed']:raise RuntimeError('Main shape failed; do not shrink')
    state['stage']='screening';save(state)
    for index,pattern in enumerate(('same','alternating','random')):
        order=ARMS[index:]+ARMS[:index]
        for arm in order:
            item=job(PREFIX+'-screen-'+pattern+'-'+arm+'-r1',arm,steps=25,pattern=pattern)
            state['runs'].append(item);save(state)
            if not item['passed']:raise RuntimeError('Screening failure: '+item['tag'])
    state['stage']='screening_complete';save(state)

if __name__=='__main__':
    if STATE.exists():raise SystemExit('Registration exists; choose a fresh prefix')
    try:main()
    except BaseException as error:
        if STATE.exists():
            state=json.loads(STATE.read_text());state.update(stage='stopped_at_gate',error=str(error));save(state)
        raise
