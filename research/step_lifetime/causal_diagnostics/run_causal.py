"""Finite qualified causal capture; never launch a performance matrix."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
root=Path(__file__).resolve().parent
prefix=sys.argv[1] if len(sys.argv)>1 else 'causal-v1'
state_path=root/(prefix+'.json')
lane='/home/z00956592/megaflux/step-lifetime-causal-20261116'
variant='ep8-cap64-lifetime-timer-causal-v1'
container='step-lifetime-causal-20261116'
modes=('fixed_home','anchor','b0','b1','b3')
sys.path.insert(0,str(root/'stack/tools'))
from causal_proof import REQUIRED


def save(state):state_path.write_text(json.dumps(state,indent=2)+'\n')


def run(mode,small):
    tag=f'{prefix}-'+('math-' if small else 'main-')+mode
    args=['--layers','2' if small else '4','--steps','3','--tokens','256' if small else '16384',
        '--hidden','512' if small else '4096','--ffn','512' if small else '1024','--microbatches','4',
        '--forward-window','2','--recv-capacity','4096' if small else '262144','--maxvio','10',
        '--planner','moonep' if small else 'native','--temporal-pattern','same',
        '--temporal-policy','fixed_home' if mode=='fixed_home' else 'fixed_native',
        '--diagnostic-timer','--worker-variant',variant,'--grad-sync','--hold-fsdp-forward',
        '--temporal-prefix-reuse','--temporal-scheduler','ascendc','--temporal-scheduler-library',
        lane+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
    if mode!='fixed_home':args+=['--lifetime-mode',mode]
    if small:args+=['--causal-qualification','--dump-grad-step0','--route-bank',lane+'/stack/small-routes']
    else:args+=['--causal-capture-proof',lane+'/stack/configs/'+prefix+'-qualification.json']
    with (root/'diagnostics'/(tag+'-driver.log')).open('wb') as log:
        code=subprocess.run([sys.executable,str(root/'remote_causal.py'),'gate','--tag',tag,*args],
                            stdout=log,stderr=subprocess.STDOUT).returncode
    folder=root/'evidence'/tag
    status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    passed=code==0 and all(status.get(key) for key in ('qualified','source_unchanged','exclusive_devices')) and status['peak_driver_MiB']<=90*1024
    item=dict(tag=tag,mode=mode,small=small,exit=code,passed=bool(passed))
    if not passed:return item
    if small:
        # Reference also uses same temporal pattern; use a new TimerOFF
        # fixed-home run generated below rather than an incompatible bank.
        reference=lane+'/stack/evidence/'+prefix+'-reference'
        output=lane+'/stack/evidence/'+tag+'-comparison.json'
        command=['ssh','pod6-1-root','docker','exec','-e','OMP_NUM_THREADS=1',container,
            '/usr/local/python3.12.13/bin/python3',lane+'/stack/tools/compare_temporal_gradients.py',
            reference,lane+'/stack/evidence/'+tag,'--step0-only','--out',output]
        with (root/'diagnostics'/(tag+'-compare.log')).open('wb') as log:
            compared=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        subprocess.run(['scp','pod6-1-root:'+output,str(root/'diagnostics')],check=True)
        item['gradient_comparison_exit']=compared
        item['passed'] &= compared==0
    coverage=[]
    for rank in range(8):
        calls=0;words=0;stages=0;replicas=0;phases=set();replay=False
        for path in (folder/'timer').glob(f'rank{rank}-*.npz'):
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']));calls+=1;phases.add(meta['phase']);replay|=meta['recompute']
                words+=int(np.count_nonzero(data['original_'+meta['phase']]))
                stages+=len(data['stage_pairs'])
                copies=data['copies'][rank];counts=data['execution_counts'][32:32+len(copies)]
                replicas+=int(np.count_nonzero((copies>=0)&(counts>0)))
        passed=calls>0 and words>0 and phases=={'forward','backward'} and replay
        if mode!='fixed_home':passed &= replicas>0 and stages>0
        coverage.append(dict(rank=rank,calls=calls,words=words,stage_records=stages,active_replicas=replicas,passed=bool(passed)))
    item['coverage']=coverage;item['passed'] &= all(row['passed'] for row in coverage)
    return item


def main():
    if state_path.exists():raise RuntimeError('Keep registration, choose fresh prefix')
    state=dict(stage='reference',runs=[],formal_performance=False);save(state)
    # Current raw ABI1 TimerOFF worker is copied from the already-qualified B
    # lane. This reference never loads the new Timer or instrumentation.
    args=['--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
        '--planner','native','--temporal-pattern','same','--temporal-policy','fixed_home',
        '--worker-variant','ep8-cap64-lifetime-v1','--grad-sync','--hold-fsdp-forward',
        '--temporal-prefix-reuse','--route-bank',lane+'/stack/small-routes','--dump-grad-step0']
    # Reference uses the original unmodified B lane driver through its locked
    # controller, to retain a TimerOFF execution path and avoid changing it.
    reference_tag=prefix+'-reference'
    script=f"cd {lane}/stack\npython3 tools/controller.py --backend balanced --tag {reference_tag} --worker-variant ep8-cap64-lifetime-v1 --seconds 900 "+' '.join(args)+'\n'
    with (root/'diagnostics'/(reference_tag+'-driver.log')).open('wb') as log:
        result=subprocess.run(['ssh','pod6-1-root','bash','-s'],input=script.encode(),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('TimerOFF reference failed')
    state['stage']='small_qualification';save(state)
    for mode in modes:
        item=run(mode,True);state['runs'].append(item);save(state)
        if not item['passed']:raise RuntimeError('Causal numerical/record gate failed: '+mode)
    first=root/'evidence'/state['runs'][0]['tag']/'source-identity.json'
    sources=json.loads(first.read_text());critical={name:sources[name] for name in REQUIRED}
    for item in state['runs']:
        other=json.loads((root/'evidence'/item['tag']/'source-identity.json').read_text())
        if any(other.get(name)!=digest for name,digest in critical.items()):raise RuntimeError('Mixed causal source cohort')
    manifest=json.loads((root/'stack/configs/temporal-worker-timer-freeze.json').read_text())
    proof=dict(passed=True,modes=list(modes),runtime_sources=critical,worker_variant=variant,
        worker_sha256=next(iter(manifest['binaries'].values())),scope='small complete model, all modes, full FP32 gradients and record coverage',
        performance=False,runs=state['runs'])
    path=root/'stack/configs'/(prefix+'-qualification.json');path.write_text(json.dumps(proof,indent=2)+'\n')
    subprocess.run([sys.executable,str(root/'publish.py'),'configs/'+path.name],check=True)
    state['stage']='main_capture';save(state)
    for mode in modes:
        item=run(mode,False);state['runs'].append(item);save(state)
        if not item['passed']:raise RuntimeError('Main diagnostic failed: '+mode)
    state.update(stage='complete',passed=True);save(state)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        state.update(stage='failed',error=str(error));save(state);raise
