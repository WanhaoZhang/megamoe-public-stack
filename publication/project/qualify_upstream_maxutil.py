"""Freeze author-style selection, pass device/math/memory gates, then screen."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
root=Path(__file__).resolve().parent;path=root/'upstream-maxutil-results.json'
if path.exists():raise RuntimeError('Existing registration; never reuse tags')
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
library=remote+'/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so'
arms={'native_resident':('legacy',False,'work_tie'),
      'fixed_home':('fixed_home',True,'work_tie'),
      'upstream_home':('easy_home',True,'upstream_first'),
      'upstream_independent':('easy_independent',True,'upstream_first')}
orders={'same':('native_resident','fixed_home','upstream_home'),
        'alternating':('fixed_home','upstream_home','native_resident'),
        'random':('upstream_home','native_resident','fixed_home')}
state=dict(stage='registered',created=datetime.now(timezone.utc).isoformat(),gates=[],pilot_jobs=[],formal=False,
    new_planner=False,author_commit='6327f7bf24086547de8f622f2a00fd763024b3e9',
    selector_abi=4,selector_sha256='871c167357d1d6e1575d331b3e73c7b36bcc115fb41988e02d4d8a3bb029013e',
    ordinary_worker_unchanged=True,old_abi3_binary_preserved=True,
    preregistered_screen_order=orders,independent_repeats=1,
    author_scope='MaxUtil selected sets, first numeric arrival-mask tie, append forced after selection; training execution remains canonical and<=2 layer domains',
    old_results='V1 and v3 use work_tie and cannot reject the canonical author selection rule',
    matrix='9 fresh25-update processes at MaxVio10 only, after all gates; not108 formal runs')
def save():
    state['updated']=datetime.now(timezone.utc).isoformat()
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
def run(tag,label,*,small=False,steps=3,pattern='random'):
    policy,prefix,rule=arms[label]
    args=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only','--tag',tag,
        '--layers','2' if small else '4','--steps',str(steps),'--tokens','256' if small else '16384',
        '--hidden','512' if small else '4096','--ffn','512' if small else '1024',
        '--microbatches','4','--forward-window','2','--recv-capacity','4096' if small else '262144',
        '--maxvio','10','--planner','native','--temporal-pattern',pattern,'--temporal-policy',policy,
        '--temporal-scheduler','ascendc','--temporal-scheduler-library',library,'--grad-sync','--hold-fsdp-forward']
    if prefix:args+=['--temporal-prefix-reuse']
    if rule=='upstream_first':args+=['--maxutil-rule',rule]
    if small:args+=['--route-bank','/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes','--dump-grad-step0']
    code=subprocess.run(args).returncode
    folder=root/'evidence'/tag
    receipt=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    qualified=bool(code==0 and receipt.get('qualified') and receipt.get('source_unchanged') and
        receipt.get('exclusive_devices') and receipt.get('peak_driver_MiB',float('inf'))<=90*1024)
    return dict(tag=tag,label=label,policy=policy,prefix_reuse=prefix,hold_fsdp_forward=True,maxutil_rule=rule,
        execution_exit=code,receipt=receipt,qualified=qualified,pattern=pattern,repeat=1,formal=False)
save()
try:
    cpu=json.loads((root/'diagnostics/upstream-maxutil-cpu.json').read_text())
    if not cpu['passed'] or cpu['author_selected_set_cases']!=560:raise RuntimeError('Author core conformance is not qualified')
    state.update(stage='device_scheduler_gate',running_tag='upstream-selector-abi4-gate');save()
    code=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag','upstream-selector-abi4-gate','--mode','scheduler','--library',library]).returncode
    folder=root/'evidence/upstream-selector-abi4-gate'
    receipt=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    state['device_gate']=dict(exit=code,receipt=receipt);save()
    if code or not receipt.get('qualified'):raise RuntimeError('Device scheduler gate refused/failed; no training')
    for rank in range(8):
        item=json.loads((folder/f'rank{rank}.json').read_text())
        if not item['passed'] or item['cases']!=674 or item['sha256']!=state['selector_sha256']:
            raise RuntimeError('Incomplete ABI4 device decisions')
    for label in ('upstream_home','upstream_independent','fixed_home'):
        tag='upstream-maxutil-gate-'+label;state.update(stage='small_full_gradient',running_tag=tag);save()
        gate=run(tag,label,small=True);gate['comparison_exit']=None
        if gate['qualified']:
            destination=remote+'/evidence/'+tag+'-comparison.json'
            gate['comparison_exit']=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
                '/usr/local/python3.12.13/bin/python3',remote+'/tools/compare_temporal_gradients.py',
                remote+'/evidence/gradient-w2-legacy',remote+'/evidence/'+tag,'--step0-only','--out',destination]).returncode
            subprocess.run(['scp','pod6-1-root:'+destination,str(root/'evidence')],check=True)
        state['gates'].append(gate);save()
        if not gate['qualified'] or gate['comparison_exit']!=0:raise RuntimeError('New order failed a numeric/source/memory gate: '+tag)
    state.update(stage='main_memory_gate',running_tag='upstream-maxutil-primary-same-adam3');save()
    memory=run('upstream-maxutil-primary-same-adam3','upstream_home',pattern='same')
    state['main_memory_gate']=memory;save()
    if not memory['qualified']:raise RuntimeError('Common90GiB main gate failed; no shape shrink')
    for pattern,sequence in orders.items():
        for label in sequence:
            tag=f'upstream-maxutil-screen-{pattern}-{label}-r1'
            state.update(stage='pilot',running_tag=tag);save()
            job=run(tag,label,steps=25,pattern=pattern)
            state['pilot_jobs'].append(job);save()
            if not job['qualified']:raise RuntimeError('Screen stopped at first failed process: '+tag)
            subprocess.run([sys.executable,'-X','utf8',str(root/'analyze_upstream_maxutil.py')],check=True)
    state.update(stage='pilot_complete');state.pop('running_tag',None);save()
    subprocess.run([sys.executable,'-X','utf8',str(root/'analyze_upstream_maxutil.py')],check=True)
except BaseException as error:
    state.update(stage='stopped_at_gate',error=f'{type(error).__name__}: {error}');save();raise
