"""Two fresh, rotated confirmations of the v3 screen's critical control.

No sync/build/source change. Round1 is preserved, not rerun or selected from
multiple repeats. No new planner or108-run formal matrix is claimed.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
path=root/'resident-v3-confirmation.json'
if path.exists():raise RuntimeError('Existing confirmation registration; do not overwrite/reuse tags')
prior=json.loads((root/'resident-prefix-v3.json').read_text())
if prior['stage']!='pilot_complete' or len(prior['pilot_jobs'])!=7:
    raise RuntimeError('The seven-arm v3 pilot must finish first')
current=hashlib.sha256((root/'stack/tools/temporal_prefix_reuse.py').read_bytes()).hexdigest()
if current!=prior['source_module_sha256']:raise RuntimeError('Do not change runtime before fresh confirmations')
labels=('native_resident','easy_home','fixed_home')
jobs=[dict(job,repeat=1) for job in prior['pilot_jobs'] if job['label'] in labels]
order={2:('fixed_home','native_resident','easy_home'),3:('easy_home','fixed_home','native_resident')}
state=dict(stage='registered',created=datetime.now(timezone.utc).isoformat(),gates=prior['gates'],
    pilot_jobs=jobs,independent_repeats=3,formal=False,new_planner=False,
    source_module_sha256=current,source_change_allowed=False,first_round='resident-prefix-v3.json',
    selection_rule='fixed_home was fastest available control in the completed seven-arm screen; no Afore device baseline',
    interpretation='Confirm temporal selection against fixed same-layer grouping; do not claim a new method',
    preregistered_order={str(key):value for key,value in order.items()})
def save():
    state['updated']=datetime.now(timezone.utc).isoformat()
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
save()
try:
    for repeat,sequence in order.items():
        for label in sequence:
            tag=f'resident-v3-confirm-{label}-r{repeat}'
            state.update(stage='pilot',running_tag=tag);save()
            policy='legacy' if label=='native_resident' else label
            args=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
                '--tag',tag,'--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
                '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
                '--planner','native','--temporal-pattern','random','--temporal-policy',policy,
                '--temporal-scheduler','ascendc','--grad-sync','--hold-fsdp-forward']
            prefix=label!='native_resident'
            if prefix:args+=['--temporal-prefix-reuse']
            code=subprocess.run(args).returncode
            receipt_file=root/'evidence'/tag/'status.json'
            receipt=json.loads(receipt_file.read_text()) if receipt_file.exists() else {}
            qualified=bool(code==0 and receipt.get('qualified') and receipt.get('source_unchanged') and
                receipt.get('exclusive_devices') and receipt.get('peak_driver_MiB',float('inf'))<=90*1024)
            job=dict(tag=tag,label=label,policy=policy,prefix_reuse=prefix,hold_fsdp_forward=True,
                repeat=repeat,execution_exit=code,receipt=receipt,qualified=qualified,
                status='completed' if qualified else 'failed',warmup=5,measured=20,formal=False)
            state['pilot_jobs'].append(job);save()
            if not qualified:raise RuntimeError('Confirmation stopped at first failed attempt: '+tag)
            old=next(row for row in prior['pilot_jobs'] if row['label']==label)
            original=json.loads((root/'evidence'/old['tag']/'source-identity.json').read_text())
            actual=json.loads((root/'evidence'/tag/'source-identity.json').read_text())
            if actual!=original:raise RuntimeError('Source/route identity differs from preregistered first round')
            subprocess.run([sys.executable,'-X','utf8',str(root/'analyze_resident_confirmation.py')],check=True)
    state.update(stage='pilot_complete');state.pop('running_tag',None);save()
    subprocess.run([sys.executable,'-X','utf8',str(root/'analyze_resident_confirmation.py')],check=True)
except BaseException as error:
    state.update(stage='stopped_at_gate',error=f'{type(error).__name__}: {error}');save();raise
