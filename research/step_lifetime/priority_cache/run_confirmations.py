"""Predeclared bounded confirmation and matched controls; no large matrix."""
from pathlib import Path
import json
import remote
from audit_runs import audit, load, WORKERS

ROOT = Path(__file__).resolve().parent
STATE = ROOT/'confirmation-v1.json'
if STATE.exists():
    raise SystemExit('Immutable confirmation state already exists; inspect it instead of restarting.')
if not load(ROOT/'diagnostics/pilot-identity-audit.json')['passed']:
    raise SystemExit('Actual initial pair identity audit required.')
candidate = list(WORKERS)[1]
reference = list(WORKERS)[0]
schedule = [
    ('repeat2-cached_keys', candidate, 'b1'),
    ('repeat2-reference', reference, 'b1'),
    ('repeat3-reference', reference, 'b1'),
    ('repeat3-cached_keys', candidate, 'b1'),
    ('control1-anchor', candidate, 'off'),
    ('control1-fixed_home', candidate, 'off'),
]
state = dict(stage='running', predeclared_schedule=schedule, runs=[], new_planner=False,
             note='Two more reversed-order pairs plus one fresh common-runtime anchor/home screen; no stable policy claim from controls N=1.')
def save():
    STATE.write_text(json.dumps(state, indent=2)+'\n', newline='\n')
save()
try:
    for name, variant, lifetime in schedule:
        tag = 'priority-'+name+'-v1'
        policy = 'fixed_home' if name.endswith('fixed_home') else 'fixed_native'
        args = ['--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
                '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
                '--planner','native','--temporal-pattern','same','--temporal-policy',policy,
                '--worker-variant',variant,'--grad-sync','--hold-fsdp-forward','--temporal-prefix-reuse',
                '--temporal-scheduler','ascendc','--temporal-scheduler-library',remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
        if lifetime != 'off':
            args += ['--lifetime-mode', lifetime, '--lifetime-raw-schedule','preserve']
        state['current'] = tag
        save()
        try:
            remote.gate(tag, args, metadata_only=True)
        except SystemExit as done:
            code = done.code
        if code != 0:
            raise RuntimeError('Admitted run failed: '+tag)
        result = audit(tag, variant, reference='priority-pilot-reference-v1', lifetime=lifetime)
        result.update(name=name, variant=variant, lifetime=lifetime, temporal_policy=policy)
        state['runs'].append(result)
        save()
        if not result['passed']:
            raise RuntimeError('Identity/timing qualification failed: '+str(result['failed_checks']))
    state['stage'] = 'complete'
    state['passed'] = True
    state.pop('current', None)
    save()
except BaseException as error:
    state.update(stage='failed', error=str(error))
    save()
    raise
