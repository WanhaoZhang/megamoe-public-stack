"""Paired same-weight qualification for explicitly offline interventions."""
import json
import hashlib
from controller import TASK,run_job,source_identity


def adopt_native_gates():
    previous=json.loads((TASK/'evidence/prepared-plan-gates-v2.json').read_text())
    current=source_identity();results=[]
    old_choices="choices=('moonep','native','ultra_ep','echo','eplb_fixed','step_time')"
    new_choices="choices=('moonep','native','ultra_ep','echo','eplb_fixed','step_time',*[f'eplb_budget_{b}' for b in (0,1,2,4,8,16,32,64,128,256)])"
    text=(TASK/'tools/gradient_diagnostic.py').read_text();assert text.count(new_choices)==1
    restored_hash=hashlib.sha256(text.replace(new_choices,old_choices).encode()).hexdigest()
    for kind in ('online','quota','body'):
        tag=f'prepared-accuracy-native-{kind}-original-v2'
        old=next(r for r in previous if r['tag']==tag)
        assert old['qualified'] and old['numerical_gate_passed'] and old['source_unchanged']
        identities=json.loads((TASK/'evidence'/tag/'source-identity.json').read_text())
        for key,digest in identities.items():
            if key.startswith('source/') or key.startswith('configs/') or key.startswith('routes/') or key in (
                    'tools/prepared_plan_train.py','tools/train.py','tools/fixed_state_pair.py',
                    'tools/launch.sh','tools/run_audit.py','tools/benchmark_backend.py'):
                assert current[key]==digest,(key,'adoption dependency changed')
        assert identities['tools/gradient_diagnostic.py']==restored_hash,'Diagnostic changed beyond parser choices'
        results.append(dict(old,adopted_from='prepared-plan-gates-v2.json',
            adoption_proof='all core/source/SO/input/wrapper dependencies identical; generated diagnostic differs only by accepted planner names'))
    return results


def main():
    results=adopt_native_gates()
    for planner,kind,proposal in [('eplb_budget_32','body','local-source'),('eplb_budget_32','body','reverse-slots'),
            ('native','body','return-smallest')]:
        tag=f'prepared-accuracy-{planner}-{kind}-{proposal}-v3'
        status=run_job('balanced','prepared_validate',tag,['--prepared-kind',kind,'--proposal-mode',proposal,
            '--paired-diagnostic','--planner',planner,'--maxvio','5','--steps','3'],1800)
        # Paired diagnostics write rank*.json rather than performance completions.
        out=TASK/'evidence'/tag
        summaries=[json.loads((out/f'rank{rank}.json').read_text()) for rank in range(8)] if status['exit']==0 else []
        passed=status['exit']==0 and status['source_unchanged'] and len(summaries)==8 and all(
            s['passed'] and s['steps']==3 and s['training_performance_score'] is False for s in summaries)
        results.append(dict(tag=tag,numerical_gate_passed=passed,**status))
        (TASK/'evidence/prepared-plan-gates-v3.json').write_text(json.dumps(results,indent=2))
        if not passed or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
    print('PREPARED_PLAN_GATES_PASS',len(results),flush=True)


if __name__=='__main__':main()
