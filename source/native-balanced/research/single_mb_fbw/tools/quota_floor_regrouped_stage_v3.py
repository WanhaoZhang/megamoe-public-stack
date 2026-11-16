"""Native source-quota headroom after the sealed fast-pipeline stage retires."""
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

TASK=Path('/home/z00956592/megaflux/single-mb-fbw-planner-study')
sys.path.insert(0,str(TASK/'tools'))
from controller import run_job
from remaining_causal_stages import compiling
from verify_training_run import verify,verify_loaded_worker


def main():
    if not json.loads((TASK/'evidence/fast-redispatch-stage-v1.json').read_text()).get('complete'):
        raise RuntimeError('Prior pipeline queue is not retired')
    caps=json.loads((TASK/'evidence/fast-redispatch-capabilities-v1.json').read_text())
    inputs=[TASK/'artifacts/quota_floor_regrouped_stage_v3.py',TASK/'artifacts/audit_quota_bank.py',
            *sorted((TASK/'artifacts').glob('quota-floor-vio20-*')),
            *sorted((TASK/'artifacts/source-heterogeneity-bank-v1/regrouped').glob('*.npy')),
            *sorted((TASK/'artifacts/source-heterogeneity-bank-v1/regrouped').glob('*.json'))]
    def identity():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    frozen=identity()
    (TASK/'evidence/quota-floor-input-identity-v3.json').write_text(json.dumps(frozen,indent=2))
    worker='fastpipe-r0-t0';so=next(r['so_sha256'] for r in caps if r['variant']==worker)
    eligible=[];gates=[]
    for regime in ('regrouped',):
        for policy in ('native','ultra_ep'):
            route_dir=TASK/'routes' if regime=='original' else TASK/'artifacts/source-heterogeneity-bank-v1/regrouped'
            bank=TASK/f'artifacts/quota-floor-vio20-{regime}-{policy}.npz'
            tag=f'quota-floor-gate-{regime}-{policy}-v3'
            status=run_job('balanced','prepared_validate',tag,['--prepared-kind','body','--proposal-mode','bank',
                '--candidate-bank',str(bank),'--paired-diagnostic','--planner',policy,'--route-bank',str(route_dir),
                '--maxvio','20','--steps','3'],1800,worker)
            if identity()!=frozen:raise RuntimeError('Immutable quota input/code changed during a gate')
            root=TASK/'evidence'/tag
            summaries=[json.loads((root/f'rank{r}.json').read_text()) for r in range(8)] if status['exit']==0 else []
            passed=len(summaries)==8 and all(r['passed'] and r['steps']==3 and r['training_performance_score'] is False for r in summaries)
            proof=None
            if passed:
                output=root/'independent-physical-audit.json'
                subprocess.run(['docker','exec','single-mb-fbw-planner-study','/usr/local/python3.12.13/bin/python3',
                    str(TASK/'artifacts/audit_quota_bank.py'),'--directory',str(root),'--bank',str(bank),
                    '--route',str(route_dir/'maxvio-20.npy'),'--output',str(output)],check=True)
                proof=json.loads(output.read_text())
                if not proof['passed']:raise RuntimeError('Independent physical audit failed')
                eligible.append((regime,policy))
            gates.append(dict(tag=tag,regime=regime,planner=policy,numerical_passed=passed,physical=proof,**status))
            (TASK/'evidence/quota-floor-gates-v3.json').write_text(json.dumps(gates,indent=2))
            if not status['source_unchanged'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
    rows=[];identities={}
    for regime in ('regrouped',):
        route_dir=TASK/'routes' if regime=='original' else TASK/'artifacts/source-heterogeneity-bank-v1/regrouped'
        for policy in ('native','ultra_ep'):
            if (regime,policy) not in eligible:continue
            for proposal in ('original','local-source','bank'):
                if compiling():raise RuntimeError('Compiler active during diagnostic timing')
                extra=['--planner',policy,'--route-bank',str(route_dir),'--maxvio','20','--steps','25',
                    '--prepared-kind','body','--proposal-mode',proposal]
                bank=TASK/f'artifacts/quota-floor-vio20-{regime}-{policy}.npz'
                if proposal=='bank':extra+=['--candidate-bank',str(bank)]
                tag=f'quota-floor-pilot-{regime}-{policy}-{proposal}-v3'
                status=run_job('balanced','prepared',tag,extra,1800,worker)
                if identity()!=frozen:raise RuntimeError('Immutable quota input/code changed during a timing')
                if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
                root=TASK/'evidence'/tag;checked=verify(root);actual=[r['identities'] for r in checked]
                if actual!=identities.setdefault(regime,actual):raise RuntimeError('Initial identity mismatch')
                verify_loaded_worker(checked,worker,so)
                ranks=[[json.loads(l) for l in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
                mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
                rows.append(dict(tag=tag,regime=regime,planner=policy,proposal=proposal,mean_ms=mean,
                    online_score=False,diagnostic_only=True,independent_processes=1,**status))
                (TASK/'evidence/quota-floor-pilot-v3.json').write_text(json.dumps(rows,indent=2))
    (TASK/'evidence/quota-floor-stage-v3.json').write_text(json.dumps(dict(complete=True,cells=len(rows),eligible=eligible),indent=2))
    print('QUOTA_FLOOR_DIAGNOSTIC_STAGE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
