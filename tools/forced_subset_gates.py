"""Numerical-only native gates, allowed to overlap task-owned CPU compilation."""
import json
from controller import TASK,run_job


def main():
    result=[]
    for vio in (1,5,20):
        for candidate in ('proxy_best','sparse_best'):
            bank=TASK/f'artifacts/native-subsets-vio{vio:g}-{candidate}.npz'
            metadata=json.loads(bank.with_suffix('.json').read_text())
            if sum(r['changed_from_native'] for r in metadata['records'][:6])<2:
                raise RuntimeError('First three steps do not exercise enough changed plans')
            tag=f'forced-subset-gate-vio{vio:g}-{candidate}-v1'
            status=run_job('balanced','prepared_validate',tag,['--prepared-kind','body','--proposal-mode','bank',
                '--candidate-bank',str(bank),'--paired-diagnostic','--planner','native',
                '--maxvio',str(vio),'--steps','3'],1800)
            if not status['source_unchanged']:raise RuntimeError('Source changed during numerical gate')
            summaries=[json.loads((TASK/'evidence'/tag/f'rank{r}.json').read_text()) for r in range(8)] if status['exit']==0 else []
            passed=len(summaries)==8 and all(r['passed'] and r['steps']==3 and r['training_performance_score'] is False for r in summaries)
            changed=[]
            if passed:
                for rank in range(8):
                    plans=json.loads((TASK/'evidence'/tag/f'prepared-plan-manifest-rank{rank}.json').read_text())
                    n=sum(r['changed_from_original'] for r in plans['records'])
                    if n<2:raise RuntimeError('Changed plan did not reach the native compiler')
                    changed.append(n)
            result.append(dict(tag=tag,candidate=candidate,maxvio=vio,numerical_gate_passed=passed,
                changed_prepared_plans_by_rank=changed,diagnostic_only=True,training_performance_score=False,**status))
            (TASK/'evidence/forced-subset-gates-v1.json').write_text(json.dumps(result,indent=2))
            if status['peak_driver_MiB']>90*1024:raise RuntimeError('Numerical gate exceeded common memory ceiling')
            # A failed forced candidate stays ineligible. It does not erase or
            # invalidate an unrelated matched metadata-cache experiment.
    print('FORCED_SUBSET_NUMERICAL_GATES',sum(r['numerical_gate_passed'] for r in result),'of',len(result),flush=True)


if __name__=='__main__':main()
