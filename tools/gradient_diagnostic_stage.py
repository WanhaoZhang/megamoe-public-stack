"""Separate baseline repeatability from differences between planner policies."""
import json
from controller import TASK,run_job


def main():
    results=[]
    for reference in ('native','moonep'):
        tag=f'diagnostic-{reference}-native-pair-vio5-v1'
        status=run_job('balanced','validate',tag,['--gradient-diagnostic','--root',str(TASK),
            '--planner',reference,'--maxvio','5','--steps','3'],1800)
        results.append(dict(tag=tag,reference=reference,candidate='native',**status))
        (TASK/'evidence/gradient-diagnostic-stage-v1.json').write_text(json.dumps(results,indent=2))
        if not status['qualified']:raise RuntimeError(tag)
    print('GRADIENT_DIAGNOSTIC_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()
