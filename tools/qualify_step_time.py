"""Qualify the final frozen plugin/models before any new formal timing cell."""
import hashlib,json,math,sys
from pathlib import Path
from controller import TASK,run_job,source_identity
from resource_admission import wait_if_occupied
from campaign_one import foreign_observation

def checked(mode,tag,extra):
    directory=TASK/'evidence'/tag
    if directory.exists():
        status=json.loads((directory/'status.json').read_text())
        assert status['qualified'] and status['source_unchanged']
        assert json.loads((directory/'source-identity.json').read_text())==source_identity(),'qualification source changed'
    else:
        wait_if_occupied(tag);status=run_job('balanced',mode,tag,extra)
    assert status['qualified'] and status['peak_driver_MiB']<=90*1024,status
    overlap=foreign_observation(dict(tag=tag,**status))
    if overlap:raise RuntimeError('independent qualification overlapped another GPU workload: '+tag)
    return dict(tag=tag,mode=mode,**status)

def main():
    assert json.loads((TASK/'evidence/CALIBRATION-ACCEPTANCE.json').read_text())['passed']
    frozen=json.loads((TASK/'evidence/step-time-models-frozen.json').read_text());models={}
    for path,digest in frozen['models'].items():
        assert hashlib.sha256((TASK/path).read_bytes()).hexdigest()==digest
        models[json.loads((TASK/path).read_text())['variant']]=path
    results=[]
    for mode,tag,extra in [
        ('plan','step-time-plan-semantic-final',['--step-time-gate','--model',str(TASK/models['full'])]),
        ('plan','step-time-plan-main-shape-final',['--step-time-gate','--main-shape','--model',str(TASK/models['full'])]),
        ('validate','step-time-numeric-final',['--planner','step_time','--cost-model',str(TASK/'configs/step-time-semantic-numeric.json'),'--semantic-cost-model','--recv-capacity','4096']),
        ('pool','step-time-pool-final',['--planner','step_time','--cost-model',str(TASK/'configs/step-time-semantic-pool.json'),'--semantic-cost-model',
            '--experts','256','--slots','32','--capacity','8192','--restore','panel','--checkpoint','--parameter-fp32','--limit','10','--require-replicas']),
        ('train','step-time-three-adam-final',['--planner','step_time','--cost-model',str(TASK/models['full']),'--recv-capacity','163840',
            '--maxvio','7','--steps','3','--route-bank',str(TASK/'routes/calibration')])]:
        results.append(checked(mode,tag,extra))
        (TASK/'evidence/step-time-qualification-progress.json').write_text(json.dumps(results,indent=2))
    # Verify real FP32 parameters/gradients/moments, parameter updates and exact
    # one original plan with two checkpoint executions per layer/step.
    directory=TASK/'evidence/step-time-three-adam-final';base=TASK/'evidence/calibration-select-vio7-native'
    plugin=TASK/'source/native-balanced/ascend_deepep/lib/libascend_deepep_step_time_planner.so'
    plugin_sha=hashlib.sha256(plugin.read_bytes()).hexdigest()
    trajectory=[]
    for rank in range(8):
        planning=json.loads((directory/f'planning-audit-rank{rank}.json').read_text())
        assert sum(x['calls'] for x in planning)==6 and all(x['live']==0 and x['replay']==dict(original=6,recompute=6,backward=6) for x in planning)
        assert all(x['model_sha'] in frozen['models'].values() for x in planning)
        libraries=json.loads((directory/f'libraries-rank{rank}.json').read_text())
        assert len(libraries['shmem'])==1 and [x['sha256'] for p,x in libraries['libraries'].items() if p.endswith('/libascend_deepep_step_time_planner.so')]==[plugin_sha]
        updates=json.loads((directory/f'updates-rank{rank}.json').read_text());assert len(updates)==3 and all(x['changed'] for x in updates)
        rows=[json.loads(x) for x in (directory/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        old=[json.loads(x) for x in (base/f'steps-rank{rank}.jsonl').read_text().splitlines()][:3]
        for i,(new,ref) in enumerate(zip(rows,old)):
            assert all(math.isfinite(new[k]) for k in ('loss','grad_norm','milliseconds'))
            loss=abs(new['loss']-ref['loss'])/max(abs(ref['loss']),1e-6)
            grad=abs(new['grad_norm']-ref['grad_norm'])/max(abs(ref['grad_norm']),1e-6)
            assert loss<=.004 and grad<=.02,(rank,i,loss,grad)
            trajectory.append(dict(rank=rank,step=i,loss_relative=loss,grad_norm_relative=grad))
    record=dict(passed=True,approved_model_shas=list(frozen['models'].values()),plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
        models=models,checks=results,trajectory=trajectory,semantic_fixtures_excluded_from_formal=True,
        independent_cpu=json.loads((TASK/'evidence/step-time-cpu.json').read_text()))
    (TASK/'evidence/step-time-qualification.json').write_text(json.dumps(record,indent=2))
    freeze=dict(source=source_identity(),plugin_sha256=record['plugin_sha256'],approved_model_shas=record['approved_model_shas'],
        qualification_sha256=hashlib.sha256((TASK/'evidence/step-time-qualification.json').read_bytes()).hexdigest())
    (TASK/'evidence/STEP-TIME-FREEZE.json').write_text(json.dumps(freeze,indent=2))
    print('STEP_TIME_QUALIFIED',record['plugin_sha256'],flush=True)
if __name__=='__main__':main()
