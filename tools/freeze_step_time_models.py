"""Freeze four models after independent fitting and real device planning costs."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-tag',default='step-time-plan-main-shape-v1');a=p.parse_args()
    directory=ROOT/'evidence'/a.plan_tag;status=json.loads((directory/'status.json').read_text());assert status['qualified']
    ranks=[json.loads((directory/f'rank{r}.json').read_text()) for r in range(8)]
    assert all(x['passed'] and x['main_shape'] for x in ranks)
    coefficients=[];summary=[]
    for scenario in range(3):
        trials=[]
        for repeat in (1,2):
            new=max(x['records'][scenario]['timings'][repeat]['device_ms'] for x in ranks)
            old=max(x['records'][scenario]['initial_policy_timing']['device_ms'] for x in ranks)
            evaluations=int(ranks[0]['records'][scenario]['stats'][0]);rounds=int(ranks[0]['records'][scenario]['stats'][1])
            denom=evaluations+2*rounds+1
            coefficients.append(max(0.,new-old)/denom);trials.append(dict(new_ms=new,initial_ms=old,extra_ms=new-old,charged_units=denom))
        summary.append(dict(scenario=scenario,warm_trials=trials))
    # Worst calibration observation, not a clock-frequency or theoretical cost.
    increment=max(coefficients);model=json.loads((ROOT/'calibration/step-time-guarded-provisional.json').read_text())
    model['provisional']=False;model['device_vector'][31]=int(np.float32(round(increment*1e6/16)))
    model['search_budget']=dict(rounds=1,candidates_per_round=4,measured_device_increment_ms_per_charged_unit=increment,
        charged_units='evaluations+2*rounds+1',two_layers_charged=True,selection='smallest prespecified four-class neighbourhood; measured roughly4-5ms per invocation',
        main_shape_measurements=summary,source_status_sha256=sha(directory/'status.json'))
    result={}
    for variant,code in [('full',0),('no_gradient_service',1),('linear_kernel',2),('uniform_topology',3)]:
        current=json.loads(json.dumps(model));current['variant']=variant;current['device_vector'][16]=code
        if variant=='uniform_topology':current['topology_ablation_unresolvable']=True
        path=ROOT/f'configs/step-time-{variant.replace("_","-")}.json'
        if path.exists():raise RuntimeError('do not overwrite a frozen policy model: '+str(path))
        path.write_text(json.dumps(current,indent=2)+'\n');result[str(path.relative_to(ROOT))]=sha(path)
    (ROOT/'evidence/step-time-models-frozen.json').write_text(json.dumps(dict(models=result,planning_increment=increment,uncertainty=model['uncertainty']),indent=2))
    # These are deliberately synthetic semantic fixtures, forbidden in train.py.
    for name,t,k,h,f,cap in [('numeric',128,8,256,256,4096),('pool',257,2,512,512,8192)]:
        current=json.loads(json.dumps(model));v=current['device_vector'];current['variant']='full';current['semantic_only']=True;current['provisional']=True
        current['shape'].update(tokens=t,topk=k,hidden=h,ffn=f);v[5:10]=[t,k,h,f,cap]
        v[18:20]=[10000000,64];v[60]=v[31]=v[56]=0;v[35]=1
        v[36:38]=[((2*f+127)//128)*((h+255)//256),((h+127)//128)*((f+255)//256)]
        (ROOT/f'configs/step-time-semantic-{name}.json').write_text(json.dumps(current,indent=2)+'\n')
    print('STEP_TIME_MODELS_FROZEN',json.dumps(result),increment,flush=True)
if __name__=='__main__':main()
