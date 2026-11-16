"""Formal stage runner for the calibrated policy; never fits or changes a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from controller import TASK,run_job
from analyze_matrix import audit_run
from campaign_records import resolved_existing,resolved_campaign
from recover_initialization import recover_campaign
from campaign_one import foreign_observation,write_json
from resource_admission import wait_if_occupied
from controller import source_identity

TARGETS=(.2,1,5,10,15,20)

def strongest_existing():
    rows=resolved_existing(TASK)
    freeze=json.loads((TASK/'evidence/BASELINE-FREEZE.json').read_text())
    if len(rows)!=90 or any(x['state']!='pass' for x in rows):raise RuntimeError('existing primary matrix is incomplete')
    scores={}
    for policy in dict.fromkeys(x['planner'] for x in rows):
        means=[]
        for vio in TARGETS:
            selected=[x for x in rows if x['planner']==policy and x['maxvio']==vio]
            if len(selected)!=3:raise RuntimeError('not three existing repeats')
            means.append(statistics.mean(audit_run(TASK/'evidence'/x['tag'],freeze)['mean_ms'] for x in selected))
        scores[policy]=math.exp(statistics.mean(map(math.log,means)))
    best=min(scores,key=scores.get)
    return dict(planner=best,scores_geomean_ms=scores,selection='geometric mean of six point mean complete-step times',
                all_existing_points_required=True)

def run_stage(name,planners,repeats,first_repeat,bank,models):
    model_sha={p:hashlib.sha256(Path(path).read_bytes()).hexdigest() for p,path in models.items()}
    contract=dict(name=name,planners=planners,repeats=repeats,first_repeat=first_repeat,bank=str(bank),
                  model_paths=models,model_sha=model_sha,capacity=163840,targets=TARGETS,steps=25,warmup=5,
                  online=True,ordinary_executor=True,fixed_owners=True,rotate_order=True,
                  fourth_pair_order='alternate across MaxVio')
    path=TASK/f'configs/campaign-{name}.json';result=TASK/f'evidence/campaign-{name}.json'
    canonical=json.loads(json.dumps(contract))
    if path.exists() and json.loads(path.read_text())!=canonical:raise RuntimeError('frozen stage contract differs')
    path.write_text(json.dumps(contract,indent=2));rows=json.loads(result.read_text()) if result.exists() else []
    done={row['tag'] for row in rows}
    for repeat in range(first_repeat,first_repeat+repeats):
        offset=(repeat-1)%len(planners);order=planners[offset:]+planners[:offset]
        for point_index,vio in enumerate(TARGETS):
            if repeats==1 and len(planners)==2:
                offset=(repeat-1+point_index)%2;order=planners[offset:]+planners[:offset]
            for policy in order:
                tag=f'{name}-vio{vio:g}-{policy}-r{repeat}'
                if tag in done:continue
                for p,sha in model_sha.items():
                    if hashlib.sha256(Path(models[p]).read_bytes()).hexdigest()!=sha:raise RuntimeError('calibration model changed after freeze')
                extra=['--planner',policy,'--recv-capacity','163840','--maxvio',str(vio),'--steps','25','--route-bank',str(bank)]
                if policy in models:extra+=['--cost-model',str(models[policy])]
                wait_if_occupied(tag)
                directory=TASK/'evidence'/tag
                if directory.exists():
                    if not (directory/'status.json').exists():raise RuntimeError('partial formal attempt requires review: '+tag)
                    status=json.loads((directory/'status.json').read_text())
                    if status['command'][-len(extra):]!=extra or json.loads((directory/'source-identity.json').read_text())!=source_identity():
                        raise RuntimeError('unledgered receipt has different request/source: '+tag)
                else:status=run_job('balanced','train',tag,extra)
                text=(TASK/f'evidence/{tag}/stdout.log').read_text(errors='replace')
                state='pass' if status['qualified'] else 'failed'
                if not status['qualified']:
                    if 'out of memory' in text.lower():state='oom'
                    elif status['exit'] in (124,137):state='timeout'
                    elif 'device plan failed' in text:state='invalid_plan'
                if status['peak_driver_MiB']>90*1024:state='memory_gate_exceeded'
                row=dict(tag=tag,planner=policy,maxvio=vio,repeat=repeat,state=state,**status)
                observation=foreign_observation(row)
                if observation:row['foreign_gpu_observation']=observation
                rows.append(row);write_json(result,rows);done.add(tag)
                print(json.dumps(dict(stage=name,completed=len(rows),total=repeats*len(planners)*6,last=row)),flush=True)
                if state in ('oom','timeout','memory_gate_exceeded'):raise RuntimeError(f'run safety gate: {tag}: {state}')
                if observation:raise RuntimeError('foreign overlap recorded; wait before continuing: '+tag)
    return recover_campaign(name,repeats*len(planners)*6)

def stable_points(primary,fourth,best):
    rows=primary+fourth;points=[]
    freeze=json.loads((TASK/'evidence/BASELINE-FREEZE.json').read_text())
    for vio in TARGETS:
        gains=[]
        for repeat in range(1,5):
            pair={p:[r for r in rows if r['planner']==p and r['maxvio']==vio and r['repeat']==repeat] for p in ('step_time',best)}
            if any(len(v)!=1 or v[0]['state']!='pass' for v in pair.values()):continue
            baseline=audit_run(TASK/f'evidence/{pair[best][0]["tag"]}',freeze)['mean_ms']
            new=audit_run(TASK/f'evidence/{pair["step_time"][0]["tag"]}',freeze)['mean_ms']
            gains.append(1-new/baseline)
        points.append(dict(maxvio=vio,repeat_reductions=gains,stable=len(gains)==4 and min(gains)>0 and statistics.median(gains)>=.01))
    return points

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('primary','fourth','heldout','ablation'));ap.add_argument('--model',type=Path,required=True);ap.add_argument('--ablation',choices=('no_gradient_service','linear_kernel','uniform_topology'));a=ap.parse_args()
    acceptance=TASK/'evidence/step-time-qualification.json'
    if not acceptance.exists() or not json.loads(acceptance.read_text())['passed']:raise RuntimeError('step_time correctness/calibration is not accepted')
    approval=json.loads(acceptance.read_text())
    if hashlib.sha256(a.model.read_bytes()).hexdigest() not in approval['approved_model_shas']:
        raise RuntimeError('the selected model was not part of the independent qualification')
    if a.stage!='ablation' and a.ablation is not None:
        raise RuntimeError('an ablation flag is valid only for the conditional ablation stage')
    variant=json.loads(a.model.read_text()).get('variant')
    expected_variant=a.ablation if a.stage=='ablation' else 'full'
    if variant!=expected_variant:
        raise RuntimeError('model variant does not match the requested formal stage')
    if a.stage=='primary':run_stage('primary-step-time',['step_time'],3,1,TASK/'routes',{'step_time':str(a.model)})
    else:
        selection=strongest_existing();path=TASK/'configs/strongest-existing.json'
        if path.exists() and json.loads(path.read_text())!=selection:raise RuntimeError('strongest-baseline selection changed')
        path.write_text(json.dumps(selection,indent=2));best=selection['planner']
        primary=resolved_existing(TASK)+resolved_campaign(TASK,'primary-step-time',18)
        if len(primary)!=108:raise RuntimeError('complete the 108 primary attempts first')
        if a.stage=='fourth':
            fourth=run_stage('fourth-paired',[best,'step_time'],1,4,TASK/'routes',{'step_time':str(a.model)})
            (TASK/'analysis/stability-four-rounds.json').write_text(json.dumps(stable_points(primary,fourth,best),indent=2))
        elif a.stage=='heldout':
            for bank in ('heldout-a','heldout-b'):run_stage(bank,[best,'step_time'],3,1,TASK/f'routes/{bank}',{'step_time':str(a.model)})
        else:
            stable=json.loads((TASK/'analysis/stability-four-rounds.json').read_text())
            if not any(p['stable'] for p in stable):raise RuntimeError('no stable point: conditional ablation must not run')
            if a.ablation is None:raise RuntimeError('choose the prespecified ablation')
            run_stage(f'ablation-{a.ablation}',['step_time'],3,1,TASK/'routes',{'step_time':str(a.model)})

if __name__=='__main__':main()
