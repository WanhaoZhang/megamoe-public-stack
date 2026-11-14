"""Freeze uncertainty from independent, Timer-OFF selection experiments.

Only source histograms for the independent calibration bank are consumed.
No primary or held-out routes/times, learned future schedule or offline plans.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
from step_time_reference import contract,source_for,compiled_score,simulate
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=ROOT/'calibration/step-time-provisional.json');a=p.parse_args()
    model=json.loads(a.model.read_text());v=model['device_vector'];ref=contract();predictions=[]
    selection=json.loads((ROOT/'evidence/calibration-selection.json').read_text())
    for row in selection:
        policy=row['planner'];target=row['maxvio'];base=np.load(ROOT/f'routes/calibration/maxvio-{target:g}.npy')
        perstep=[];plan_hashes=[];domain=True
        for step in range(5,10):
            values=[]
            for layer in range(2):
                source=source_for(base,step,layer);proposal=getattr(ref,policy)(source,32,32)
                counts=proposal.quota.sum(axis=0);active=(counts>0).sum(axis=0)
                domain=domain and int(counts.max())<=v[18] and int(active.max())<=v[19] and int(active.min())>=v[60]
                predicted=compiled_score(source,v,proposal.copies,proposal.quota)
                if step==5:
                    independent=simulate(v,proposal.copies,proposal.quota)
                    assert predicted==independent,(policy,target,layer,predicted,independent)
                values.append(predicted*16/1e6)
                plan_hashes.append(hashlib.sha256(proposal.copies.tobytes()+proposal.quota.tobytes()).hexdigest())
            perstep.append(statistics.mean(values))
        predictions.append(dict(planner=policy,maxvio=target,observed_ms=row['mean_ms'],predicted_ms=statistics.mean(perstep),
            predicted_step_ms=perstep,measured_step_ms=[max(json.loads((ROOT/'evidence'/row['tag']/f'steps-rank{rank}.jsonl').read_text().splitlines()[step])['milliseconds'] for rank in range(8)) for step in range(5,10)],
            tag=row['tag'],inside_calibrated_domain=bool(domain),plan_hashes=plan_hashes))
        print(json.dumps({k:val for k,val in predictions[-1].items() if k not in ('plan_hashes','predicted_step_ms','measured_step_ms')}),flush=True)
    by={(x['planner'],x['maxvio']):x for x in predictions};baseline=model['initial_policy'];errors=[];paired=[]
    for row in predictions:
        base=by[baseline,row['maxvio']]
        delta=row['predicted_ms']-base['predicted_ms'];observed=row['observed_ms']-base['observed_ms']
        error=abs(delta-observed)
        if row['planner']!=baseline and row['inside_calibrated_domain'] and base['inside_calibrated_domain']:errors.append(error)
        paired.append(dict(planner=row['planner'],maxvio=row['maxvio'],predicted_delta_ms=delta,observed_delta_ms=observed,absolute_error_ms=error,
                           used_in_guard=row['planner']!=baseline and row['inside_calibrated_domain'] and base['inside_calibrated_domain']))
    if not errors:raise RuntimeError('no in-domain independent policy differential errors for the guard')
    # Three target-group leave-one-out diagnostic errors; report all residuals.
    # This is calibration, not an independent claim of final-strategy benefit.
    guard=float(np.quantile(errors,.9));background=max(0.,statistics.median(x['observed_ms']-x['predicted_ms'] for x in predictions if x['planner']==baseline))
    # Add actual Timer-OFF run-mean uncertainty (5 observed steps), not Timer data.
    mean_error=max(statistics.stdev(x['measured_step_ms'])/np.sqrt(5) for x in predictions if x['planner']==baseline)
    guard+=float(mean_error)
    model['device_vector'][28]=int(np.float32(round(background*1e6/16)))
    model['device_vector'][56]=int(np.float32(round(guard*1e6/16)))
    model['uncertainty']=dict(differential_p90_ms=guard-float(mean_error),baseline_mean_error_ms=float(mean_error),total_ms=guard,
        within_domain_pairs=len(errors),background_ms=background,selection_targets=[.3,7,18],future_or_primary_or_heldout_used=False)
    model['guard_calibration_sha256']=hashlib.sha256((ROOT/'evidence/calibration-selection.json').read_bytes()).hexdigest()
    (ROOT/'calibration/step-time-guard.json').write_text(json.dumps(dict(predictions=predictions,paired_errors=paired,**model['uncertainty']),indent=2))
    path=ROOT/'calibration/step-time-guarded-provisional.json';path.write_text(json.dumps(model,indent=2)+'\n')
    print('GUARD',json.dumps(model['uncertainty']),flush=True)
if __name__=='__main__':main()
