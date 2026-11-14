"""Read-only investigation of the retained fourth-round norm mismatch."""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FAILED='fourth-paired-vio5-step_time-r4'
def digest(p):return hashlib.sha256(json.dumps(json.loads(p.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    tags=['primary-existing-vio5-moonep-r1','primary-existing-vio5-native-r1','primary-existing-vio5-native-r2','primary-existing-vio5-native-r3',
          'primary-step-time-vio5-step_time-r1','primary-step-time-vio5-step_time-r2','primary-step-time-vio5-step_time-r3',FAILED,'fourth-paired-vio5-ultra_ep-r4']
    data={};identities={}
    for tag in tags:
        d=ROOT/'evidence'/tag
        data[tag]=[[json.loads(x) for x in (d/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
        identities[tag]=[{kind:digest(d/f'{prefix}-rank{rank}.json') for kind,prefix in [('parameter','initial'),('input','input-identity'),('route','route-identity')]} for rank in range(8)]
    assert all(x==identities[FAILED] for x in identities.values()),'initial parameter/input/route identities do not match'
    steps=[]
    for step in range(25):
        ref=data[tags[0]][0][step];current=data[FAILED][0][step]
        row=dict(step=step,loss={t:data[t][0][step]['loss'] for t in tags},norm={t:data[t][0][step]['grad_norm'] for t in tags},
            failed_vs_moon_norm_relative=(current['grad_norm']/ref['grad_norm']-1),failed_vs_moon_loss_relative=(current['loss']/ref['loss']-1))
        steps.append(row)
        if abs(row['failed_vs_moon_norm_relative'])>.005:print('TRAJECTORY_STEP',json.dumps(row),flush=True)
    stats=[]
    for tag in tags:
        rows=data[tag]
        stats.append(dict(tag=tag,max_rank_norm_spread=max(max(rows[r][s]['grad_norm'] for r in range(8))-min(rows[r][s]['grad_norm'] for r in range(8)) for s in range(25)),
            norm_max_vs_moon=max(abs(rows[0][s]['grad_norm']/data[tags[0]][0][s]['grad_norm']-1) for s in range(25)),
            norm_max_vs_failed=max(abs(rows[0][s]['grad_norm']/data[FAILED][0][s]['grad_norm']-1) for s in range(25))))
    decisions=[np.load(ROOT/'evidence'/t/'planner-decisions-rank0.npy') for t in tags if t.startswith('primary-step') or t==FAILED]
    result=dict(failed_tag=FAILED,identities_identical=True,actual_step_time_decisions_identical=all(np.array_equal(decisions[0],x) for x in decisions[1:]),
                statistics=stats,steps=steps)
    (ROOT/'analysis/fourth-vio5-trajectory-investigation.json').write_text(json.dumps(result,indent=2))
    print('TRAJECTORY_INVESTIGATION',json.dumps({k:v for k,v in result.items() if k!='steps'}),flush=True)
if __name__=='__main__':main()
