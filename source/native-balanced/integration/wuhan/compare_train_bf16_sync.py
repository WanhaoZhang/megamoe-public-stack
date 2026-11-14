"""Offline same-input S9 migration comparison; never initializes an NPU."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'test/megamoe_planner_test'))
from compare_results import metric


def admitted(folder):
    gate = json.loads((folder/'host-acceptance.json').read_text())
    if gate != dict(pre_gate=True, post_gate=True, run_exit=0, state='COMPLETED'):
        raise ValueError('incomplete/interfered device run: '+str(folder))


def ranks(folder):
    result = [json.loads((folder/f'rank{i}.json').read_text()) for i in range(8)]
    if any(not r['pass_'] or r['rank'] != i or r['world_size'] != 8 for i,r in enumerate(result)):
        raise ValueError('require all eight successful rank receipts')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='baseline sync-evidence directory')
    p.add_argument('--candidate', type=Path, required=True, help='candidate sync-evidence directory')
    p.add_argument('--baseline-output-control', type=Path,
                   help='optional untimed replay control directory; its receipts must exactly match the original baseline')
    p.add_argument('--baseline-host', required=True)
    p.add_argument('--candidate-host', required=True)
    p.add_argument('--performance', nargs='*', type=Path, default=[], help='ordered run directories with performance/rankN.json')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(1)
    records = []
    for folder in (a.baseline,a.candidate):
        capture = folder/'capture-a1'
        admitted(capture)
        acceptance = json.loads((capture/'capture/ACCEPTANCE.json').read_text())
        if not acceptance['pass_'] or not acceptance['five_outputs_bitwise_TimerOFF_ON']:
            raise ValueError('TimerOFF/ON five-output comparison failed')
        records.append(ranks(capture/'capture/control'))
    output_dirs = [folder/'capture-a1/capture/control' for folder in (a.baseline,a.candidate)]
    if a.baseline_output_control:
        admitted(a.baseline_output_control.parent)
        restored = ranks(a.baseline_output_control)
        for old,new in zip(records[0],restored):
            for key in ('config','inputs','source','outputs','libraries','plan_tensors','copies'):
                if old[key] != new[key]:raise ValueError('baseline output replay changed '+key)
            if new['timer'] or not new['diagnostic_only'] or new['samples']:
                raise ValueError('baseline output replay must be untimed TimerOFF')
        output_dirs[0] = a.baseline_output_control
    details = []
    for rank,(old,new) in enumerate(zip(*records)):
        for key in ('config','inputs','source','plan_tensors','copies'):
            if old[key] != new[key]:
                raise ValueError(f'rank{rank}: changed {key}; not a paired migration comparison')
        tensors = [torch.load(folder/f'outputs-rank{rank}.pt',
                              map_location='cpu',weights_only=True) for folder in output_dirs]
        # Use the already established bridge predicates and additional 5% L2
        # guard. The upstream SwiGLU numerical correction is not a bitwise claim.
        result = {key:metric(tensors[1][key].float().numpy(),tensors[0][key].float().numpy(),key=='y')
                  for key in ('y','dx','dp','dw1','dw2')}
        details.append(dict(rank=rank,outputs=result))
        del tensors
    timing = []
    for folder in a.performance:
        admitted(folder)
        run = ranks(folder/'performance')
        matching = [i for i,ref in enumerate(records) if run[0]['libraries']==ref[0]['libraries']]
        if len(matching)!=1:raise ValueError('unknown or ambiguous performance binary')
        version=matching[0]
        for measured,ref in zip(run,records[version]):
            for key in ('config','inputs','outputs','libraries','plan_tensors'):
                if measured[key]!=ref[key]:raise ValueError('uncertified performance '+key)
            if measured['timer'] or measured['diagnostic_only'] or not measured['samples']:
                raise ValueError('formal performance requires TimerOFF samples')
        count=len(run[0]['samples'])
        if any(len(r['samples'])!=count for r in run):raise ValueError('unequal sample counts')
        phases={}
        for key in ('full_ms','forward_ms','backward_ms'):
            values=[max(r['samples'][i][key] for r in run) for i in range(count)]
            phases[key]=dict(p50=float(np.median(values)),p95=float(np.percentile(values,95)),
                             min=min(values),max=max(values),samples=values)
        timing.append(dict(directory=str(folder),version=('baseline','candidate')[version],phases=phases))
    passed=all(m['pass_'] for rank in details for m in rank['outputs'].values())
    result=dict(pass_=passed,precision=details,performance=timing,
                configuration=records[0][0]['config'],paired_inputs_and_plans=True,
                hosts=dict(baseline=a.baseline_host,candidate=a.candidate_host),
                same_host=a.baseline_host==a.candidate_host,
                baseline_output_replay=str(a.baseline_output_control) if a.baseline_output_control else None,
                libraries=dict(baseline=records[0][0]['libraries'],candidate=records[1][0]['libraries']),
                scope=records[0][0]['scope'],tolerance_source='compare_results.metric (unchanged)')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(pass_=passed,report=str(a.out),
        max_relative_l2=max(m['relative_l2'] for r in details for m in r['outputs'].values()),
        runs=[dict(version=r['version'],p50=r['phases']['full_ms']['p50'],p95=r['phases']['full_ms']['p95']) for r in timing])))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
