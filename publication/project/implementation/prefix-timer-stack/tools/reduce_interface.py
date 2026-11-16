"""Separate original-capacity interface diagnostics from the formal matrix."""
import json
from pathlib import Path
import statistics
from analyze_matrix import world_max_steps
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'evidence/interface-regression.json').read_text())
output=[];trajectories={}
for row in rows:
    if not row['qualified']:raise RuntimeError('unqualified interface diagnostic')
    directory=ROOT/'evidence'/row['tag']
    streams=[[json.loads(x) for x in (directory/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
    times=world_max_steps(streams)[5:]
    for rank in range(8):
        plan=json.loads((directory/f'planning-audit-rank{rank}.json').read_text())
        expected=100 if row['interface']=='legacy' else 50
        if sum(x['calls'] for x in plan)!=expected or any(x['live'] for x in plan):raise ValueError('wrong planner/replay count')
        trajectories[row['maxvio'],row['interface'],rank]=streams[rank]
    output.append(dict(interface=row['interface'],maxvio=row['maxvio'],receive_capacity=131072,
        mean_ms=statistics.mean(times),p50_ms=statistics.median(times),peak_hbm_gib=row['peak_driver_MiB']/1024,
        planner_calls_per_process=expected,independent_repeats=1,tag=row['tag']))
differences=[]
for vio in (.2,20):
    maximum_loss=maximum_norm=0.
    for rank in range(8):
        old=trajectories[vio,'legacy',rank];new=trajectories[vio,'unified',rank]
        for a,b in zip(old,new):
            maximum_loss=max(maximum_loss,abs(a['loss']-b['loss'])/max(abs(a['loss']),1e-12))
            maximum_norm=max(maximum_norm,abs(a['grad_norm']-b['grad_norm'])/max(abs(a['grad_norm']),1e-12))
    if maximum_loss>.004 or maximum_norm>.02:raise ValueError('old/new interface trajectory differs')
    differences.append(dict(maxvio=vio,maximum_relative_loss_difference=maximum_loss,maximum_relative_global_norm_difference=maximum_norm))
result=dict(passed=True,rows=output,trajectory_checks=differences,
            limitation='One independent full-length diagnostic pair per endpoint; this is not a stable performance gain claim.',
            isolation='131072 legacy vs131072 unified isolates interface/replay changes;163840 primary uses a larger common receive budget.')
(ROOT/'analysis/interface-regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
