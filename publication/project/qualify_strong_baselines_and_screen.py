"""Strengthen routing-agnostic baselines before the first 18-run screen."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
deadline=time.monotonic()+7200
while True:
    path=root/'observation-ablation.json'
    data=json.loads(path.read_text()) if path.exists() else dict(jobs=[])
    if any(job['status']=='failed' for job in data['jobs']):raise RuntimeError('Prior ablation failed; source not changed')
    if len(data['jobs'])==4 and all(job['status']=='completed' for job in data['jobs']):break
    if time.monotonic()>deadline:raise RuntimeError('Ablation did not finish; refusing a source update during an active run')
    time.sleep(5)
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
proofs=[]
for policy in ('fixed_native','fifo_native'):
    tag=f'no-probe-baseline-{policy}-adam3'
    run=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',tag,'--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
        '--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--grad-sync','--dump-grad-step0'])
    comparison=None
    if run.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        comparison=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
            f'{remote}/evidence/gradient-w2-legacy',f'{remote}/evidence/{tag}',
            '--step0-only','--out',destination]).returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    proofs.append(dict(tag=tag,policy=policy,execution_exit=run.returncode,comparison_exit=comparison,
                       routing_statistics_required=False,formal=False))
    (root/'strong-baseline-qualification.json').write_text(json.dumps(proofs,indent=2))
    if run.returncode or comparison:raise RuntimeError('Stronger baseline failed; screening remains closed')

gates=dict(evidence_paths_relative_to='project artifact root, parent of local stack',
    cpu_contract=dict(passed=True,evidence='evidence/cpu-contract-current.log',tests=9),
    source_archive_integrity=dict(passed=True,evidence='evidence/source-freeze-tests.log',tests=3),
    same_weight_full_gradient=dict(passed=True,evidence='strong-baseline-qualification.json',
        other_policies='ordered-w2-complete.json',threshold=.02),
    legacy_w1_full_gradient=dict(passed=True,evidence='observation-gates.json'),
    device_scheduler_reference=dict(passed=True,evidence='evidence/device-scheduler-reference-v3/status.json'),
    primary_online_adam3=dict(passed=True,evidence='pilot-results.json',actual_updates=25),
    common_memory_budget=dict(passed=True,evidence='pilot-results.json',ceiling_gib=90),
    group_protocol_recovery=dict(passed=True,evidence='evidence/grouped-domain-rejection/status.json'),
    timer_v2_diagnostic=dict(passed=True,evidence='timer-v2-matched-qualification.json'),
    formal_matrix_enabled=False,afore_cache_implemented=False,joint_strategy_implemented=False)
(root/'stack/configs/temporal-gates.json').write_text(json.dumps(gates,indent=2))
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
arms=(('legacy_w1','legacy',1),('legacy_w2','legacy',2),('fixed_native','fixed_native',2),
      ('fifo_native','fifo_native',2),('easy_home','easy_home',2),('easy_independent','easy_independent',2))
jobs=[dict(tag=f'screen-v1-vio10-{pattern}-{label}-r1',label=label,policy=policy,pattern=pattern,
           window=window,maxvio=10,repeat=1,independent_repeats=1,status='planned',
           optimizer_updates=25,warmup_updates=5,measured_updates=20,formal=False)
      for pattern in ('same','alternating','random') for label,policy,window in arms]
def save():
    (root/'screening-results.json').write_text(json.dumps(dict(stage='screening',jobs=jobs,
        full_plan_complete=False,formal_matrix=False,afore_cache_included=False,
        joint_policy_included=False,route_agnostic_baselines_do_not_probe=True),indent=2))
save()
for job in jobs:
    job['status']='running';save()
    run=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',job['tag'],'--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
        '--microbatches','4','--forward-window',str(job['window']),'--recv-capacity','262144',
        '--maxvio','10','--planner','native','--temporal-pattern',job['pattern'],
        '--temporal-policy',job['policy'],'--temporal-scheduler','ascendc','--grad-sync'])
    path=root/'evidence'/job['tag']/'status.json';status=json.loads(path.read_text()) if path.exists() else {}
    job.update(status='completed' if run.returncode==0 and status.get('qualified') and
        status.get('peak_driver_MiB',float('inf'))<=90*1024 else 'failed',receipt=status)
    save()
    if job['status']!='completed':raise RuntimeError('Screen stopped at failure/common-memory gate: '+job['tag'])
