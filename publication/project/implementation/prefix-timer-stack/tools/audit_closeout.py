"""Read-only120-run closeout audit; never asserts the canceled protocol passed."""
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
from analyze_matrix import audit_run
from campaign_records import resolved_existing,resolved_campaign
from controller import source_identity
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(json.dumps(load(p),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    request=load(ROOT/'evidence/USER-CLOSEOUT-REQUEST.json');release=load(ROOT/'evidence/USER-CLOSEOUT-RELEASE.json')
    assert request['no_further_device_runs'] and release['passed'] and not release['remaining_registered_task_processes'] and not release['device_reset']
    stages={'primary-existing':resolved_existing(ROOT),'primary-step-time':resolved_campaign(ROOT,'primary-step-time',18),
        'fourth-paired':resolved_campaign(ROOT,'fourth-paired',12)}
    freeze=load(ROOT/'evidence/BASELINE-FREEZE.json');final=load(ROOT/'evidence/STEP-TIME-FREEZE.json')
    assert source_identity()==final['source']
    parameters={};inputs={};routes={};runtime=None;checked=[]
    for stage,runs in stages.items():
        for run in runs:
            directory=ROOT/'evidence'/run['tag'];metrics=audit_run(directory,freeze)
            assert metrics['receive_capacity']==163840
            for rank in range(8):
                for name,prefix,table,key in [('parameter','initial',parameters,rank),('input','input-identity',inputs,rank),
                        ('route','route-identity',routes,(run['maxvio'],rank))]:
                    value=digest(directory/f'{prefix}-rank{rank}.json')
                    if key in table:assert table[key]==value,(stage,run['tag'],rank,name)
                    else:table[key]=value
                libs=load(directory/f'libraries-rank{rank}.json')
                actual={Path(p).name:v['sha256'] for p,v in libs['libraries'].items() if Path(p).name=='libshmem.so' or Path(p).name.startswith('libstdc++.so')}
                assert 'libshmem.so' in actual and any(x.startswith('libstdc++.so') for x in actual)
                current=dict(python=libs['python'],torch=libs['torch'],ascend_home=libs['ascend_home'],runtime=actual,sitecustomize=libs['sitecustomize'])
                if runtime is None:runtime=current
                assert runtime==current,(run['tag'],rank,'runtime')
            checked.append(dict(stage=stage,tag=run['tag'],mean_ms=metrics['mean_ms'],peak_hbm_gib=metrics['peak_hbm_gib']))
    assert len(checked)==120
    decisions=load(ROOT/'analysis/planner-decisions-summary.json')
    primary=[x for x in decisions['points'] if x['stage']=='primary-step-time']
    assert sum(x['global_invocations'] for x in primary)==900 and sum(x['adopted'] for x in primary)==0
    warning=load(ROOT/'evidence/trajectory-audit-strict-fourth-pause.json')
    assert warning['runs']==120 and not warning['passed']
    warning_tags=[r['tag'] for r in warning['rows'] if not r['passed']]
    assert warning_tags==['fourth-paired-vio5-step_time-r4']
    diagnostics=[]
    for tag in ('diagnostic-fixed-state-pair-vio5','diagnostic-fixed-state-pair-aligned-vio5'):
        d=ROOT/'evidence'/tag;status=load(d/'status.json');assert not status['qualified']
        counts=[];records=[]
        for rank in range(8):
            p=d/f'paired-rank{rank}.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
            counts.append(len(rows));records+=rows
        diagnostics.append(dict(tag=tag,completed_steps_by_rank=counts,completed_25_step_gate=False,
            state='user_canceled' if 'aligned' in tag else 'hccl_failure',exit=status['exit'],
            maxima_so_far={k:max(x[k] for x in records) for k in ('loss_relative','fixed_weight_global_norm_relative',
                'complete_parameter_gradient_relative_l2','critical_max_gradient_relative_l2','all_parameter_max_gradient_relative_l2')}))
    result=dict(passed=True,meaning='user-requested bounded closeout and retained-data identity audit; not full protocol acceptance',
        time_utc=datetime.now(timezone.utc).isoformat(),original_protocol_complete=False,user_requested_closeout=True,
        qualified_retained_runs=len(checked),rank_runs=len(checked)*8,measured_world_steps=len(checked)*20,
        stages={k:len(v) for k,v in stages.items()},parameters_and_inputs_match_all_120=True,route_identity_groups=len(routes),
        actual_runtime=runtime,frozen_final_sources_match=True,primary_new_plan_adoptions=0,primary_planner_calls=900,
        trajectory_warning_tags=warning_tags,trajectory_warning_resolved=False,diagnostics=diagnostics,
        canceled_stages=request['canceled_stages'],release_verified=True,checks=checked)
    (ROOT/'CLOSEOUT-AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()
