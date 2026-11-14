"""Run independent calibration only after the complete frozen existing matrix."""
import json
from pathlib import Path
import statistics
import subprocess
import hashlib
import sys
from controller import TASK,run_job,source_identity
from campaign_records import resolved_existing,init_failure,verified_contention
from campaign_one import foreign_observation
from resource_admission import wait_if_occupied

POLICIES=('moonep','native','ultra_ep','echo','eplb_fixed')
NEW_JOBS=0
class StagePause(Exception):pass

def checked(tag,extra,worker=None,mode='train'):
    global NEW_JOBS
    files={};snapshots={}
    names=['calibrate_links.py'] if mode=='links' else (['capture_diagnostics.py','capture_framework.py'] if '--diagnostic-timer' in extra else [])
    for name in names:
        snapshots[name]=(TASK/'tools'/name).read_bytes()
        files[name]=hashlib.sha256(snapshots[name]).hexdigest()
    request=dict(arguments=extra,worker=worker,mode=mode,core_sources=source_identity(),diagnostic_files=files)
    signature=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:12]
    base=tag
    for number in range(4):
        directory=TASK/'evidence'/tag;status_file=directory/'status.json';request_file=directory/'diagnostic-request.json'
        if directory.exists():
            if not status_file.exists():raise RuntimeError('partial diagnostic attempt requires review: '+tag)
            same=request_file.exists() and json.loads(request_file.read_text())==request
            old=json.loads(status_file.read_text());row=dict(tag=tag,**old)
            observed=foreign_observation(row)
            if observed:row['foreign_gpu_observation']=observed
            if same and old['qualified'] and old['peak_driver_MiB']<=90*1024 and not verified_contention(TASK,row):
                return dict(cached=True,mode=mode,**row)
            if not same:
                tag=base+'-revision-'+signature;continue
            if not init_failure(TASK,row) and not verified_contention(TASK,row):raise RuntimeError('diagnostic failure needs correction: '+tag)
            tag=base+f'-infrastructure-retry{number+1}';continue
        if '--one' in sys.argv and NEW_JOBS>=1:raise StagePause()
        wait_if_occupied(tag)
        status=run_job('balanced',mode,tag,extra,1800,worker=worker)
        NEW_JOBS+=1
        source_copy=directory/'diagnostic-sources';source_copy.mkdir(exist_ok=False)
        for name,payload in snapshots.items():(source_copy/name).write_bytes(payload)
        files_unchanged=all((TASK/'tools'/name).read_bytes()==payload for name,payload in snapshots.items())
        request_file.write_text(json.dumps(request,indent=2))
        record=dict(tag=tag,mode=mode,request_signature=signature,diagnostic_files_unchanged=files_unchanged,**status)
        observation=foreign_observation(record)
        if observation:record['foreign_gpu_observation']=observation
        ledger=TASK/'evidence/CALIBRATION-ATTEMPTS.json'
        attempts=json.loads(ledger.read_text()) if ledger.exists() else [];attempts.append(record);ledger.write_text(json.dumps(attempts,indent=2))
        if observation:raise RuntimeError('foreign overlap in diagnostic; retain evidence and stop: '+tag)
        if not files_unchanged:raise RuntimeError('diagnostic instrumentation changed during the process: '+tag)
        if not status['qualified'] or status['peak_driver_MiB']>90*1024:
            if init_failure(TASK,record):tag=base+f'-infrastructure-retry{number+1}';continue
            raise RuntimeError(f'calibration gate failed: {tag}')
        return record
    raise RuntimeError('diagnostic retry limit reached: '+base)

def main():
    primary=resolved_existing(TASK)
    contract=TASK/'configs/calibration-protocol.json'
    protocol=dict(selection_bank='calibration',selection_targets=[.3,7,18],selection_steps=10,
                  selection_warmup=5,selection_policies=list(POLICIES),
                  selection_objective='geometric mean of the three max8rank mean complete-step times',
                  selection_tie_break=list(POLICIES),timer_targets=[7],timer_steps=4,
                  primary_routes_used_for_fitting=False,owner_fixed=True,
                  additional_branch_calibration=dict(bank='calibration-uniform',target=0,planner='moonep',
                      timer_off_steps=10,timer_on_steps=4,used_for_initial_policy_selection=False,
                      reason='source-confirmed no-replica home32/gradient.ready branch'))
    if contract.exists() and json.loads(contract.read_text())!=protocol:raise RuntimeError('calibration protocol changed')
    contract.write_text(json.dumps(protocol,indent=2))
    records=[]
    subprocess.run(['docker','exec','planner-executor-ep8-lab','bash',str(TASK/'tools/build_link_diagnostics.sh')],check=True)
    interface=[]
    # One full-length diagnostic pair at each endpoint, at the ORIGINAL
    # capacity. These are not extra main-matrix repeats or stable-gain claims.
    interface_contract=dict(targets=[.2,20],steps=25,warmup=5,repeats=1,receive_capacity=131072,
        interfaces=['legacy','unified'],purpose='isolate interface/replay changes from receive-capacity changes',
        original_policy_calls_per_process=dict(legacy=100,unified=50))
    (TASK/'configs/interface-regression-protocol.json').write_text(json.dumps(interface_contract,indent=2))
    for index,vio in enumerate(interface_contract['targets']):
        for api in (['legacy','unified'] if index==0 else ['unified','legacy']):
            tag=f'interface-original-cap-vio{vio:g}-{api}'
            result=checked(tag,['--planner','moonep','--recv-capacity','131072','--maxvio',str(vio),
                                '--steps','25','--interface',api])
            interface.append(dict(interface=api,maxvio=vio,**result));records.append(result)
            (TASK/'evidence/interface-regression.json').write_text(json.dumps(interface,indent=2))
            (TASK/'evidence/calibration-jobs.json').write_text(json.dumps(records,indent=2))
    links=checked('calibration-links-v1',[],mode='links');records.append(links)
    (TASK/'evidence/calibration-jobs.json').write_text(json.dumps(records,indent=2))
    selection=[]
    for vio in protocol['selection_targets']:
        for policy in POLICIES:
            tag=f'calibration-select-vio{vio:g}-{policy}'
            result=checked(tag,['--planner',policy,'--recv-capacity','163840','--maxvio',str(vio),
                '--steps','10','--skip-identity','--route-bank',str(TASK/'routes/calibration')])
            tag=result['tag']
            rank_rows=[[json.loads(x) for x in (TASK/'evidence'/tag/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
            means=statistics.mean(max(rank_rows[r][s]['milliseconds'] for r in range(8)) for s in range(5,10))
            selection.append(dict(planner=policy,maxvio=vio,mean_ms=means,**result));records.append(result)
            (TASK/'evidence/calibration-selection.json').write_text(json.dumps(selection,indent=2))
            (TASK/'evidence/calibration-jobs.json').write_text(json.dumps(records,indent=2))
    import math
    scores={p:math.exp(statistics.mean(math.log(r['mean_ms']) for r in selection if r['planner']==p)) for p in POLICIES}
    initial=min(POLICIES,key=lambda p:scores[p])
    (TASK/'configs/calibrated-initial-policy.json').write_text(json.dumps(dict(initial_policy=initial,scores=scores,protocol=protocol),indent=2))
    uniform=checked('calibration-select-vio0-moonep',['--planner','moonep','--recv-capacity','163840','--maxvio','0',
        '--steps','10','--skip-identity','--route-bank',str(TASK/'routes/calibration-uniform')])
    records.append(uniform);(TASK/'evidence/calibration-jobs.json').write_text(json.dumps(records,indent=2))
    if '--before-timers' in sys.argv:
        print('TIMER_OFF_CALIBRATION_COMPLETE',initial,scores,flush=True)
        return
    for policy in POLICIES:
        targets=[.3,7,18] if policy==initial else [7]
        if policy=='moonep':targets.append(0)
        for vio in targets:
            tag=f'calibration-timer-vio{vio:g}-{policy}'
            result=checked(tag,['--planner',policy,'--recv-capacity','163840','--maxvio',str(vio),
                '--steps','4','--skip-identity','--route-bank',str(TASK/('routes/calibration-uniform' if vio==0 else 'routes/calibration')),'--diagnostic-timer'],
                worker='unified-cap64-calibration-timer')
            tag=result['tag']
            # Hooks/event scopes are diagnostic instrumentation. Check their
            # numerical trajectory against the timer-OFF independent selection.
            errors=[]
            for rank in range(8):
                reference_tag=uniform['tag'] if vio==0 else next(r['tag'] for r in selection if r['planner']==policy and r['maxvio']==vio)
                plain=TASK/f'evidence/{reference_tag}/steps-rank{rank}.jsonl'
                if not plain.exists():raise RuntimeError('missing timer-OFF diagnostic reference')
                reference=[json.loads(x) for x in plain.read_text().splitlines()]
                measured=[json.loads(x) for x in (TASK/f'evidence/{tag}/steps-rank{rank}.jsonl').read_text().splitlines()]
                for row in measured:
                    base=reference[row['step']]
                    loss=abs(row['loss']-base['loss'])/max(abs(base['loss']),1e-12)
                    norm=abs(row['grad_norm']-base['grad_norm'])/max(abs(base['grad_norm']),1e-12)
                    errors.append(dict(rank=rank,step=row['step'],loss_relative=loss,norm_relative=norm))
                    if loss>.004 or norm>.02:raise RuntimeError('diagnostic hooks changed the numerical trajectory')
            (TASK/f'evidence/{tag}/instrumentation-consistency.json').write_text(json.dumps(dict(passed=True,rows=errors),indent=2))
            records.append(result);(TASK/'evidence/calibration-jobs.json').write_text(json.dumps(records,indent=2))
    print('INDEPENDENT_CALIBRATION_COMPLETE',initial,scores,flush=True)

if __name__=='__main__':
    try:main()
    except StagePause:print('CALIBRATION_ONE_JOB_COMPLETE',NEW_JOBS,flush=True)
