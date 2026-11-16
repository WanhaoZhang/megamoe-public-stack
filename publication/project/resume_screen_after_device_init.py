"""One bounded infrastructure retry, preserving the original failed attempt."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parent
manifest=root/'screening-results.json'
data=json.loads(manifest.read_text());jobs=data['jobs']
failed=[job for job in jobs if job['status']=='failed']
if len(failed)!=1 or any(job['status']=='running' for job in jobs):
    raise RuntimeError('Expected one stopped infrastructure failure, no active matrix writer')
job=failed[0];old_tag=job['tag'];folder=root/'evidence'/old_tag
log=(folder/'stdout.log').read_text()
if job.get('failed_attempts') or 'Inner_Error_Device_Subprocess_Startup_Timeout(E39007)' not in log:
    raise RuntimeError('Retry is limited to the first explicit device-startup timeout')
if list(folder.glob('steps-rank*.jsonl')) or job['receipt'].get('rank_completions'):
    raise RuntimeError('No automatic substitution of a run containing training observations')
if not job['receipt'].get('source_unchanged') or not job['receipt'].get('exclusive_devices'):
    raise RuntimeError('Source or device ownership mismatch requires independent investigation')
shutil.copyfile(manifest,root/'screening-before-device-retry.json')
job['preregistered_tag']=old_tag
job['failed_attempts']=[dict(tag=old_tag,classification='device_initialization_timeout',
    phase='torch.accelerator.set_device; before first training update',updates=0,
    error_codes=['507033','E39007'],root_cause_not_established=True,receipt=job['receipt'])]
job['tag']=old_tag+'-retry1';job['status']='planned';job.pop('receipt',None)
data['bounded_infrastructure_retry']=dict(maximum=1,original_tag=old_tag,
    code_and_recipe_unchanged=True,health_log='evidence/device-init-failure-health.log',
    host_resources_log='evidence/device-init-failure-host-resources.log',device_or_service_reset=False)

def save():
    temp=manifest.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    temp.replace(manifest)

save()
reference=json.loads((root/'evidence'/jobs[0]['tag']/'source-identity.json').read_text())
for job in jobs:
    if job['status']=='completed':continue
    if job['status']!='planned':raise RuntimeError('Unexpected unresolved job state')
    job['status']='running';save()
    args=[sys.executable,'-X','utf8',root/'frontier_remote.py','gate','--metadata-only',
        '--tag',job['tag'],'--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
        '--microbatches','4','--forward-window',str(job['window']),'--recv-capacity','262144',
        '--maxvio','10','--planner','native','--temporal-pattern',job['pattern'],
        '--temporal-policy',job['policy'],'--temporal-scheduler','ascendc','--grad-sync']
    code=subprocess.run([str(value) for value in args]).returncode
    folder=root/'evidence'/job['tag']
    status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    same_source=(json.loads((folder/'source-identity.json').read_text())==reference
                 if (folder/'source-identity.json').exists() else False)
    job.update(status='completed' if code==0 and status.get('qualified') and same_source and
        status.get('peak_driver_MiB',float('inf'))<=90*1024 else 'failed',
        receipt=status,source_identical_to_first_screen_run=same_source)
    save()
    if job['status']!='completed':
        raise RuntimeError('Bounded retry/continuation failed; remaining points stay unrun: '+job['tag'])
