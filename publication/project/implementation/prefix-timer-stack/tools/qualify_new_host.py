"""Qualify one new single-host experiment, then start its complete fresh matrix."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from controller import TASK,source_identity,CONTAINER

deadline=time.time()+2400
while not (TASK/'evidence/NEW-HOST-IMAGE-VERIFY.json').exists():
    if time.time()>deadline:raise RuntimeError('verified private image was not loaded; no device work started')
    time.sleep(5)
workspace=json.loads((TASK/'incoming/WORKSPACE-MIGRATION.json').read_text())
assert source_identity()==workspace['source_identity'],'migrated measured sources differ from the original'
for name,digest in workspace['baseline_binaries'].items():
    assert hashlib.sha256((TASK/name).read_bytes()).hexdigest()==digest,name
provenance=TASK/'evidence/provenance';provenance.mkdir(exist_ok=True)
old_selection=TASK/'configs/selected-capacity.json'
if old_selection.exists():old_selection.rename(provenance/'old-host-selected-capacity.json')
subprocess.run(['bash',str(TASK/'tools/create_replay_container.sh'),CONTAINER],check=True)
identity=dict(time_utc=datetime.now(timezone.utc).isoformat(),host='pod5-8-root',
    hostname=subprocess.run(['hostname'],capture_output=True,text=True,check=True).stdout.strip(),
    device_info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout,
    same_image_id=json.loads((TASK/'evidence/NEW-HOST-IMAGE-VERIFY.json').read_text())['image_id'],
    original_node='pod6-1-root',original_driver='25.1.rc3.2',actual_driver='25.1.rc3.1',
    full_new_matrix_required=True,no_cross_host_timing_aggregation=True,
    source_identity_equal=True,binary_identity_equal=True)
(TASK/'evidence/HARDWARE-IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
manifest=[]
for argv,label in [
    (['final_gates.py','--revision','v4'],'final_interface_32_64_scaling_clip'),
    (['qualification.py','--phase','accuracy','--revision','new-host-v1'],'all_five_independent_gradients'),
    (['qualification.py','--phase','pool','--revision','new-host-v1'],'all_five_two_layer_replay'),
    (['qualify_capacity.py'],'all_five_full_shape_three_Adam_updates')]:
    started=time.time();result=subprocess.run([sys.executable,str(TASK/'tools'/argv[0]),*argv[1:]])
    manifest.append(dict(stage=label,command=argv,exit=result.returncode,start=started,finish=time.time()))
    (TASK/'evidence/NEW-HOST-QUALIFICATION-STAGES.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if result.returncode:raise RuntimeError('new host qualification failed: '+label)
print('NEW_HOST_QUALIFIED_STARTING_FULL_FRESH_MATRIX',flush=True)
subprocess.run([sys.executable,str(TASK/'tools/start_primary.py')],check=True)
