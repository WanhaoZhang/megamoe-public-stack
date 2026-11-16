"""Persist CPU-only Afore-style reference results with exact source identities."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
files=['prototypes/afore_reference.py','prototypes/test_afore_reference.py',
       'stack/source/native-balanced/ascend_deepep/planner_contract.py']
identity={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
run=subprocess.run([sys.executable,'-X','utf8',str(root/files[1])],capture_output=True,text=True,encoding='utf-8')
folder=root/'diagnostics';folder.mkdir(exist_ok=True)
(folder/'afore-reference-tests.log').write_text(run.stdout+run.stderr,encoding='utf-8')
receipt=dict(created=datetime.now(timezone.utc).isoformat(),exit=run.returncode,
    suite='10 CPU semantic tests including120 seeded compiled-plan cases',seed=4913,
    source_sha256=identity,actual_weight_migration=False,npu_qualified=False,
    online_qualified=False,calibrated_costs=False,training_gradient_qualified=False,
    source='https://arxiv.org/html/2610.03203v1',paper_scope='AFD serving; this is a fixed-owner training-oriented reference')
(folder/'afore-reference-tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(run.stdout+run.stderr,end='')
raise SystemExit(run.returncode)
