"""Prepare independent deployment/build paths and retain the exact reference."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent;base=root.parent/'raw-control-20261116'
for folder in ('implementation','diagnostics','evidence'):(root/folder).mkdir(exist_ok=True)
old='step-lifetime-raw-control-20261116';new='step-lifetime-priority-cache-20261116'
baseline_variant='ep8-cap64-lifetime-raw-control-v2';candidate='ep8-cap64-lifetime-priority-cache-v1'
remote=(base/'remote.py').read_text().replace(old,new)
remote=remote.replace('/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack',
                      '/home/z00956592/megaflux/'+old+'/stack')
(root/'remote.py').write_text(remote,newline='\n')
for source,target in (
 ('stack/configs/lifetime-runtime-freeze.json','stack/configs/priority-baseline-freeze.json'),
 ('implementation/worker-build.json','implementation/baseline-worker-build.json')):
 (root/target).write_bytes((base/source).read_bytes())
script=(base/'implementation/build-raw-worker.sh').read_text().replace(old,new).replace(baseline_variant,candidate).replace('RAW_CONTROL_WORKER','PRIORITY_WORKER')
(root/'implementation/build-priority-worker.sh').write_text(script,newline='\n')
driver=(base/'build_worker.py').read_text().replace(old,new).replace(baseline_variant,candidate).replace('build-raw-worker.sh','build-priority-worker.sh')
(root/'build_worker.py').write_text(driver,newline='\n')
proof=root.parent/'causal-diagnostics-20261116/diagnostics/gradient-priority-key-oracle.json'
data=json.loads(proof.read_text())
(root/'diagnostics/order-oracle-reference.json').write_text(json.dumps(dict(
 path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),passed=data['passed'],
 plans=data['checked'],scope=data['scope']),indent=2)+'\n',newline='\n')
