"""Stage the exact measured MaxVio10 bank; generate no replacement routes."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=root/'implementation/replay-inputs/routes')
args=parser.parse_args();destination=args.out.resolve();destination.mkdir(parents=True,exist_ok=True)
source=root/'diagnostics/screening-maxvio-10.npy';raw=source.read_bytes();identity=hashlib.sha256(raw).hexdigest()
state=json.loads((root/'upstream-maxutil-results.json').read_text())
for job in state['pilot_jobs']:
    if not job['qualified']:continue
    frozen=json.loads((root/'evidence'/job['tag']/'source-identity.json').read_text())
    if frozen['routes/maxvio-10.npy']!=identity:raise RuntimeError('Local bank differs from actual measured source')
ids=np.load(source,allow_pickle=False)
if ids.shape!=(8,16384,8) or ids.dtype!=np.int32 or ids.min()<0 or ids.max()>=256:
    raise RuntimeError('Invalid measured bank shape/type/IDs')
if np.any(np.diff(np.sort(ids,axis=-1),axis=-1)==0):raise RuntimeError('Unexpected repeated expert within TopK')
counts=np.bincount(ids.ravel(),minlength=256)
metadata=(root/'stack/routes/maxvio-10.json').read_bytes()
if counts.tolist()!=json.loads(metadata)['counts']:raise RuntimeError('Route metadata count mismatch')
for filename,value in (('maxvio-10.npy',raw),('maxvio-10.json',metadata)):
    target=destination/filename
    if target.exists() and target.read_bytes()!=value:raise RuntimeError('Refuse to overwrite different replay data')
    if not target.exists():target.write_bytes(value)
record=dict(destination=str(destination),sha256=identity,shape=list(ids.shape),expert_maxvio=float(counts.max()/counts.mean()-1),
    token_inputs='Seed42 synthesis in frozen tools/train.py; input/label hashes per measured run',
    new_routes_generated=False,only_measured_maxvio10_staged=True,not_the108_run_formal_matrix=True)
(root/'implementation/replay-inputs.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
