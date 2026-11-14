"""Special calibration-only balanced bank for the no-replica kernel branch."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'routes/calibration-uniform';out.mkdir(exist_ok=True)
base=(np.arange(16384*8,dtype=np.int32).reshape(16384,8)%256)
ids=np.stack([np.roll(base,rank*17,axis=0) for rank in range(8)])
counts=np.bincount(ids.reshape(-1),minlength=256)
assert np.all(counts==4096) and not np.any(np.sort(ids,axis=-1)[...,1:]==np.sort(ids,axis=-1)[...,:-1])
path=out/'maxvio-0.npy';np.save(path,ids,allow_pickle=False)
meta=dict(target_maxvio=0.,realized_maxvio=0.,counts=counts.tolist(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    purpose='calibration-only no-replica/home32 branch; not a main or heldout benchmark point',
    generation='deterministic round-robin unique Top8; each source has512 assignments per expert',not_official_random_bank=True)
path.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k!='counts'}))
