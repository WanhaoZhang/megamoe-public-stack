set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 tools/restore_frozen_link.py
python3 tools/assemble_step_time_model.py
python3 - <<'PY'
import json,numpy as np
from pathlib import Path
import sys
sys.path.insert(0,'tools')
from step_time_reference import contract,propose,simulate,compiled_score,source_for
r=Path('.');model=json.loads((r/'calibration/step-time-provisional.json').read_text());v=model['device_vector']
print('MODEL', {k:v[k] for k in [18,19,27,29,38,39,44,45,46,47,48,49,50,51,52,53,54,60]})
for target in (.3,7,18):
 ids=np.load(r/f'routes/calibration/maxvio-{target:g}.npy',mmap_mode='r');print('ROUTE',target,ids.shape)
 # Route bank convention: [step,layer,rank,token,topk].
 src=source_for(ids,1,0)
 cp,quota,stats=propose(src,v);py=simulate(v,cp,quota);cc=compiled_score(src,v,cp,quota)
 print('ORACLE',target,stats.tolist(),py,cc,flush=True)
 assert py==cc,(py,cc)
PY
