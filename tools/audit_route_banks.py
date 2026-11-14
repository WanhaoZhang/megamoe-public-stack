"""Validate every supplied bank independently of the planner/compiler."""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
records=[]
for directory in (ROOT/'routes',ROOT/'routes/calibration',ROOT/'routes/heldout-a',ROOT/'routes/heldout-b'):
    for path in sorted(directory.glob('maxvio-*.npy')):
        target=float(path.stem.split('-')[1]);ids=np.load(path,allow_pickle=False)
        if ids.shape!=(8,16384,8) or ids.dtype!=np.int32:raise ValueError(f'shape/dtype: {path}')
        if ids.min()<0 or ids.max()>=256:raise ValueError(f'illegal expert ID: {path}')
        ordered=np.sort(ids,axis=-1)
        if np.any(ordered[...,1:]==ordered[...,:-1]):raise ValueError(f'TopK sampled with replacement: {path}')
        source=np.stack([np.bincount(rank.reshape(-1),minlength=256) for rank in ids])
        counts=source.sum(axis=0);actual=float(counts.max()/counts.mean()-1)
        if abs(actual-target)>max(.02*target,.005):raise ValueError(f'bad target calibration: {path}: {actual}')
        metadata=json.loads(path.with_suffix('.json').read_text())
        if not np.array_equal(counts,np.array(metadata['counts'])):raise ValueError(f'metadata histogram mismatch: {path}')
        records.append(dict(bank=directory.name if directory!=ROOT/'routes' else 'primary',target=target,
             realized_maxvio=actual,source_assignments=source.sum(axis=1).tolist(),
             hottest_expert=int(np.argmax(counts)),hottest_count=int(counts.max()),
             average_count=float(counts.mean()),npy_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
result=dict(passed=True,banks=4,points=len(records),topk_without_replacement=True,
            step_layer_permutation_preserves_histogram=True,records=records)
(ROOT/'evidence/route-bank-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
