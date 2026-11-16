"""Complete RouteProvider sidecars without modifying frozen token arrays."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def route_sidecar(path,original):
    ids=np.load(path,allow_pickle=False);base=np.load(original,allow_pickle=False)
    counts=np.bincount(ids.ravel(),minlength=256)
    if not np.array_equal(counts,np.bincount(base.ravel(),minlength=256)):
        raise ValueError('Global expert counts changed')
    old=json.loads(original.with_suffix('.json').read_text())
    if not np.array_equal(counts,old['counts']):raise ValueError('Original count metadata disagrees with actual routes')
    metadata=dict(shape=list(ids.shape),dtype=str(ids.dtype),counts=counts.tolist(),
        realized=float(counts.max()/counts.mean()-1),target=old['target'],
        synthetic=True,scope='Complete-row source regrouping only; not natural training routing',
        npy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_npy_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        original_json_sha256=hashlib.sha256(original.with_suffix('.json').read_bytes()).hexdigest())
    text=json.dumps(metadata,indent=2)+'\n';output=path.with_suffix('.json')
    if output.exists() and output.read_text()!=text:raise RuntimeError('Existing sidecar differs')
    output.write_text(text,newline='\n');return metadata

def main():
    manifest=json.loads((ROOT/'artifacts/source-heterogeneity-bank-v1/manifest.json').read_text())
    rows=[]
    for row in manifest['records']:
        path=Path(row['path']);original=Path(row['source_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Frozen route identity drift')
        data=route_sidecar(path,original);rows.append(dict(path=str(path.with_suffix('.json')),npy_sha256=data['npy_sha256'],
            json_sha256=hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest()))
    (ROOT/'evidence/source-regroup-sidecars-v2.json').write_text(json.dumps(dict(passed=True,banks=len(rows),
        token_arrays_unchanged=True,records=rows,reason='RouteProvider requires counts sidecar even when only its expert count is read'),indent=2))
    print('ROUTE_SIDECARS_COMPLETED_WITHOUT_TOKEN_CHANGES',len(rows))

if __name__=='__main__':main()
