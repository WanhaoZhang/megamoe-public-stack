"""Explicit diagnostic route replacement for a full-model run, not natural routing."""
from pathlib import Path
import hashlib,json,os
import numpy as np

_CACHE={}


def fixed_routes(index, weights, experts):
    import torch
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8 or experts!=8 or tuple(index.shape)!=(8192,2):
        raise ValueError(f'frozen Python case requires world8/E8/M8192/Top2: {world,experts,tuple(index.shape)}')
    root=Path(os.environ['WUHAN_FIXED_CASE_ROUTES'])
    key=(rank,str(index.device),index.dtype,weights.dtype)
    if key not in _CACHE:
        case=Path('/project/source/ascend_deepep_hkq/integration/wuhan/scenarios/python-olmoe-high-skew/case.json')
        meta=json.loads(case.read_text());reference=meta['exact_route_files'][rank%4]
        path=root/f'rank{rank%4}-ids.npy'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==reference['sha256']
        ids=np.load(path,allow_pickle=False)
        assert ids.dtype==np.int32 and ids.shape==(8192,2)
        assert hashlib.sha256(ids.tobytes()).hexdigest()==reference['raw_tensor_sha256']
        assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
        with torch.no_grad():
            fixed=torch.from_numpy(ids.copy()).to(device=index.device,dtype=index.dtype)
            prob=torch.full_like(weights,.5)
        # First-call warmup-only check, outside the selected profiling step.
        actual=fixed.detach().to(device='cpu',dtype=torch.int32).contiguous().numpy()
        assert hashlib.sha256(actual.tobytes()).hexdigest()==reference['raw_tensor_sha256']
        _CACHE[key]=(fixed,prob)
        out=Path(os.environ['WUHAN_STEP_RECORD_DIR']).parent/'route-inputs'
        out.mkdir(parents=True,exist_ok=True)
        record=dict(rank=rank,ep_rank=rank%4,kind=meta['kind'],shape=list(ids.shape),
            source_kappa=meta['source_kappa'],target_kappa=meta['target_kappa'],
            local_counts=np.bincount(ids.ravel(),minlength=8).tolist(),ep_group_counts=meta['counts'],
            logical_ids_sha256=reference['raw_tensor_sha256'],probability=.5,
            note='real gate still executes; debug helper replaces its output with frozen diagnostic IDs/prob')
        target=out/f'rank{rank}-{str(index.dtype).split(".")[-1]}.json'
        target.write_text(json.dumps(record,indent=2))
    return _CACHE[key]
