"""CPU loader for generated fixed-case inputs, independent of Ascend imports."""
import hashlib
import json
from pathlib import Path


def tensor_sha(value):
    import torch
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_scenario_inputs(directory, rank, world):
    root = Path(__file__).resolve().parent/'scenarios'/'python-olmoe-high-skew'
    case = json.loads((root/'case.json').read_text(encoding='utf-8'))
    frozen = json.loads((root/'expected-inputs.json').read_text(encoding='utf-8'))
    if world != case['world'] or rank not in range(world):
        raise ValueError('fixed Python scenario requires world8/global ranks0..7')
    path = Path(directory)/f'rank{rank}-inputs.pt'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta['kind'] != case['kind'] or meta['name'] != case['name']:
        raise ValueError('scenario identity mismatch')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != meta['file_sha256']:
            raise ValueError('scenario input file SHA mismatch')
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    config, values = payload['config'], payload['tensors']
    from timer_capture_config import window_bytes
    required = dict(rank=rank, ep_rank=rank%case['ep'], world_size=world,
                    ep=case['ep'], tokens=case['M'], hidden=case['H'], ffn=case['F'],
                    home=case['home'], slots=case['slots'], topk=case['TopK'],
                    capacity=case['capacity'], limit=case['limit'])
    required['window_bytes'] = window_bytes(case['M'], case['H'], case['TopK'],
                                           case['ep'], case['home']+case['slots'], case['capacity'])
    if config != required or config != meta['config']:
        raise ValueError('scenario config differs from frozen shape/rank/EP/window contract')
    actual = {name: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value))
              for name, value in values.items()}
    if actual != meta['tensors'] or actual != frozen[str(rank)]['tensors']:
        raise ValueError('scenario tensors differ from frozen historical fingerprints')
    source = dict(kind=case['kind'], name=case['name'], file_sha256=meta['file_sha256'],
                  source_kappa=case['source_kappa'], target_kappa=case['target_kappa'],
                  placement='online MoonEP; generated inputs do not freeze the planner',
                  historical_outputs_checked=False)
    return config, values, source
