"""Check replay serialization/fingerprints without loading an NPU extension."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import torch


def main():
    path = Path(__file__).resolve().parents[2]/'ascend_deepep/wuhan_replay_capture.py'
    package = ModuleType('capture_cpu')
    package.__path__ = [str(path.parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location('capture_cpu.wuhan_replay_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        pool = NS(dist=NS(get_rank=lambda:0, get_world_size=lambda:2),
                  shmem=NS(rank=lambda:0, world_size=lambda:2),
                  module=NS(_worker_native=NS(_name=__file__)), planner_name='moonep',
                  backward_restore='panel', home=1, slots=1, hidden=2, ffn=3,
                  tokens=4, topk=1, capacity=8, window_bytes=4096)
        x = torch.arange(8).reshape(4,2).to(torch.bfloat16)
        w1, w2 = torch.zeros(1,6,2), torch.zeros(1,2,3)
        ids = torch.zeros(4,1,dtype=torch.int32)
        placement = (ids, ids.clone(), torch.tensor([[-1],[0]],dtype=torch.int32),
                     torch.tensor([8,0],dtype=torch.int32))
        result = (x.clone(), torch.ones(4,1), w1.clone(), w2.clone())
        capture = module.WuhanReplayCapture(temporary)
        capture.save(pool, x, x+1, x-1, torch.ones(4,1), w1, w2,
                     [0,8,1,0,0,0,0,8], placement, result, 0.)
        file = Path(temporary)/'rank0.pt'
        manifest = json.loads(file.with_suffix('.json').read_text())
        assert module.file_sha(file) == manifest['file_sha256']
        payload = torch.load(file, weights_only=True)
        assert payload['config'] == manifest['config'] and payload['expected'] == manifest['expected']
        for name,value in payload['tensors'].items():
            assert module.tensor_sha(value) == manifest['tensors'][name]['sha256']
        assert payload['expected']['y'] == module.tensor_sha(x+1)
        assert payload['expected']['dw1'] == module.tensor_sha(w1)
        assert payload['config']['save_gmm1_output'] == 'bf16'
        assert payload['config']['resolved_save_gmm1_output'] == 'bf16'
        for requested, limit, resolved, mask in (
                ('none', 10., 'none', 'none'),
                ('mxfp8', 10., 'mxfp8', 'three_bitplanes_v1'),
                ('auto', 0., 'mxfp8', 'none'),
                ('auto', 10., 'bf16', 'none')):
            pool.save_gmm1_output = requested
            other = module.WuhanReplayCapture(Path(temporary)/f'{requested}-{limit}')
            other.save(pool, x, x+1, x-1, torch.ones(4,1), w1, w2,
                       [0,8,1,0,0,0,0,8], placement, result, limit)
            cfg = json.loads((other.directory/'rank0.json').read_text())['config']
            assert cfg['save_gmm1_output'] == requested
            assert cfg['resolved_save_gmm1_output'] == resolved
            assert cfg['saved_clip_mask_kind'] == mask
            assert cfg['normalized_swiglu_limit'] == (limit if limit else float.fromhex('0x1.fffffep+127'))
        try:
            module.WuhanReplayCapture(temporary).save(pool, x, x, x, torch.ones(4,1),
                w1, w2, [0,8,1,0,0,0,0,8], placement, result, 0.)
        except FileExistsError:
            pass
        else:
            raise AssertionError('overwrote an existing capture')
    print('REPLAY_CAPTURE_CPU_PASS')


if __name__ == '__main__':
    main()
