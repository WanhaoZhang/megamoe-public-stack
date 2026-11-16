"""Opt-in, one-layer diagnostic capture. Never enabled in formal training."""
import hashlib
import json
from pathlib import Path

import torch
from .saved_gate_up_mode import saved_gate_up_metadata


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class WuhanReplayCapture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.written = False

    def save(self, pool, x, y, dy, prob, w1, w2, status, placement, result, limit):
        if self.written:
            return
        rank = pool.dist.get_rank()
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory/f'rank{rank}.pt'
        manifest = target.with_suffix('.json')
        if target.exists() or manifest.exists():
            raise FileExistsError(f'capture output already exists: {target}')
        values = dict(x=x, dy=dy, prob=prob, w1=w1, w2=w2,
                      **dict(zip(('logical_ids', 'physical_ids', 'copies', 'logical_counts'), placement)))
        expected = {key: tensor_sha(value) for key,value in
                    dict(y=y, **dict(zip(('dx', 'dp', 'dw1', 'dw2'), result))).items()}
        cpu = {name: value.detach().contiguous().cpu().clone() for name, value in values.items()}
        worker = Path(pool.module._worker_native._name)
        mode = saved_gate_up_metadata(getattr(pool, 'save_gmm1_output', 'bf16'), limit)
        config = dict(rank=rank, ep_rank=pool.shmem.rank(), ep=pool.shmem.world_size(),
                      world_size=pool.dist.get_world_size(), planner=pool.planner_name,
                      restore=pool.backward_restore, home=pool.home, slots=pool.slots,
                      hidden=pool.hidden, ffn=pool.ffn, tokens=pool.tokens, topk=pool.topk,
                      capacity=pool.capacity, window_bytes=pool.window_bytes,
                      status=list(status), limit=limit, scope='first_expert_backward_before_efsdp_reduce_scatter',
                      save_gmm1_output=mode['requested'],
                      resolved_save_gmm1_output=mode['resolved'],
                      normalized_swiglu_limit=mode['normalized_limit'],
                      saved_clip_mask_kind=mode['clip_mask_kind'],
                      diagnostic_only=True, worker=str(worker),
                      worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest())
        temporary = target.with_suffix('.partial')
        torch.save(dict(config=config, tensors=cpu, expected=expected), temporary)
        temporary.rename(target)
        manifest.write_text(json.dumps(dict(config=config,
            file_sha256=file_sha(target), expected=expected,
            tensors={name: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_sha(value))
                     for name, value in cpu.items()}), indent=2))
        self.written = True
        print('WUHAN_REPLAY_CAPTURED', rank, str(target), flush=True)
