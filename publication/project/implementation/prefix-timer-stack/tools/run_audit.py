"""Run evidence collected outside measured training intervals."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import platform
from pathlib import Path
import torch
from torch.distributed.tensor import DTensor

def local(value):
    return value.to_local() if isinstance(value, DTensor) else value

def optimizer_list(optimizer):
    return list(optimizer.optimizers_dict.values()) if hasattr(optimizer, 'optimizers_dict') else [optimizer]

def parameter_samples(model):
    return {name: local(value).detach().reshape(-1)[:128].cpu().clone()
            for name, value in model.named_parameters() if value.requires_grad}

def check_persistent(model, optimizer, gradients=False):
    result = dict(parameter_bytes=0, gradient_bytes=0, adam_bytes=0, parameters=0,
                  gradients=0, states=0, optimizer_types=[])
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        tensor = local(value)
        if tensor.dtype != torch.float32:
            raise RuntimeError(f'Persistent parameter {name}: {tensor.dtype}')
        result['parameter_bytes'] += tensor.numel() * tensor.element_size()
        result['parameters'] += 1
        if gradients and value.grad is not None:
            gradient = local(value.grad)
            if gradient.dtype != torch.float32:
                raise RuntimeError(f'Persistent gradient {name}: {gradient.dtype}')
            result['gradient_bytes'] += gradient.numel() * gradient.element_size()
            result['gradients'] += 1
    for opt in optimizer_list(optimizer):
        result['optimizer_types'].append(type(opt).__name__)
        result.setdefault('hyperparameters', []).append([
            {key: group.get(key) for key in ('lr','betas','eps','weight_decay')}
            for group in opt.param_groups])
        if type(opt).__name__ != 'FunctionalFusedAdamW':
            raise RuntimeError('Unexpected optimizer implementation')
        for state in opt.state.values():
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    tensor = local(state[key])
                    if tensor.dtype != torch.float32:
                        raise RuntimeError(f'Persistent Adam {key}: {tensor.dtype}')
                    result['adam_bytes'] += tensor.numel() * tensor.element_size()
                    result['states'] += 1
    return result

def loaded_libraries(out, rank):
    paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if '.so' in line and line.split()[-1].startswith('/')})
    shmem = [path for path in paths if Path(path).name == 'libshmem.so']
    if len(shmem) != 1:
        raise RuntimeError(f'Expected exactly one SHMEM runtime: {shmem}')
    records = {}
    for name in paths:
        if '/source/native-balanced/' in name or Path(name).name.startswith('libstdc++.so') or Path(name).name in ('libshmem.so', 'libprofapi.so', 'libprofimpl.so'):
            file = Path(name)
            records[name] = dict(size=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (out / f'libraries-rank{rank}.json').write_text(json.dumps(dict(shmem=shmem, libraries=records, python=platform.python_version(), torch=torch.__version__,
        default_dtype=str(torch.get_default_dtype()), sitecustomize=getattr(sys.modules.get('sitecustomize'), '__file__', None),
        ascend_home=os.environ.get('ASCEND_HOME_PATH'), sys_path=sys.path), indent=2))
