"""CPU-side benchmark contracts; importing this file never initializes an NPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ('original', 'moonep-reference', 'local-serial', 's9')
OUTPUTS = ('y', 'dx', 'dp', 'dw1', 'dw2')


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def align128(value):
    return (int(value) + 127) // 128 * 128


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_case(c):
    for key in ('ep', 'experts', 'slots', 'tokens', 'hidden', 'ffn', 'topk', 'capacity'):
        if isinstance(c[key], bool) or not isinstance(c[key], int):
            raise ValueError(f'{key} must be an integer')
    if c['ep'] != 8:
        raise ValueError('first runner supports true EP8 only (WORLD=EP=8)')
    if c['experts'] <= 0 or c['experts'] % 8:
        raise ValueError('experts must be divisible by EP8')
    home = c['experts'] // 8
    if c['slots'] != home or not 1 <= home <= 16:
        raise ValueError('four-way comparison requires slots=home<=16')
    if not (0 < c['tokens'] <= 16384 and 0 < c['topk'] <= c['experts']):
        raise ValueError('invalid tokens/TopK')
    if min(c['hidden'], c['ffn']) <= 0 or c['hidden'] % 256 or c['ffn'] % 256:
        raise ValueError('H/F must be positive multiples of 256')
    if c['capacity'] < align128(c['tokens'] * c['topk']) or c['capacity'] % 128:
        raise ValueError('capacity must be aligned128 and cover average rows')
    if c['capacity'] > 2147483392:
        raise ValueError('capacity exceeds int32 bound')
    if home * c['hidden'] * c['ffn'] * 2 % (2 * 1024**2):
        raise ValueError('original MoonEP BF16 weight blocks require 2MiB alignment')
    if c['route'] not in ('balanced', 'balanced_random', 'random', 'controlled', 'hot', 'rotate', 'rank_skew'):
        raise ValueError('unknown route')
    if c['planner'] not in ('native', 'moonep') or c['scope'] not in ('online', 'replay'):
        raise ValueError('unknown planner/scope')
    if not math.isfinite(c['rho']) or c['rho'] < 0:
        raise ValueError('rho must be finite and nonnegative')


def validate_receipts(records, backend=None):
    if len(records) != 8 or sorted(r['rank'] for r in records) != list(range(8)):
        raise ValueError('require exactly eight distinct rank receipts')
    first = records[0]
    for r in records:
        if not r.get('pass_') or r.get('ep') != 8 or r.get('world') != 8:
            raise ValueError('missing successful true EP8 receipt')
        for key in ('case', 'backend', 'mode', 'scope', 'code_identity',
                    'worker_variant', 'routing_variant', 'execution'):
            if r.get(key) != first.get(key):
                raise ValueError(f'rank disagreement: {key}')
        if backend is not None and r['backend'] != backend:
            raise ValueError('unexpected backend')
    return sorted(records, key=lambda r: r['rank'])


def check_pair(a, b, require_plan=False):
    a, b = validate_receipts(a), validate_receipts(b)
    for x, y in zip(a, b):
        for key in ('case', 'inputs', 'code_identity', 'scope'):
            if x[key] != y[key]:
                raise ValueError(f'incomparable {key} at rank {x["rank"]}')
        if require_plan and (not x['plan_digest'] or x['plan_digest'] != y['plan_digest']):
            raise ValueError('same-plan comparison has different consumed placement')


def summarize_samples(records):
    records = validate_receipts(records)
    samples = [r['samples_ms'] for r in records]
    if not samples[0] or any(len(s) != len(samples[0]) for s in samples):
        raise ValueError('empty or incomplete timing samples')
    if any(not math.isfinite(v) or v <= 0 for s in samples for v in s):
        raise ValueError('nonpositive/nonfinite timing sample')
    maximum = [max(row) for row in zip(*samples)]
    ordered = sorted(maximum)
    return dict(samples_ms=maximum, median_ms=statistics.median(maximum),
                p95_ms=ordered[math.ceil(.95*len(ordered))-1],
                min_ms=min(maximum), max_ms=max(maximum),
                stdev_ms=statistics.pstdev(maximum))


def audit_placement(c, logical, physical, copies, counts, active, expected_counts):
    import numpy as np
    logical,physical,copies,counts=map(np.asarray,(logical,physical,copies,counts))
    home=c['experts']//8
    if (logical.shape != (c['tokens'],c['topk']) or physical.shape != logical.shape or
        copies.shape != (8,c['slots']) or counts.shape != (c['experts'],)):
        raise ValueError('invalid placement tensor shapes')
    if not np.array_equal(counts,expected_counts) or counts.sum()!=8*c['tokens']*c['topk']:
        raise ValueError('planner lost or duplicated logical routes')
    if np.any(copies < -1) or np.any(copies >= c['experts']):
        raise ValueError('invalid copy owner')
    if active:
        slotmap=np.concatenate((np.arange(c['experts']).reshape(8,home),copies),axis=1).ravel()
        if np.any(physical<0) or np.any(physical>=len(slotmap)) or not np.array_equal(slotmap[physical],logical):
            raise ValueError('physical routes do not preserve expert identity')
    elif not np.array_equal(logical,physical):
        raise ValueError('owner-only physical routes changed')


def window_bytes(c, local_experts):
    align = lambda n, a: (n+a-1)//a*a
    # Same host formula as FSDPTurbo dispatcher_mega_moe._local_mem_bytes.
    size = 72*1024 + 4*c['ep']*align(c['ep']*local_experts, 128)
    size += 4*max(c['tokens']*c['topk'], c['capacity'])*c['hidden']
    size += align(2*c['tokens']*c['topk'], 512)
    return align(size, 2*1024**2)


def code_identity():
    # Hash executable benchmark and all production inputs, not branch labels.
    paths = []
    for directory in ('ascend_deepep', 'csrc', 'include', 'kernels', 'cmake'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py', '.h', '.hpp', '.cpp', '.cmake')]
    paths += list(Path(__file__).parent.glob('*.py'))
    paths += list(Path(__file__).parent.glob('*.sh'))
    paths += [ROOT/'CMakeLists.txt', ROOT/'setup.py']
    return fingerprint({str(p.relative_to(ROOT)).replace('\\', '/'): sha_file(p) for p in sorted(paths)})
