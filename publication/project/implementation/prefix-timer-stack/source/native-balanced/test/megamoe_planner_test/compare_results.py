#!/usr/bin/env python3
"""Offline acceptance and max-rank timing. Never initialize a device here."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import (ROOT, OUTPUTS, check_pair, load_file, sha_file,
                    summarize_samples, validate_receipts)


def read_run(folder):
    folder = Path(folder)
    gate = json.loads((folder/'acceptance.json').read_text())
    if not gate['eligible'] or gate['run_exit'] != 0 or not gate['post_gate']:
        raise ValueError(f'failed device/exit gate: {folder}')
    return validate_receipts([json.loads(p.read_text()) for p in folder.glob('rank*.json')])


def metric(a, b, forward):
    # Unchanged predicates from integration/wuhan/test_bridge_npu.py.
    golden = load_file('_bench_forward_golden', ROOT/'test/megamoe_bf16_test/mega_moe_bf16_golden.py')
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    if a.shape != b.shape: raise ValueError('output shape mismatch')
    d = np.abs(a-b)
    if forward:
        bad = golden.deviation_ratio(a, b, .01) >= .01
        passed = bad.mean() <= .05
    else:
        bad = (d > .05) & (d/(np.abs(b)+1e-6) > .15)
        passed = not bad.any() or (bad.mean() <= .05 and d.max() <= max(.1, float(np.abs(b).max())*.1))
    relative_l2=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12))
    # Additional guard, never a relaxation: small-valued tensors must not pass
    # simply because all their elements fall below an absolute threshold.
    passed = passed and relative_l2 <= .05
    return dict(pass_=bool(np.isfinite(a).all() and np.isfinite(b).all() and passed),
                bitwise=bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
                max_abs=float(d.max(initial=0)), bad_fraction=float(bad.mean()),
                relative_l2=relative_l2)


def precision(directories):
    import torch
    runs = {str(Path(d).resolve()):(Path(d), read_run(d)) for d in directories}
    if len(runs) != len(directories): raise ValueError('duplicate run directory')
    by_backend = {records[0]['backend']:(folder,records) for folder,records in runs.values()}
    if len(by_backend) != len(runs) or 'original' not in by_backend:
        raise ValueError('precision requires one original golden and distinct backends')
    base_folder, base = by_backend['original']
    details = {}
    for backend,(folder,records) in by_backend.items():
        check_pair(base, records)
        if any(r['mode'] != 'precision' for r in records): raise ValueError('not a precision run')
        for x,y in zip(base,records):
            if x['dependency_digest'] != y['dependency_digest']:
                raise ValueError('external dependency mismatch')
        rank_results = []
        for rank, (x,y) in enumerate(zip(base, records)):
            paths = [base_folder/f'rank{rank}.pt',folder/f'rank{rank}.pt']
            for path,r in zip(paths,(x,y)):
                if sha_file(path) != r['artifact_sha256']: raise ValueError('modified precision artifact')
            expected,actual = [torch.load(p,map_location='cpu',weights_only=True) for p in paths]
            metrics = {name:metric(actual[name].float().numpy(), expected[name].float().numpy(), name=='y')
                       for name in OUTPUTS}
            rank_results.append(dict(rank=rank, outputs=metrics))
        details[backend] = rank_results
    if 'local-serial' in by_backend and 's9' in by_backend:
        check_pair(by_backend['local-serial'][1], by_backend['s9'][1], require_plan=True)
    passed = all(m['pass_'] for ranks in details.values() for r in ranks for m in r['outputs'].values())
    return dict(pass_=passed,case=base[0]['case'],code_identity=base[0]['code_identity'],
        dependency_digest=base[0]['dependency_digest'], backends=list(by_backend),
        input_hashes={str(r['rank']):r['inputs'] for r in base},
        native_by_backend={b:{str(r['rank']):r['native'] for r in records}
                           for b,(_,records) in by_backend.items()},
        output_hashes_by_backend={b:{str(r['rank']):r['outputs'] for r in records}
                                 for b,(_,records) in by_backend.items()},
        execution_by_backend={b:records[0].get('execution') for b,(_,records) in by_backend.items()},
        details=details, tolerance_source='integration/wuhan/test_bridge_npu.py:metrics',
        additional_relative_l2_limit=.05,
        scope=base[0]['scope'])


def performance(directories):
    records = [read_run(d) for d in directories]
    first = records[0]
    for run in records:
        check_pair(first, run)
        if any(r['mode'] != 'performance' for r in run): raise ValueError('not a performance run')
        for a,b in zip(first,run):
            if a['dependency_digest'] != b['dependency_digest']: raise ValueError('dependency mismatch')
    # Per-run rows preserve AB/BA direction; do not merge before inspecting drift.
    rows = [dict(directory=str(d),backend=r[0]['backend'],**summarize_samples(r))
            for d,r in zip(directories,records)]
    paired=[]
    expected=['original','moonep-reference','local-serial','s9','s9','local-serial','moonep-reference','original']
    if [r['backend'] for r in rows]==expected:
        for direction,subset in (('forward',rows[:4]),('reverse',rows[4:])):
            by_backend={r['backend']:r for r in subset}
            for baseline in ('original','moonep-reference','local-serial'):
                serial=by_backend[baseline]['median_ms'];fused=by_backend['s9']['median_ms']
                paired.append(dict(direction=direction,baseline=baseline,
                                   baseline_over_s9=serial/fused,s9_change_percent=(fused/serial-1)*100))
    return dict(pass_=True,case=first[0]['case'],scope=first[0]['scope'],runs=rows,
                paired_comparisons=paired,
                note='Each sample is max over all eight EP ranks; compare paired run directions separately.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('precision','performance'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('runs',type=Path,nargs='+')
    a=p.parse_args()
    result=(precision if a.mode=='precision' else performance)(a.runs)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(pass_=result['pass_'],report=str(a.out))))
    if not result['pass_']:raise SystemExit(1)


if __name__=='__main__':main()
