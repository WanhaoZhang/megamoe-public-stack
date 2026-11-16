"""Explicit same-node publication ablation; never substitute missing A/D controls."""
import argparse
import json
from pathlib import Path
import statistics

from summarize_training import compare_runs, load_run


def peak(run):
    path = Path(run['path'])/'hbm-samples.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    good = [r for r in records if r['rc'] == 0]
    assert good and all(len(r['hbm']) == 8 for r in good), 'missing whole-device HBM samples'
    return max(v['used_mb'] for r in good for v in r['hbm'].values())


def same_config(a, b):
    assert Path(a['path']).parent.resolve() == Path(b['path']).parent.resolve(), 'cross-node denominator'
    keys = ('rho', 'activation', 'layers', 'experts', 'topk', 'sequence', 'ep', 'expert_fsdp', 'steps')
    assert {k:a['launch'][k] for k in keys} == {k:b['launch'][k] for k in keys}, 'different training config'


def summarize(project, rho):
    rows = []
    for order in ('f', 'r'):
        a = load_run(project/f'evidence/fast-rho{rho}-A-{order}-a1', 40, 10)
        d = load_run(project/f'evidence/fast-rho{rho}-D-{order}-a1', 40, 10)
        device = load_run(project/f'evidence/opt-rho{rho}-C-device-{order}-a1', 40, 10)
        stream = load_run(project/f'evidence/opt-rho{rho}-C-stream-{order}-a1', 40, 10)
        for run in (a, d, device, stream):
            same_config(a, run)
            assert compare_runs(run, a)['speedup_vs_A'] is not None, 'missing matching fingerprints/timing'
        for mode, run in (('device', device), ('stream', stream)):
            assert run['launch']['publish_sync'] == mode
            pair = compare_runs(run, a)
            rows.append(dict(order=order, mode=mode, median_ms=run['median_ms'], p90_ms=run['p90_ms'],
                A_ms=a['median_ms'], D_ms=d['median_ms'], speedup_vs_A=pair['speedup_vs_A'],
                change_pct_vs_D=100*(run['median_ms']/d['median_ms']-1),
                change_pct_vs_device=100*(run['median_ms']/device['median_ms']-1),
                peak_hbm_mb=peak(run), D_peak_hbm_mb=peak(d),
                max_local_lm_relative_error_vs_A=pair['max_local_lm_relative_error_vs_A'],
                max_reduced_lm_relative_error_vs_A=pair['max_reduced_lm_relative_error_vs_A'],
                max_reduced_lm_relative_error_vs_device=compare_runs(run, device)['max_reduced_lm_relative_error_vs_A'],
                path=run['path']))
    combined = []
    for mode in ('device', 'stream'):
        group = [r for r in rows if r['mode'] == mode]
        med = lambda key: statistics.median(r[key] for r in group)
        combined.append(dict(mode=mode, median_ms=med('median_ms'),
            A_ms=med('A_ms'), D_ms=med('D_ms'), speedup_vs_A=med('A_ms')/med('median_ms'),
            change_pct_vs_D=100*(med('median_ms')/med('D_ms')-1),
            max_peak_hbm_mb=max(r['peak_hbm_mb'] for r in group),
            max_D_peak_hbm_mb=max(r['D_peak_hbm_mb'] for r in group),
            each_order_change_pct_vs_device=[r['change_pct_vs_device'] for r in group]))
    return dict(rho=rho, steps=40, warmup=10, A_D_reused_from_frozen_same_node_matrix=True,
                aggregation='median of two process medians; ratio of aggregated times',
                process_rows=rows, combined=combined)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--rho', type=int, choices=(0, 3, 6), required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.project, args.rho)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['combined'], indent=2))
