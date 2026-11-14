"""Fail-closed compact comparison with ratios of paired aggregate times."""
import argparse
import json
from pathlib import Path
import statistics

from summarize_training import compare_runs, load_run


def peak_hbm(run_path):
    values = []
    for line in (Path(run_path)/'hbm-samples.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('rc') == 0 and len(row.get('hbm', {})) == 8:
            values.extend(x['used_mb'] for x in row['hbm'].values())
    if not values:
        raise ValueError('missing complete HBM observations')
    return max(values)


def aggregate(runs):
    methods = ('A', 'B', 'B-compact', 'D')
    for method in methods:
        forward, reverse = runs[method+'-f'], runs[method+'-r']
        for order in ('f', 'r'):
            for reference in ('A', 'D'):
                if compare_runs(runs[method+'-'+order], runs[reference+'-'+order])['speedup_vs_A'] is None:
                    raise ValueError('unpaired node/config/input/capture: '+method+'-'+order)
        keys = ('rho', 'activation', 'layers', 'experts', 'topk', 'sequence', 'ep',
                'expert_fsdp', 'world_size', 'steps', 'method', 'restore', 'publish_sync',
                'worker_variant', 'native_slots', 'recv_capacity_override', 'control_tape_device')
        signature = lambda r: (r['node'], {k:r['launch'].get(k) for k in keys},
            r['launch'].get('worker_hashes'),
            [x for x in r['launch'].get('command', []) if x.endswith('/runtime_bootstrap.py')])
        if signature(forward) != signature(reverse):
            raise ValueError('different implementation/node between orders: '+method)
    median = lambda method, key: statistics.median(runs[method+'-'+o][key] for o in ('f', 'r'))
    result = {}
    for method in methods:
        elapsed = median(method, 'median_ms')
        result[method] = dict(median_ms=elapsed, p90_ms=median(method, 'p90_ms'),
            speedup_vs_A=median('A', 'median_ms')/elapsed,
            change_vs_D_pct=100*(elapsed/median('D', 'median_ms')-1),
            peak_hbm_mb=max(peak_hbm(runs[method+'-'+o]['path']) for o in ('f', 'r')),
            same_node_A_f=True, same_node_A_r=True, initial_match_A_f=True,
            initial_match_A_r=True, batch_match_A_f=True, batch_match_A_r=True)
    result['B-compact']['change_vs_B_pct'] = 100*(result['B-compact']['median_ms']/result['B']['median_ms']-1)
    result['B-compact']['hbm_delta_vs_B_mb'] = result['B-compact']['peak_hbm_mb']-result['B']['peak_hbm_mb']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', nargs=9, action='append', required=True,
                   metavar=('RHO', 'AF', 'AR', 'DF', 'DR', 'BF', 'BR', 'CF', 'CR'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cases = []
    for rho, *paths in a.case:
        labels = ('A-f', 'A-r', 'D-f', 'D-r', 'B-f', 'B-r', 'B-compact-f', 'B-compact-r')
        runs = {label:load_run(Path(path), 40, 10) for label, path in zip(labels, paths)}
        assert all(r['launch']['rho'] == int(rho) for r in runs.values())
        cases.append(dict(rho=int(rho), orders={label:dict(path=r['path'], node=r['node'],
            median_ms=r['median_ms'], p90_ms=r['p90_ms']) for label,r in runs.items()}, combined=aggregate(runs)))
    result = dict(steps=40, warmup=10, aggregation='ratio of median process times in both orders', cases=cases)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([dict(rho=c['rho'], **c['combined']['B-compact']) for c in cases], indent=2))


if __name__ == '__main__':
    main()
