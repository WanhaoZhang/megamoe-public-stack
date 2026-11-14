"""Assemble global rank wall samples only after all node receipts pass."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from summarize_training import compare_runs, load_run


def load_distributed(paths, steps, warmup):
    paths = list(map(Path, paths))
    parts = []
    for path in paths:
        launch = json.loads((path/'launch.json').read_text())
        t = launch['topology']
        assert t['nodes'] == len(paths) and t['local_world_size'] == 8
        assert type(t['node_rank']) is int and 0 <= t['node_rank'] < len(paths), 'invalid node rank'
        assert launch['world_size'] == len(paths)*8, 'world size does not match nodes'
        ranks = range(t['node_rank']*8, (t['node_rank']+1)*8)
        recorded = {int(p.stem[4:]) for p in (path/'steps').glob('rank*.jsonl')
                    if re.fullmatch(r'rank[0-9]+', p.stem)}
        assert recorded == set(ranks), 'missing, duplicate-node or extra global rank records'
        part = load_run(path, steps, warmup, ranks=ranks)
        assert part['node'], 'missing sealed host provenance'
        assert part['timing_eligible'], 'diagnostic capture cannot enter performance table'
        parts.append(part)
    parts.sort(key=lambda r:r['launch']['topology']['node_rank'])
    assert [r['launch']['topology']['node_rank'] for r in parts] == list(range(len(parts)))
    assert len({r['node']['machine_id_sha256'] for r in parts}) == len(parts), 'duplicate physical node'
    normalized = []
    for part in parts:
        launch = dict(part['launch'])
        topology = dict(launch.pop('topology'))
        topology.pop('node_rank')
        # Per-node source paths are deliberately allowed; binaries must match.
        hashes = {Path(p).name:value for p,value in launch.pop('worker_hashes', {}).items()}
        launch.pop('command')
        normalized.append((launch, topology, hashes))
    assert all(x == normalized[0] for x in normalized), 'nodes ran different configuration or worker SO'
    rows = [rows for part in parts for rows in part['rows']]
    world = len(parts)*8
    assert len(rows) == parts[0]['launch']['world_size'] == world
    maxima = [max(rank[s]['wall_ms'] for rank in rows) for s in range(warmup, steps)]
    return dict(path=[str(p) for p in paths], launch=parts[0]['launch'],
        rows=rows, initial=[v for part in parts for v in part['initial']],
        node=[part['node'] for part in parts],
        timing_eligible=all(part['timing_eligible'] for part in parts),
        rank_max_wall_ms=maxima, median_ms=statistics.median(maxima),
        p90_ms=sorted(maxima)[math.ceil(.9*len(maxima))-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', nargs='+', action='append', required=True, help='all node run directories for one method')
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--warmup', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    runs = [load_distributed(paths, a.steps, a.warmup) for paths in a.case]
    methods = {r['launch']['method']:r for r in runs}
    assert len(methods) == len(runs), 'one process order per report'
    rows = [dict(method=r['launch']['method'], median_ms=r['median_ms'], p90_ms=r['p90_ms'],
        world_size=r['launch']['world_size'], node=r['node'], path=r['path'],
        **compare_runs(r, methods.get('A'))) for r in runs]
    a.out.write_text(json.dumps(dict(steps=a.steps, warmup=a.warmup, rows=rows), indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
