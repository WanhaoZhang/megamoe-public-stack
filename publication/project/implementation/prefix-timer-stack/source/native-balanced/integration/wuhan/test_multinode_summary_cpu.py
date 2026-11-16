"""Exercise global-step pairing and reject malformed multi-node receipts."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from summarize_multinode import load_distributed


def main():
    with tempfile.TemporaryDirectory() as directory:
        paths, parts = [], {}
        for node in range(2):
            path = Path(directory)/str(node)
            (path/'steps').mkdir(parents=True)
            for rank in range(node*8, node*8+8):
                (path/'steps'/f'rank{rank}.jsonl').touch()
            launch = dict(topology=dict(nodes=2, node_rank=node, local_world_size=8,
                ep=4, expert_fsdp=4), world_size=16, method='B',
                command=['per-node-path'], worker_hashes={f'/node{node}/kernel.so':'same'})
            (path/'launch.json').write_text(json.dumps(launch))
            times = [1., 100., 10.] if node == 0 else [1., 10., 100.]
            parts[path] = dict(launch=launch, node=dict(machine_id_sha256=str(node)),
                timing_eligible=True, rows=[[dict(wall_ms=t) for t in times] for _ in range(8)],
                initial=[['same']]*8)
            paths.append(path)
        def load(path, steps, warmup, ranks):
            assert list(ranks) == list(range(int(path.name)*8, int(path.name)*8+8))
            return copy.deepcopy(parts[path])
        with patch('summarize_multinode.load_run', side_effect=load):
            combined = load_distributed(paths[::-1], 3, 1)
            assert combined['rank_max_wall_ms'] == [100., 100.]
            assert combined['median_ms'] == 100.  # Node medians would both be 55.
            for kind in ('missing-rank', 'extra-rank', 'duplicate-node', 'same-host', 'capture', 'hash', 'config'):
                original = copy.deepcopy(parts)
                launches = [(p/'launch.json').read_text() for p in paths]
                removed = paths[1]/'steps/rank15.jsonl'
                extra = paths[1]/'steps/rank0.jsonl'
                if kind == 'missing-rank':
                    removed.unlink()
                elif kind == 'extra-rank':
                    extra.touch()
                elif kind == 'duplicate-node':
                    changed = json.loads(launches[1])
                    changed['topology']['node_rank'] = 0
                    (paths[1]/'launch.json').write_text(json.dumps(changed))
                elif kind == 'same-host':
                    parts[paths[1]]['node'] = parts[paths[0]]['node']
                elif kind == 'capture':
                    parts[paths[1]]['timing_eligible'] = False
                elif kind == 'hash':
                    parts[paths[1]]['launch']['worker_hashes'] = {'/node1/kernel.so':'different'}
                else:
                    parts[paths[1]]['launch']['topology']['expert_fsdp'] = 2
                try:
                    load_distributed(paths, 3, 1)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(kind)
                parts.clear()
                parts.update(original)
                removed.touch(exist_ok=True)
                extra.unlink(missing_ok=True)
                for path, text in zip(paths, launches):
                    (path/'launch.json').write_text(text)
    print('MULTINODE_GLOBAL_MAX_AND_REJECTION_CPU_PASS')


if __name__ == '__main__':
    main()
