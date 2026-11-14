"""Regression coverage for topology and aggregate-denominator mistakes."""
import copy
from subprocess import TimeoutExpired
from unittest.mock import patch

from distributed_config import LaunchTopology
from run_multinode import best_effort_abort, ssh_probe
from summarize_compact import aggregate
from subprocess_bounds import bound_command


def main():
    command = ['docker', 'exec', '-e', 'WORLD_SIZE=16', 'isolated-container', 'bash', 'run.sh']
    bounded, deadline = bound_command(command, 600)
    assert bounded == command[:5]+['timeout', '--signal=TERM', '--kill-after=20s', '600s']+command[5:]
    assert deadline == 645
    assert bound_command(['python3', 'run.py'], 30) == (['python3', 'run.py'], 30)
    node = dict(ssh='node')
    with patch('run_multinode.ssh', side_effect=TimeoutExpired(['ssh'], 15)):
        assert ssh_probe(node, 'cat ready') is None
        assert best_effort_abort(node, 'abort') is False
    for invalid in (['docker','exec','-d','container','bash'], ['docker','exec','-e']):
        try:
            bound_command(invalid, 30)
        except ValueError:
            pass
        else:
            raise AssertionError('unbounded or detached command accepted')
    assert LaunchTopology.from_env({}).world_size == 8
    t = LaunchTopology.from_env(dict(WUHAN_NNODES='2', WUHAN_NODE_RANK='1',
        WUHAN_EXPERT_FSDP='4', WUHAN_MASTER_ADDR='177.28.25.1'))
    assert t.world_size == 16 and t.node_rank == 1
    for kwargs in (dict(nodes=2), dict(nodes=2, expert_fsdp=4), dict(node_rank=1),
                   dict(expert_fsdp=3), dict(ep=16), dict(master_port=1)):
        try:
            LaunchTopology(**kwargs).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    runs = {}
    for method in ('A', 'B', 'B-compact', 'D'):
        for order, ms in (('f', 100.), ('r', 200.)):
            rows = [[dict(lm_loss=1., reduced_losses=dict(lm_loss=1.),
                          batch_fingerprints=['same'])] for _ in range(8)]
            runs[method+'-'+order] = dict(launch=dict(method=method, world_size=8),
                rows=rows, initial=[['same']]*8, node=dict(hostname='fixture'),
                median_ms=ms if method == 'A' else 50., p90_ms=ms, path='fixture')
    with patch('summarize_compact.peak_hbm', return_value=100):
        assert aggregate(runs)['B-compact']['speedup_vs_A'] == 3., 'must aggregate both A orders'
        for kind in ('node', 'capture', 'implementation', 'control-tape'):
            changed = copy.deepcopy(runs)
            r = changed['B-compact-r']
            if kind == 'node':
                r['node'] = dict(hostname='other')
            elif kind == 'capture':
                r['timing_eligible'] = False
            elif kind == 'implementation':
                r['launch']['restore'] = 'expert'
            else:
                r['launch']['control_tape_device'] = 'cpu'
            try:
                aggregate(changed)
            except ValueError:
                pass
            else:
                raise AssertionError(kind)
    print('MULTINODE_CONFIG_AND_PAIRING_CPU_PASS')


if __name__ == '__main__':
    main()
