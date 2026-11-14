"""Reject unmatched long runs and preserve rank-max/segment drift evidence."""
import copy
from unittest.mock import patch

from run_long_queue import training_command
from summarize_long import analyze


def main():
    command = training_command('/project/frozen', 'isolated', dict(name='long-rho6-A-a1', method='A', rho=6))
    assert command[command.index('isolated')+1:command.index('bash')] == [
        'timeout', '--signal=TERM', '--kill-after=20s', '2400s']
    assert command[-5:] == ['A', '6', 'recompute', '200', '8']
    assert 'WUHAN_WORKER_VARIANT=restore-panel-ready' in command
    release = training_command('/project/frozen', 'isolated',
        dict(name='long-rho6-B-a1', method='B', rho=6), worker_variant='release-panel-ready')
    assert 'WUHAN_WORKER_VARIANT=release-panel-ready' in release
    assert release[-5:] == ['B', '6', 'recompute', '200', '8']
    for invalid in ('', '../escape', 'with space', 'safe\nunsafe'):
        try:
            training_command('/project/frozen', 'isolated',
                dict(name='valid', method='B', rho=6), worker_variant=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe worker variant accepted')
    for invalid in ('../escape', '', 'with space'):
        try:
            training_command('/project/frozen', 'isolated', dict(name=invalid, method='A', rho=6))
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe name accepted')
    rows = [[dict(step=s+1, wall_ms=10.+(s//50)*10.+rank, lm_loss=1., grad_norm=2.,
                  batch_fingerprints=['same'], allocated_bytes=1024**3, reserved_bytes=2*1024**3,
                  peak_allocated_bytes=3*1024**3, peak_reserved_bytes=4*1024**3,
                  reduced_losses=dict(lm_loss=1.)) for s in range(200)] for rank in range(8)]
    base = dict(launch=dict(method='A', steps=200, world_size=8), rows=rows,
                node=dict(hostname='same'), initial=[['same']]*8, median_ms=32., p90_ms=47., path='fixture')
    candidate = copy.deepcopy(base)
    candidate['launch']['method'] = 'B'
    with patch('summarize_long.peak_hbm', return_value=100):
        result, output = analyze([base, candidate])
        assert [s['median_ms'] for s in result['rows'][0]['segments']] == [17.,27.,37.,47.]
        assert len(output) == 400 and output[0]['rank_max_wall_ms'] == 17.
        assert output[0]['allocated_gib_max'] == 1. and output[0]['peak_allocated_gib_max'] == 3.
        candidate['rows'][7][199]['batch_fingerprints'] = ['changed']
        try:
            analyze([base, candidate])
        except ValueError:
            pass
        else:
            raise AssertionError('unpaired late-step batch accepted')
    print('LONG_RUN_PAIRING_SEGMENTS_AND_DEADLINE_CPU_PASS')


if __name__ == '__main__':
    main()
