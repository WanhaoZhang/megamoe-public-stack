"""CPU evidence tests for rank-max timing and fail-closed run validation."""
import json
from pathlib import Path
import tempfile

from summarize_training import load_run, compare_runs


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'steps').mkdir()
        def put(path, value):
            (root / path).write_text(json.dumps(value))
        put('launch.json', dict(steps=4))
        put('acceptance.json', dict(eligible=True, run_exit=0, post_gate=True))
        put('pre.json', dict(pass_=True))
        put('post.json', dict(pass_=True))
        (root / 'run.exit').write_text('0\n')
        (root / 'run.log').write_text('training completed\n')
        for rank in range(8):
            rows = [dict(rank=rank, step=step, wall_ms=10 if step == rank + 2 else 1,
                         loss=1, lm_loss=1, mtp_loss=1, aux_loss=1, grad_norm=1,
                         boundary='before_fetch_to_optimizer_zero_grad_device_complete')
                    for step in range(1, 5)]
            path = root / 'steps' / f'rank{rank}.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        result = load_run(root, 4, 1)
        # Three different slow ranks, one at each measured step. Taking each
        # rank's median first would incorrectly return 1 instead of 10.
        assert result['rank_max_wall_ms'] == [10, 10, 10]
        assert result['median_ms'] == result['p90_ms'] == 10
        result['initial'] = [[dict(name='weight', sha256='same')]] * 8
        assert compare_runs(result, result)['speedup_vs_A'] is None, 'missing batch identity accepted'
        for rank in result['rows']:
            for row in rank:
                row['batch_fingerprints'] = [dict(path='input_ids', sha256='same')]
                row['reduced_losses'] = dict(lm_loss=1.)
        assert compare_runs(result, result)['speedup_vs_A'] is None, 'missing host identity accepted'
        result['node'] = dict(hostname='fixture-node', boot_id='fixture-boot')
        assert compare_runs(result, result)['speedup_vs_A'] == 1
        result['timing_eligible'] = False
        assert compare_runs(result, result)['speedup_vs_A'] is None, 'diagnostic capture claimed speedup'
        result['timing_eligible'] = True
        import copy
        changed = copy.deepcopy(result)
        changed['node']['hostname'] = 'another-node'
        assert compare_runs(changed, result)['speedup_vs_A'] is None, 'cross-node comparison accepted'
        changed = copy.deepcopy(result)
        changed['launch']['activation'] = 'uam'
        assert compare_runs(changed, result)['speedup_vs_A'] is None, 'different activation accepted'
        changed = copy.deepcopy(result)
        changed['rows'][7][3]['batch_fingerprints'][0]['sha256'] = 'different'
        checked = compare_runs(changed, result)
        assert checked['data_identical'] is False and checked['speedup_vs_A'] is None
        # Rank errors can cancel in the global mean; both metrics are needed.
        changed = copy.deepcopy(result)
        changed['rows'][0][0]['lm_loss'] = 1.1
        changed['rows'][1][0]['lm_loss'] = .9
        checked = compare_runs(changed, result)
        assert checked['max_local_lm_relative_error_vs_A'] > .09
        assert checked['max_mean_lm_relative_error_vs_A'] == 0
        assert checked['max_reduced_lm_relative_error_vs_A'] == 0
        path = root / 'steps' / 'rank7.jsonl'
        saved = path.read_text()
        path.write_text('\n'.join(saved.splitlines()[:-1]) + '\n')
        try:
            load_run(root, 4, 1)
        except AssertionError:
            pass
        else:
            raise AssertionError('missing last rank step was accepted')
        path.write_text(saved)
        put('post.json', dict(pass_=False))
        try:
            load_run(root, 4, 1)
        except AssertionError:
            pass
        else:
            raise AssertionError('failed post-gate was accepted')
    print('TRAINING_SUMMARY_CPU_PASS')


if __name__ == '__main__':
    main()
