"""CPU-only checks for actual fetched-data identity and recorder boundaries."""
import argparse
import importlib.util

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('paired_record', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recorder = module.StepRecorder.__new__(module.StepRecorder)
    recorder.started = None
    tokens = torch.arange(12, dtype=torch.int64).reshape(3, 4).T
    labels = tokens.clone()
    batches = [dict(input_ids=tokens, labels=labels, optional=None)]
    held = recorder.hold_batch(batches)
    batches[0]['labels'] = None
    state = torch.get_rng_state().clone()
    rows = recorder.fingerprint_batch(held)
    assert torch.equal(state, torch.get_rng_state()), 'data audit consumed RNG'
    by_name = {row['path']: row for row in rows}
    assert by_name['batches/0/labels']['sha256'] == by_name['batches/0/input_ids']['sha256']
    assert by_name['batches/0/labels']['shape'] == [4, 3]
    assert len(rows) == 2, rows
    labels[0, 0] += 1
    changed = {row['path']: row for row in recorder.fingerprint_batch(held)}
    assert changed['batches/0/labels']['sha256'] != by_name['batches/0/labels']['sha256']
    assert changed['batches/0/input_ids'] == by_name['batches/0/input_ids']
    recorder.started = 1
    try:
        recorder.fingerprint_batch(held)
    except RuntimeError:
        pass
    else:
        raise AssertionError('fingerprinting inside timed interval was allowed')
    print('STEP_RECORD_CPU_PASS')


if __name__ == '__main__':
    main()
