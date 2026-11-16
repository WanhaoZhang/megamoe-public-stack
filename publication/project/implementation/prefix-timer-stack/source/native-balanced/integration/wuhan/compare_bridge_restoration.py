"""Exact same-plan/five-output check across independent native-SO processes."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staged', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--ep', type=int, choices=(2, 4, 8), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--same-restore', action='store_true',
                        help='Compare publication candidates with an identical restore mode.')
    args = parser.parse_args()
    for directory in (args.staged, args.candidate):
        assert read(directory/'acceptance.json')['eligible']
        assert read(directory/'pre.json')['pass_'] and read(directory/'post.json')['pass_']
        assert int((directory/'run.exit').read_text()) == 0
    rows = []
    for rank in range(args.ep):
        a, b = [read(root/f'rank{rank}.json') for root in (args.staged, args.candidate)]
        assert a['pass_'] and b['pass_'] and a['rank'] == b['rank'] == rank
        conditions = lambda r: {k: r[k] for k in ('planner', 'checkpoint', 'gradient_divide_factor')}
        assert conditions(a) == conditions(b), 'different test inputs/settings'
        for key, default in (('tokens', 257), ('hidden', 512), ('ffn', 512), ('limit', 10.)):
            assert a.get(key, default) == b.get(key, default), ('different settings', key)
        if args.same_restore:
            assert a['restore'] == b['restore'], 'restore modes differ'
        else:
            assert a['restore'] == 'staged' and b['restore'] != 'staged'
        assert len(a['rows']) == len(b['rows']) == 3
        assert a['plan_audit'] and b['plan_audit'], 'missing actual execution plans'
        assert len(a['plan_audit']) == len(b['plan_audit'])
        for pa, pb in zip(a['plan_audit'], b['plan_audit']):
            for key in ('placement', 'copies', 'execution_counts'):
                assert pa[key] == pb[key], (rank, key, 'plan mismatch')
        exact_updates = True
        explicit_inputs = True
        for ua, ub in zip(a['rows'], b['rows']):
            assert ua['kind'] == ub['kind']
            assert ua['storage'] == ub['storage'], (rank, ua['kind'], 'storage bits differ')
            has_inputs = bool(ua.get('input_storage')) and bool(ub.get('input_storage'))
            has_updates = bool(ua.get('updated_storage')) and bool(ub.get('updated_storage'))
            if args.same_restore:
                assert has_inputs and has_updates, 'publication comparison requires input and SGD fingerprints'
            if has_inputs:
                assert ua['input_storage'] == ub['input_storage'], (rank, ua['kind'], 'inputs differ')
            if has_updates:
                assert ua['updated_storage'] == ub['updated_storage'], (rank, ua['kind'], 'SGD bits differ')
            explicit_inputs &= has_inputs
            exact_updates &= has_updates
        rows.append(dict(rank=rank, exact_plans=len(a['plan_audit']),
                         exact_five_outputs=True, exact_sgd_updates=exact_updates,
                         explicit_input_hashes=explicit_inputs))
    result = dict(pass_=True, ep=args.ep, staged=str(args.staged),
                  same_restore=args.same_restore,
                  candidate=str(args.candidate), ranks=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print('RESTORE_SAME_PLAN_BITWISE_PASS', args.ep)


if __name__ == '__main__':
    main()
