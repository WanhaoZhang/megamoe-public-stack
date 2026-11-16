"""CPU-only completeness check for a paired collection and serial trace audit."""
import argparse
import json
from pathlib import Path


def verify(out, world):
    out = Path(out)
    backward_none_gm_policy_checks = {}
    for rank in range(world):
        rows = [json.loads((out/kind/f'rank{rank}.json').read_text()) for kind in ('control','timer')]
        control, timer = rows
        assert set(control['outputs']) == set(timer['outputs']) == {'y','dx','dp','dw1','dw2'}
        for kind, row in zip(('control','timer'), rows):
            assert row['pass_'] and row['rank'] == rank and row['world_size'] == world
            checks = [json.loads(line) for line in (out/kind/f'iterations-rank{rank}.jsonl').read_text().splitlines()]
            assert len(checks) == row['checked_iterations'] > 0
            assert all(v['pass_'] and v['outputs'] == row['outputs'] for v in checks)
        assert control['timer'] == 0 and timer['timer'] == 1
        key = 'backward_none_gm_requested'
        if all(key in row for row in rows):
            assert control[key] in (0, 1) and timer[key] in (0, 1), (rank, key)
            assert control[key] == timer[key], (rank, key)
            backward_none_gm_policy_checks[rank] = 'matched'
        else:
            backward_none_gm_policy_checks[rank] = 'unknown'
        assert timer['five_outputs_bitwise_reference']
        for key in ('config','inputs','outputs','copies','plan_tensors'):
            assert control[key] == timer[key], (rank,key)
        # Old paired receipts only supported BF16. New receipts must also
        # agree on the requested and effective G/U format.
        modes = [row.get('saved_gate_up', dict(requested='bf16', resolved='bf16'))
                 for row in rows]
        assert all(modes[0][key] == modes[1][key] for key in ('requested','resolved')), (rank,'saved_gate_up')
        # Positive row quotas can still produce no replicas for a balanced case;
        # require source events, but not copy/restore events unconditionally.
        for phase, kinds in [('forward',(35,37)),('backward',(38,39))]:
            assert all(timer['producer_counts'][phase].get(str(k),0) > 0 for k in kinds)
    audit = json.loads((out/'kernel-trace/trace-export-audit.json').read_text())
    assert audit['pass_'] and audit['ranks'] == list(range(world)) and audit['duration_overlap_count'] == 0
    traces = [out/'kernel-trace'/f'all_{world}_rank_{phase}_trace.json' for phase in ('forward','backward')]
    assert all(p.is_file() and p.stat().st_size > 0 for p in traces)
    return dict(pass_=True, world_size=world, five_outputs_bitwise_TimerOFF_ON=True,
                backward_none_gm_requested_check=('matched' if all(
                    value == 'matched' for value in backward_none_gm_policy_checks.values()) else 'unknown'),
                backward_none_gm_requested_check_by_rank=backward_none_gm_policy_checks,
                checked_all_iterations=True, duration_overlap_count=0,
                traces=[str(p) for p in traces], diagnostic_only=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--world-size',type=int,required=True)
    a=p.parse_args();result=verify(a.out,a.world_size)
    (a.out/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
