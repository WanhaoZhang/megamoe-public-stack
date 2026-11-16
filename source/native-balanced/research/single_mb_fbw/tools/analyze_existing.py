"""Summarize original diagnostic calls without merging independent rank clocks."""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, required=True)
    args = ap.parse_args()
    groups = collections.defaultdict(list)
    reduced = args.baseline / 'calibration/reduced'
    with (reduced / 'calls.csv').open() as stream:
        for row in csv.DictReader(stream):
            if float(row['maxvio']) == 7 and int(row['step']) in (1, 2, 3):
                groups[(row['planner'], row['phase'], row['recompute'])].append(row)
    result = []
    for (planner, phase, recompute), rows in sorted(groups.items()):
        result.append(dict(planner=planner, phase=phase, recompute=recompute, calls=len(rows),
            **{key: statistics.mean(float(r[key]) for r in rows)
               for key in ('kernel_envelope_ms', 'active_experts', 'replicas', 'rows')}))
    out = ROOT / 'analysis'
    out.mkdir(exist_ok=True)
    payload = dict(diagnostic_only=True, source=str(reduced), steps=[1, 2, 3], maxvio=7,
        clock_semantics='Per-call rank-local envelopes; overlapping lanes never summed.', groups=result)
    (out / 'existing-calls.json').write_text(json.dumps(payload, indent=2))
    lines = ['# Existing planner diagnostic evidence', '',
        'Independent calibration MaxVio7, steps1/2/3, 8 ranks x 2 layers =48 calls/phase.',
        'Timer-on measurements; these are not formal step-time performance claims.', '',
        '| Policy | Phase | Replay | Mean kernel ms | Mean active replicas/rank |',
        '|---|---|---|---:|---:|']
    for r in result:
        lines.append(f"|{r['planner']}|{r['phase']}|{r['recompute']}|{r['kernel_envelope_ms']:.3f}|{r['replicas']:.3f}|")
    lines += ['', '## Interpretation and pending interventions', '',
        'EPLB-fixed assigns its full instance budget before coalescing same-rank instances. '
        'Its substantially greater active-copy count coincides with longer forward and backward envelopes. '
        'This is correlation, not a measured causal decomposition.', '',
        'Required interventions: fixed capacity with instance-budget sweep; fixed placement with '
        'source-quota alternatives; identical-plan online/replay; per-expert order intervention. '
        'The dH/dX increase means gradient bytes alone are an insufficient explanation.', '',
        'Other planners being close in total time does not prove optimality. Old step_time adopted '
        'zero new plans, so its experiment did not test execution of an improved candidate.', '',
        'Raw source: '+str(reduced)]
    # The living REPORT.md includes later device results and must not be reset
    # by reproducing this original-evidence summary.
    (out / 'existing-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
