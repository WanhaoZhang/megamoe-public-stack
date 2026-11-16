"""Audit immutable identities before comparing instrumented full-update arms."""
import argparse
import json
from pathlib import Path
import statistics
from verify_training_run import verify

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',default='prepared-plan-pilot-v2.json')
    args=parser.parse_args();runs=json.loads((ROOT/'evidence'/args.receipt).read_text());rows=[]
    identities={};plans={}
    for run in runs:
        directory=ROOT/'evidence'/run['tag'];checked=verify(directory)
        current=[r['identities'] for r in checked]
        reference=identities.setdefault(run['maxvio'],current)
        if current!=reference:raise ValueError(f"Initial condition mismatch: {run['tag']}")
        for rank in range(8):
            manifest=json.loads((directory/f'prepared-plan-manifest-rank{rank}.json').read_text())
            if not manifest['diagnostic_only'] or manifest['online_score']:raise ValueError('Diagnostic score boundary')
            records=manifest['records']
            if len(records)!=50:raise ValueError('Preparation did not cover all actual fixtures')
            if run['proposal']=='original':
                ref=plans.setdefault((run['maxvio'],run['planner'],rank),records)
                if ref!=records:raise ValueError('Online/quota/body changed the frozen plan')
        records=[[json.loads(l) for l in (directory/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        values=[max(records[r][s]['milliseconds'] for r in range(8)) for s in range(5,25)]
        rows.append(dict(tag=run['tag'],maxvio=run['maxvio'],planner=run['planner'],kind=run['kind'],
            proposal=run['proposal'],mean_ms=statistics.mean(values),median_ms=statistics.median(values),
            postwarmup_updates=20,independent_repeats=1,peak_driver_MiB=run['peak_driver_MiB']))
    indexed={(r['maxvio'],r['planner'],r['proposal'],r['kind']):r['mean_ms'] for r in rows};contrasts=[]
    for vio in sorted(identities):
        for policy in ('native','eplb_budget_32','eplb_fixed'):
            key=lambda kind:(vio,policy,'original',kind)
            if all(key(k) in indexed for k in ('online','quota','body')):
                online,quota,body=(indexed[key(k)] for k in ('online','quota','body'))
                contrasts.append(dict(maxvio=vio,planner=policy,online_minus_quota_ms=online-quota,
                    quota_minus_body_ms=quota-body,online_minus_body_ms=online-body,
                    online_minus_body_fraction=(online-body)/online))
    output=dict(diagnostic_only=True,online_score=False,complete_cells=len(rows),rows=rows,contrasts=contrasts,
        limitations=['One process per cell, sequential execution; no significance claim',
            'All arms retain prepared storage and synchronizing route-value checks',
            'Body skips counts/common compilation/assignment/validation as well as policy generation',
            'Contrasts include interactions; not isolated host/kernel timer durations',
            'Reverse replica slots changes forward physical order, but shared-logical backward sorting may preserve backward order'])
    path=ROOT/'analysis'/f'{Path(args.receipt).stem}-summary.json';path.write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(complete_cells=len(rows),contrasts=contrasts),indent=2))


if __name__=='__main__':main()
