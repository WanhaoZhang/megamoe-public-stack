"""Max across ranks for each full step, then mean of post-warmup steps."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--receipt',default='priority-pilot-v1.json');a=ap.parse_args()
    runs=json.loads((a.project/'evidence'/a.receipt).read_text());result=[]
    for run in runs:
        if not run['qualified']:continue
        root=a.project/'evidence'/run['tag']
        records=[[json.loads(line) for line in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        if any(len(x)!=25 for x in records):raise ValueError('Incomplete steps')
        # Keep the original harness's complete wall-clock boundary.
        timekey=next(k for k in ('milliseconds','step_wall_ms','wall_ms','step_ms') if k in records[0][0])
        values=[max(records[r][s][timekey] for r in range(8)) for s in range(5,25)]
        result.append(dict(tag=run['tag'],maxvio=run['maxvio'],planner=run['planner'],worker=run['worker'],
            mean_ms=statistics.mean(values),median_ms=statistics.median(values),steps=20,
            independent_repeats=1,peak_driver_MiB=run['peak_driver_MiB']))
    destination=a.project/'analysis';destination.mkdir(exist_ok=True)
    (destination/(Path(a.receipt).stem+'-summary.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
