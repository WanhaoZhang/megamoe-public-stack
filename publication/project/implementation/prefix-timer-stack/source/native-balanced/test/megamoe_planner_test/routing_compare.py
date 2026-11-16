"""Validate policy changes separately from bitwise-equivalent validator fusion."""
import argparse,json
from pathlib import Path
from common import check_pair,summarize_samples
from compare_results import read_run,precision
from routing_variants import expected_direct


def candidate_precision(original,baseline,candidate,workers=1):
    ref,a,b=map(read_run,(original,baseline,candidate))
    check_pair(a,b)
    variant=b[0]['routing_variant']
    direct=expected_direct(b[0]['case'],b[0]['calibration'],variant)
    same_plan=not direct
    if same_plan:
        check_pair(a,b,require_plan=True)
        if any(x['outputs']!=y['outputs'] for x,y in zip(a,b)):
            raise ValueError('same-plan candidate differs from its control outputs')
    if direct and any(r['plan_status'][0] or r['actual_rank_rows']!=r['calibration']['owner_rank_rows'] for r in b):
        raise ValueError('direct path changed original routes')
    if workers>1:
        from offline.parallel_precision import precision as parallel_precision
        result=parallel_precision([original,candidate],workers=workers)
    else:
        result=precision([original,candidate])
    result.update(same_plan_control_bitwise=same_plan,direct=direct,
                  original_bitwise=all(x['outputs']==y['outputs'] for x,y in zip(ref,b)),
                  baseline_directory=str(baseline))
    return result


def performance(runs):
    records=[read_run(d) for d in runs]
    for run in records:
        check_pair(records[0],run)
        if any(r['mode']!='performance' for r in run):raise ValueError('not performance')
        if any(x['dependency_digest']!=y['dependency_digest'] for x,y in zip(records[0],run)):
            raise ValueError('dependency changed')
    rows=[dict(directory=str(d),variant=r[0]['routing_variant'],plan_status=r[0]['plan_status'],
               **summarize_samples(r)) for d,r in zip(runs,records)]
    if [r['variant'] for r in rows]!=['strict-fused','direct-fused','direct-fused','strict-fused']:
        raise ValueError('require complete strict/direct/direct/strict order')
    paired=[]
    for direction,base,candidate in [('forward',0,1),('reverse',3,2)]:
        x,y=rows[base],rows[candidate]
        paired.append(dict(direction=direction,comparison='policy',
            change_percent=(y['median_ms']/x['median_ms']-1)*100,
            p95_change_percent=(y['p95_ms']/x['p95_ms']-1)*100))

    return dict(pass_=True,case=records[0][0]['case'],calibration=records[0][0]['calibration'],runs=rows,paired=paired)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('precision','performance'),required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--workers',type=int,choices=range(1,5),default=1);a=p.parse_args()
    if a.mode=='precision':
        if len(a.runs)!=3:p.error('original, control and candidate required')
        result=candidate_precision(*a.runs,workers=a.workers)
    else:result=performance(a.runs)
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(pass_=result['pass_'],report=str(a.out))))
    if not result['pass_']:raise SystemExit(1)

if __name__=='__main__':main()
