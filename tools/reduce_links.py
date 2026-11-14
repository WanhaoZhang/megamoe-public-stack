"""Fit measured local-machine transfer costs and disclose topology resolution."""
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
jobs=json.loads((ROOT/'evidence/calibration-jobs.json').read_text())
link_tag=next(x['tag'] for x in jobs if x.get('mode')=='links')
records=[json.loads((ROOT/f'evidence/{link_tag}/links-rank{rank}.json').read_text()) for rank in range(8)]
rows=[row for record in records for row in record['rows']]
links=[];slopes=np.zeros((8,8));latencies=np.zeros((8,8));within=[]
for source in range(8):
    for destination in range(8):
        if source==destination:continue
        samples=sorted([row for row in rows if row['source']==source and row['destination']==destination and not row['fanout']],key=lambda x:x['bytes'])
        if len(samples)!=4:raise ValueError('not four transfer sizes for every directed peer pair')
        design=np.array([[1,row['bytes']] for row in samples]);y=np.array([row['median_ms'] for row in samples])
        intercept,slope=np.linalg.lstsq(design,y,rcond=None)[0]
        slope=max(float(slope),1e-15);intercept=max(float(intercept),0.)
        slopes[source,destination]=slope;latencies[source,destination]=intercept
        largest=samples[-1];repeats=np.array(largest['milliseconds']);median=float(np.median(repeats))
        jitter=float(np.median(np.abs(repeats-median))/median);within.append(jitter)
        links.append(dict(source=source,destination=destination,latency_ms=intercept,ms_per_byte=slope,
            ps_per_byte=slope*1e9,largest_bytes=largest['bytes'],largest_median_ms=median,
            largest_repeat_min_ms=float(repeats.min()),largest_repeat_max_ms=float(repeats.max()),relative_mad=jitter,
            fit_max_relative_error=float(np.max(np.abs(design@np.array([intercept,slope])-y)/y))))
slopes_nonzero=np.array([x['ms_per_byte'] for x in links])
spread=float((np.quantile(slopes_nonzero,.9)-np.quantile(slopes_nonzero,.1))/np.median(slopes_nonzero))
jitter=statistics.median(within);threshold=max(.05,3*jitter)
fast=min(links,key=lambda x:x['largest_median_ms']);slow=max(links,key=lambda x:x['largest_median_ms'])
separated=slow['largest_repeat_min_ms']>fast['largest_repeat_max_ms']
fanout=[]
for source in range(8):
    group=[x for x in rows if x['source']==source and x['fanout']]
    if len(group)!=7:raise ValueError('missing simultaneous seven-receiver transfer')
    independent=max(x['largest_median_ms'] for x in links if x['source']==source)
    duration=max(x['median_ms'] for x in group)
    fanout.append(dict(source=source,concurrent_receivers=7,worst_receiver_ms=duration,
                       relative_to_slowest_single_peer=duration/independent))
result=dict(schema=1,shape=dict(ranks=8),peer_costs=links,latency_matrix_ms=latencies.tolist(),
    slope_matrix_ms_per_byte=slopes.tolist(),fanout=fanout,
    topology=dict(p10_p90_relative_spread=spread,median_within_pair_relative_mad=jitter,
        prespecified_resolution_threshold=threshold,extreme_repeat_ranges_separated=separated,
        cost_heterogeneity_resolved=spread>threshold and separated,
        interpretation='Measured MTE/SHMEM directed-pair costs on this node. This is not a reconstruction of physical switch wiring.'),
    calibration_clock=[dict(rank=x['rank'],ticks_per_ms=x['ticks_per_ms'],trials=x['clock']) for x in records],
    limitations=['five post-warmup measurements per size and directed pair','fit extrapolates beyond32MiB',
                 'bulk link costs require operator-level utilization/contestion calibration',
                 'rank-local Timer clocks are not merged'])
destination=ROOT/'calibration/links.json';destination.parent.mkdir(exist_ok=True,parents=True)
destination.write_text(json.dumps(result,indent=2));print(json.dumps(result['topology']))
