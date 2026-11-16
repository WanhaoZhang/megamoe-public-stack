"""Same-host envelopes only: no merging device origins or summing nested scopes."""
from collections import defaultdict
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent
states = [json.loads(path.read_text()) for path in (ROOT/'diagnosis-v4.json', ROOT/'diagnosis-v5.json') if path.exists()]
state = dict(runs=[run for stage in states for run in stage['runs']])
results = []
for run in state['runs']:
    if not run['passed'] or not run['tag'].startswith('host-main-'):
        continue
    folder = ROOT/'evidence'/run['tag']
    ranks = [[json.loads(line) for line in (folder/f'host-boundary-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
    operators = defaultdict(dict)
    durations = defaultdict(list)
    for rank, rows in enumerate(ranks):
        for row in rows:
            if row['update'] is None:
                continue
            phase = 'warmed' if row['update'] >= 5 else 'prewarm'
            durations[(phase,row['name'])].append((row['host_return_ns']-row['host_begin_ns'])/1e6)
            if row['name'] == 'operator.backward':
                key = (row['update'],tuple(row['invocation']))
                if rank in operators[key]:
                    raise RuntimeError('Duplicate actual backward identity')
                operators[key][rank] = row
    skews = []
    outliers = []
    for (step,key), records in sorted(operators.items()):
        if set(records) != set(range(8)):
            raise RuntimeError('Incomplete cross-rank operator match')
        first = min(records, key=lambda rank: records[rank]['host_begin_ns'])
        last = max(records, key=lambda rank: records[rank]['host_begin_ns'])
        begin, end = records[first]['host_begin_ns'], records[last]['host_begin_ns']
        skew = (end-begin)/1e6
        skews.append(dict(update=step,invocation=list(key),skew_ms=skew,first_rank=first,last_rank=last))
        if skew >= 10:
            blockers = []
            for row in ranks[last]:
                overlap = (min(row['host_return_ns'],end)-max(row['host_begin_ns'],begin))/1e6
                if overlap >= 1:
                    blockers.append(dict(name=row['name'],invocation=row['invocation'],overlap_ms=overlap,
                                         full_host_ms=(row['host_return_ns']-row['host_begin_ns'])/1e6))
            outliers.append(dict(update=step,invocation=list(key),skew_ms=skew,first_rank=first,last_rank=last,
                                 late_rank_overlapping_scopes=sorted(blockers,key=lambda row:-row['overlap_ms'])[:16]))
    phases = {}
    for phase in ('prewarm','warmed'):
        values = [row['skew_ms'] for row in skews if (row['update']>=5) == (phase=='warmed')]
        phases[phase] = dict(calls=len(values),median_ms=statistics.median(values),max_ms=max(values),
                             over10ms=sum(value>=10 for value in values),over100ms=sum(value>=100 for value in values))
    methods = [dict(phase=phase,name=name,calls=len(values),median_host_ms=statistics.median(values),max_host_ms=max(values))
               for (phase,name),values in sorted(durations.items())]
    results.append(dict(tag=run['tag'],phases=phases,per_update=[dict(update=step,
        median_skew_ms=statistics.median([row['skew_ms'] for row in skews if row['update']==step]),
        max_skew_ms=max(row['skew_ms'] for row in skews if row['update']==step)) for step in range(11)],
        outliers=outliers,method_envelopes=methods,matched_operator_entries=skews))
output = dict(diagnostic_only=True,results=results,
              interpretation='Same host monotonic entry/return envelopes. Entry skew is not an additive step cost or proof of avoidable device wait. Overlapping nested methods are not summed.',
              rich_capture_comparison_limit='Old rich captures use TimerON and additional GPU Events/snapshots, only updates0–2. A difference does not identify one Event limit or one causal factor.')
(ROOT/'diagnostics/host-boundary-analysis.json').write_text(json.dumps(output,indent=2)+'\n',newline='\n')
print(json.dumps([dict(tag=row['tag'],phases=row['phases']) for row in results],indent=2))
