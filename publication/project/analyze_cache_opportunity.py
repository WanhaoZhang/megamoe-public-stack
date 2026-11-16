"""Use original closed-run accesses to separate cold misses from avoidable misses.

This is a sequence-conditioned byte lower bound, not an online policy score.
"""
from collections import defaultdict
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'causal-diagnostics-20261116/evidence/causal-v1-main-b1'
PLANE=24*1024*1024
records=[]
for rank in range(8):
    rows=[json.loads(line) for line in (SOURCE/f'lifetime-events-rank{rank}.jsonl').read_text().splitlines()]
    steps=defaultdict(list)
    for row in rows:
        if row['name']=='prepare':steps[row['version']].append(row)
    for version,calls in sorted(steps.items()):
        seen=set();hits=misses=cold=repeated_misses=0;observed_remote=0
        for row in calls:
            observed_remote+=row['payload_delta']['weight_remote_bytes']
            for entry in row['cache_uses']:
                key=(entry['layer'],entry['expert'])
                if entry['hit']:hits+=1
                else:
                    misses+=1
                    if key in seen:repeated_misses+=1
                    else:cold+=1
                seen.add(key)
        records.append(dict(rank=rank,version=version,accesses=hits+misses,hits=hits,misses=misses,
                            cold_misses=cold,repeated_misses=repeated_misses,distinct_keys=len(seen),
                            remote_bytes=observed_remote,access_miss_bytes_match=observed_remote==misses*PLANE))
output=dict(source=str(SOURCE),source_scope='Old rich diagnostic only, 3 updates, same artificial route; not formal steady-state performance.',
            destination_rank_is_in_outer_record=True,cache_entries_per_rank=8,
            records=records,miss_byte_reconciliation=all(row['access_miss_bytes_match'] for row in records),
            total_misses=sum(row['misses'] for row in records),total_cold_misses=sum(row['cold_misses'] for row in records),
            total_repeated_misses=sum(row['repeated_misses'] for row in records),
            max_distinct_keys_per_rank_update=max(row['distinct_keys'] for row in records),
            interpretation='An observed first use of a (version,layer,expert,destination) needs a fresh payload unless prefetched. Prefetch may hide it but cannot remove its bytes. Repeated misses are an upper bound on byte reductions from more capacity for this exact sequence; changing plans/order/version changes the bound.')
(ROOT/'diagnostics/cache-opportunity-bound.json').write_text(json.dumps(output,indent=2)+'\n',newline='\n')
print(json.dumps({key:value for key,value in output.items() if key!='records'},indent=2))
