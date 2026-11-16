"""Compare actual group manifests; shape counts are not communication costs."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
summary=json.loads((root/'upstream-maxutil-summary.json').read_text())
records=[]
def load(tag,rank):
    path=root/'evidence'/tag/f'temporal-groups-rank{rank}.jsonl'
    groups=[json.loads(line) for line in path.read_text().splitlines()]
    return groups,[{key:row[key] for key in ('step','keys','native_waves')} for row in groups]
for pattern in ('same','alternating','random'):
    rows={row['label']:row for row in summary['runs'] if row['pattern']==pattern}
    if not {'fixed_home','upstream_home'}.issubset(rows):continue
    details={};manifests={}
    for label in ('fixed_home','upstream_home'):
        groups,manifest=load(rows[label]['tag'],0)
        for rank in range(1,8):
            if load(rows[label]['tag'],rank)[1]!=manifest:raise RuntimeError('Rank group/wave consensus mismatch')
        waves=[wave for group in groups for wave in group['native_waves']]
        if len(waves)!=400 or any(wave['rows']!=16384 for wave in waves):
            raise RuntimeError('Primary-shape total waves/rows changed')
        details[label]=dict(groups=len(groups),waves=len(waves),
            mixed_layer_waves=sum(len(wave['domain_layers'])>1 for wave in waves),
            declared_home_planes_per_rank=sum(32*len(wave['domain_layers']) for wave in waves),
            label_scope='Declared logical-domain shape; not active FLOPs, transfer bytes or measured gradient time')
        manifests[label]=manifest
    records.append(dict(pattern=pattern,execution_group_and_wave_manifests_identical=manifests['fixed_home']==manifests['upstream_home'],
        all_rank_consensus=True,details=details))
result=dict(role='Offline audit of completed online runs; no data supplied to any planner',records=records,
    causal_gradient_tail_proved=False,counts_are_not_communication_bytes=True,
    no_cross_rank_clock_alignment_used=True)
(root/'diagnostics/upstream-execution-geometry.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
