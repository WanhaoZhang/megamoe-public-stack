"""Audit actual diagnostic replay exits without inferring skipped operator FLOPs."""
from pathlib import Path
from collections import Counter
import json

root=Path(__file__).resolve().parent
result={}
for label,tag in [('native','timer-v2-matched-legacy-primary'),('easy_home','timer-v2-matched-primary')]:
    per_rank=[]
    for rank in range(8):
        rows=[json.loads(line) for line in
              (root/'evidence'/tag/f'framework-events-rank{rank}.jsonl').read_text().splitlines()]
        counts=Counter(row['name'] for row in rows if row['step']==2 and
                       row.get('checkpoint_early_stop') and not row.get('initialization'))
        per_rank.append(dict(counts))
    result[label]=per_rank
output=dict(role='diagnostic invocation audit; not skipped-FLOP or isolated timing attribution',
    per_rank=result,source_tags=['timer-v2-matched-legacy-primary','timer-v2-matched-primary'],
    update=2,group_complete_replay_required_for_collective_sequence=True)
(root/'diagnostics/checkpoint-early-stop-audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({key:value[0] for key,value in result.items()}))
