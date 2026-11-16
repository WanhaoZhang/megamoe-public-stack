#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import json,statistics
p=Path('/home/z00956592/megaflux/single-mb-fbw-planner-study')
receipt=p/'evidence/codesign-pilot-v1.json'
if not receipt.exists():print('NO_COMPLETED_CELLS');raise SystemExit()
for r in json.loads(receipt.read_text()):
 if not r['qualified']:continue
 directory=p/'evidence'/r['tag']
 ranks=[[json.loads(l) for l in (directory/f'steps-rank{k}.jsonl').read_text().splitlines()] for k in range(8)]
 assert all(len(x)==25 for x in ranks)
 values=[max(x[s]['milliseconds'] for x in ranks) for s in range(5,25)]
 print(json.dumps(dict(vio=r['maxvio'],planner=r['planner'],worker=r['worker'] or 'legacy',
    mean_ms=statistics.mean(values),median_ms=statistics.median(values),independent_repeats=1)))
PY
