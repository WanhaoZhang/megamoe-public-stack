"""Regression: duplicate completion must reject atomically without retirement."""
from pathlib import Path
import json
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'stack/tools'))
from step_lifetime import UpdateContext,ExpertKey

context=UpdateContext(1,1)
context.begin(0)
context.contribute(('a',),ExpertKey(0,0,0,1),0)
context.submit((('a',),))
rejected=False
try:context.complete((('a',),('a',)))
except ValueError:rejected=True
passed=rejected and not context.tickets[('a',)].complete
print(json.dumps(dict(passed=passed,rejected=rejected,retired=context.tickets[('a',)].complete)))
raise SystemExit(0 if passed else 1)
