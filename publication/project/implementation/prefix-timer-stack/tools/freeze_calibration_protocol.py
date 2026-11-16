"""Freeze independent calibration decisions before observing its measurements."""
import json
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--extend-uniform',action='store_true');args=ap.parse_args()
protocol=dict(selection_bank='calibration',selection_targets=[.3,7,18],selection_steps=10,
              selection_warmup=5,selection_policies=['moonep','native','ultra_ep','echo','eplb_fixed'],
              selection_objective='geometric mean of the three max8rank mean complete-step times',
              selection_tie_break=['moonep','native','ultra_ep','echo','eplb_fixed'],timer_targets=[7],timer_steps=4,
              primary_routes_used_for_fitting=False,owner_fixed=True,
              additional_branch_calibration=dict(bank='calibration-uniform',target=0,planner='moonep',
                  timer_off_steps=10,timer_on_steps=4,used_for_initial_policy_selection=False,
                  reason='source-confirmed no-replica home32/gradient.ready branch'))
path=ROOT/'configs/calibration-protocol.json'
if path.exists() and json.loads(path.read_text())!=protocol:
    if not args.extend_uniform or (ROOT/'evidence/calibration-selection.json').exists():raise RuntimeError('protocol already differs or calibration data exists')
    backup=ROOT/'evidence/calibration-protocol-before-uniform-20261114.json'
    if not backup.exists():backup.write_bytes(path.read_bytes())
path.write_text(json.dumps(protocol,indent=2))
print(path)
