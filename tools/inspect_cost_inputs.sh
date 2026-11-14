python3 - <<'PY'
import csv,json,pathlib,collections,statistics
r=pathlib.Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
for f in ['calibration/framework/events.csv','calibration/framework/controls.csv','calibration/reduced/expert-phases.csv','calibration/reduced/release-owner.csv','calibration/reduced/calls.csv']:
 with (r/f).open(encoding='utf-8-sig') as h:
  reader=csv.DictReader(h);first=next(reader);print(f,first)
groups=collections.defaultdict(list);policies=collections.defaultdict(list)
with (r/'calibration/framework/events.csv').open(encoding='utf-8-sig') as h:
 for row in csv.DictReader(h):
  if int(row['step'])>0:
   groups[row['category']].append(float(row['exclusive_envelope_ms']))
   if row['category'] in ('control.online_planning','operator.forward','operator.backward'):policies[row['planner']+'.'+row['category']].append(float(row['envelope_ms']))
print('EXCLUSIVE_MEDIANS', {k:(len(v),statistics.median(v)) for k,v in groups.items()})
print('POLICY_MEDIANS', {k:(len(v),statistics.median(v)) for k,v in policies.items()})
print('ROUTES', [p.name for p in (r/'routes/calibration').iterdir()][:12])
PY
