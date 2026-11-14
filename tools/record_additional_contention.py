"""Append another evidenced exclusion window without rewriting earlier ones."""
import argparse
import hashlib
import json
from controller import TASK

ap=argparse.ArgumentParser();ap.add_argument('--hold-suffix',required=True);ap.add_argument('--resume-suffix',default='');options=ap.parse_args()
hold=json.loads((TASK/f'evidence/OWN-QUEUE-INFRASTRUCTURE-HOLD{options.hold_suffix}.json').read_text())
resume=json.loads((TASK/f'evidence/OWN-QUEUE-RESUMED{options.resume_suffix}.json').read_text())
begin,end=resume['timestamp'],hold['timestamp']
cid='e4ec55cc554dd8f7ed348dfc70c33efbfb079f0c2f23eec723be7ec85df8cf56'
proof=None
for number,line in enumerate((TASK/'evidence/driver-process-observations.jsonl').read_text().splitlines(),1):
    observed=json.loads(line)
    if begin<=observed['timestamp']<=end and any(cid in p.get('cgroup','') for p in observed['foreign_or_unidentified']):
        proof=dict(file='evidence/driver-process-observations.jsonl',line=number,sha256=hashlib.sha256(line.encode()).hexdigest(),timestamp=observed['timestamp']);break
if proof is None:raise RuntimeError('no independent foreign GPU PID/cgroup proof in the proposed window')
path=TASK/'evidence/EXTERNAL-CONTENTION.json';ledger=json.loads(path.read_text())
if not any(x['begin']==begin and x['end']==end for x in ledger['intervals']):
    ledger['intervals'].append(dict(begin=begin,end=end,container=cid,container_name='/agy-live-a5',proof=proof,
        boundary='conservative entire own resumed epoch until the next between-job hold, after foreign activity recurred',
        exclusion_rule='all task process windows overlapping this interval, independent of policy, speed or success',
        root_cause_caution='foreign occupancy is proven; causation of HCCL faults remains unproven'))
rows=json.loads((TASK/'evidence/campaign-primary-existing.json').read_text())
ledger['excluded_tags']=[r['tag'] for r in rows if any(r['start']<x['end'] and r['finish']>x['begin'] for x in ledger['intervals'])]
path.write_text(json.dumps(ledger,indent=2))
print(json.dumps(dict(intervals=len(ledger['intervals']),excluded_tags=ledger['excluded_tags'])),flush=True)
