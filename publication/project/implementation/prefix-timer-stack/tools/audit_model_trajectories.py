"""Cross-policy complete-model trajectory check, additional to independent gradients.

This is a consistency audit against the retained MoonEP policy. It does not
replace the independent dense small-shape forward/backward reference gate.
"""
import argparse
import json
from pathlib import Path
import math
from campaign_records import verified_contention

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--campaign',action='append',required=True);ap.add_argument('--out',type=Path);ap.add_argument('--require-complete',action='store_true')
    ap.add_argument('--reference',default='moonep');ap.add_argument('--eligible-only',action='store_true')
    ap.add_argument('--observational',action='store_true',help='Retain the original strict result, but exit on coverage/nonfinite failures only. Different trained weights are not an independent gradient reference.')
    a=ap.parse_args()
    all_rows=[row for campaign in a.campaign for row in json.loads((a.root/f'evidence/campaign-{campaign}.json').read_text())]
    if a.eligible_only:all_rows=[r for r in all_rows if r['qualified'] and not verified_contention(a.root,r)]
    reference={};comparisons=[]
    for row in all_rows:
        if row['planner']!=a.reference or not row['qualified']:continue
        for rank in range(8):
            values=[json.loads(x) for x in (a.root/f'evidence/{row["tag"]}/steps-rank{rank}.jsonl').read_text().splitlines()]
            for value in values:reference.setdefault((row['maxvio'],rank,value['step']),value)
    for row in all_rows:
        if not row['qualified']:continue
        loss_error=0.;norm_error=0.;count=0;finite=True
        for rank in range(8):
            values=[json.loads(x) for x in (a.root/f'evidence/{row["tag"]}/steps-rank{rank}.jsonl').read_text().splitlines()]
            for value in values:
                key=(row['maxvio'],rank,value['step'])
                if key not in reference:continue
                base=reference[key]
                finite=finite and all(math.isfinite(v) for v in (value['loss'],base['loss'],value['grad_norm'],base['grad_norm']))
                loss_error=max(loss_error,abs(value['loss']-base['loss'])/max(abs(base['loss']),1e-12))
                norm_error=max(norm_error,abs(value['grad_norm']-base['grad_norm'])/max(abs(base['grad_norm']),1e-12))
                count+=1
        comparisons.append(dict(tag=row['tag'],planner=row['planner'],maxvio=row['maxvio'],compared_rank_steps=count,
             maximum_relative_loss_difference=loss_error,maximum_relative_global_norm_difference=norm_error,
             coverage_passed=(count==200 if a.require_complete else True) and finite,
             finite=finite,
             passed=(count==200 if a.require_complete else True) and finite and loss_error<=.004 and norm_error<=.02,
             state='pending_reference' if not count else 'compared'))
    result=dict(passed=all(row['passed'] for row in comparisons),runs=len(comparisons),pending_reference=sum(x['state']=='pending_reference' for x in comparisons),
                scope='cross-policy trained-trajectory observation only; weights diverge after the first Adam update; reference='+a.reference,
                original_thresholds=dict(loss_relative=.004,global_norm_relative=.02),
                coverage_passed=bool(comparisons) and all(x['coverage_passed'] for x in comparisons),
                warning_tags=[x['tag'] for x in comparisons if not x['passed']],
                observational=a.observational,eligible_only=a.eligible_only,rows=comparisons)
    destination=a.out or a.root/'analysis/trajectory-audit.json';destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2));print(json.dumps({key:value for key,value in result.items() if key!='rows'}))
    if not result['coverage_passed'] or (not a.observational and not result['passed']):raise SystemExit(1)

if __name__=='__main__':main()
