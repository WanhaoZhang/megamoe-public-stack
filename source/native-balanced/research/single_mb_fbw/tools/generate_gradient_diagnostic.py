"""Extend frozen paired-state diagnostics with parameter-level error evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'tools/fixed_state_pair.py'
    text=source.read_text()
    replacements={
        "choices=('moonep','native','ultra_ep','echo','eplb_fixed','step_time')":
            "choices=('moonep','native','ultra_ep','echo','eplb_fixed','step_time',*[f'eplb_budget_{b}' for b in (0,1,2,4,8,16,32,64,128,256)])",
        "{'step_time':pool.policy,'native':CommonOnlinePlanner":"{'reference':pool.policy,'candidate':CommonOnlinePlanner",
        "pool.policy=policy_pairs[key][name];pool.planner_name=name":"pool.policy=policy_pairs[key][name];pool.planner_name=args.planner if name=='reference' else 'native'",
        "for policy_name in ('step_time','native'):":"for policy_name in ('reference','candidate'):",
        "policy_name=='step_time'":"policy_name=='reference'",
        "policy_name=='native'":"policy_name=='candidate'",
        "errors={};aggregate_error":"errors={};parameter_metrics={};aggregate_error",
        "errors[name]=metric['relative_l2']":"errors[name]=metric['relative_l2'];parameter_metrics[name]=metric",
        "all_parameter_max_gradient_relative_l2=max(errors.values()),output_and_input_errors=output_errors":"all_parameter_max_gradient_relative_l2=max(errors.values()),parameter_gradient_metrics=parameter_metrics,output_and_input_errors=output_errors",
        "range(25)":"range(args.steps)",
        "{'step_time':50,'native':50}":"{'reference':args.steps*args.layers,'candidate':args.steps*args.layers}",
        "dict(original=100,recompute=100,backward=100)":"dict(original=4*args.steps,recompute=4*args.steps,backward=4*args.steps)",
        "passed=True,steps=25":"passed=True,steps=args.steps,reference_policy=args.planner,candidate_policy='native'",
    }
    for old,new in replacements.items():
        if old not in text:raise ValueError(f'Frozen source marker absent: {old}')
        text=text.replace(old,new)
    destination=ROOT/'tools/gradient_diagnostic.py'
    destination.write_text(text,newline='\n')
    receipt=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        changes='reference selected by existing --planner; candidate Native; args.steps audit; per-parameter energies',
        thresholds_unchanged=True,performance_measurement=False)
    (ROOT/'evidence/gradient-diagnostic-generation.json').write_text(json.dumps(receipt,indent=2))
    print(receipt)


if __name__=='__main__':main()
