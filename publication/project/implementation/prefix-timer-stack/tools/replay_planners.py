"""Replay the frozen108-cell primary protocol under a new namespace.

The original source/model/data are reused. This never refits a model, publishes
changes, overwrites original results, or treats plan-only timing as training.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import controller
from campaign_search import run_stage,stable_points


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True,help='new letters/digits/underscore namespace')
    ap.add_argument('--container',default='planner-executor-ep8-lab')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--validation',action='store_true',help='also replay fourth/heldout and conditionally the three frozen ablations')
    ap.add_argument('--dry-run',action='store_true',help='print the frozen replay contract without launching or writing a campaign')
    options=ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9_]+',options.name) or options.name in ('primary','formal','primary_existing','primary_step_time'):
        raise ValueError('use a new replay namespace')
    root=controller.TASK
    frozen=json.loads((root/'evidence/STEP-TIME-FREEZE.json').read_text())
    if controller.source_identity()!=frozen['source']:raise RuntimeError('replay sources differ from the qualified final source')
    model=root/'configs/step-time-full.json'
    if hashlib.sha256(model.read_bytes()).hexdigest() not in frozen['approved_model_shas']:
        raise RuntimeError('full calibrated model fingerprint differs')
    if json.loads(model.read_text()).get('variant')!='full':raise RuntimeError('primary replay requires the full model')
    baseline=json.loads((root/'evidence/BASELINE-FREEZE.json').read_text())
    for path,digest in baseline['binaries'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen baseline binary differs: '+path)
    if hashlib.sha256((root/'source/native-balanced/ascend_deepep/lib/libascend_deepep_step_time_planner.so').read_bytes()).hexdigest()!=frozen['plugin_sha256']:
        raise RuntimeError('qualified policy plugin differs')
    if options.dry_run:
        print(json.dumps(dict(namespace=options.name,launches=False,source_and_binary_fingerprints_verified=True,
            main_cells=108,fourth_cells=12 if options.validation else 0,heldout_cells=72 if options.validation else 0,
            conditional_ablation_cells=54 if options.validation else 0,refits=False,
            comparator='reuse the original frozen strongest-existing selection' if options.validation else None),indent=2));return
    acceptance=json.loads((root/'ACCEPTANCE.json').read_text())
    if not acceptance.get('complete'):raise RuntimeError('the original experiment is not fully accepted yet')
    destination=root/'replays'/options.name
    if destination.exists() and not options.resume:raise RuntimeError('replay namespace exists; use --resume only to continue the same contract')
    destination.mkdir(parents=True,exist_ok=True)
    controller.CONTAINER=options.container
    contract=dict(namespace=options.name,container=options.container,main_cells=108,
        existing_first=90,step_time_after_existing=18,steps=25,warmup=5,capacity=163840,
        model_path=str(model),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        no_refitting=True,ordinary_executor_only=True,original_results_overwritten=False,
        primary_route_bank=str(root/'routes'),source_freeze='evidence/STEP-TIME-FREEZE.json',validation=options.validation,
        validation_comparator='original frozen strongest baseline; no reselection')
    path=destination/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise RuntimeError('replay contract changed')
    path.write_text(json.dumps(contract,indent=2)+'\n')
    existing=run_stage(options.name+'-existing',['moonep','native','ultra_ep','echo','eplb_fixed'],3,1,root/'routes',{})
    new=run_stage(options.name+'-step-time',['step_time'],3,1,root/'routes',{'step_time':str(model)})
    (destination/'resolved-primary.json').write_text(json.dumps(existing+new,indent=2)+'\n')
    print('REPLAY_PRIMARY_COMPLETE',options.name,len(existing)+len(new),flush=True)
    if options.validation:
        best=json.loads((root/'configs/strongest-existing.json').read_text())['planner']
        fourth=run_stage(options.name+'-fourth',[best,'step_time'],1,4,root/'routes',{'step_time':str(model)})
        stability=stable_points(existing+new,fourth,best)
        (destination/'stability-four-rounds.json').write_text(json.dumps(stability,indent=2))
        for bank in ('heldout-a','heldout-b'):
            result=run_stage(options.name+'-'+bank,[best,'step_time'],3,1,root/f'routes/{bank}',{'step_time':str(model)})
            (destination/f'resolved-{bank}.json').write_text(json.dumps(result,indent=2))
        if any(x['stable'] for x in stability):
            for variant in ('no_gradient_service','linear_kernel','uniform_topology'):
                alternative=root/f'configs/step-time-{variant.replace("_","-")}.json'
                if hashlib.sha256(alternative.read_bytes()).hexdigest() not in frozen['approved_model_shas']:raise RuntimeError('ablation model changed')
                run_stage(options.name+'-ablation-'+variant,['step_time'],3,1,root/'routes',{'step_time':str(alternative)})
        print('REPLAY_VALIDATION_COMPLETE',options.name,flush=True)


if __name__=='__main__':main()
