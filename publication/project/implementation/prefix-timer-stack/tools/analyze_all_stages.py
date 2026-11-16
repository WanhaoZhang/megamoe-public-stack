"""Reduce separate banks/stages without mixing denominator or repeat cohorts."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def reduce(names,destination):
    selected=[name for name in names if (ROOT/f'evidence/campaign-{name}.json').exists()]
    if not selected:return
    command=[sys.executable,str(ROOT/'tools/analyze_matrix.py'),'--root',str(ROOT),'--out',str(destination)]
    for name in selected:command+=['--campaign',name]
    subprocess.run(command,check=True)
def trajectories(names,destination,reference='moonep'):
    selected=[name for name in names if (ROOT/f'evidence/campaign-{name}.json').exists()]
    if not selected:return
    command=[sys.executable,str(ROOT/'tools/audit_model_trajectories.py'),'--root',str(ROOT),'--out',str(destination),'--require-complete','--eligible-only','--reference',reference,'--observational']
    for name in selected:command+=['--campaign',name]
    subprocess.run(command,check=True)
    observed=json.loads(destination.read_text())
    if observed['warning_tags']:
        qualification=ROOT/'evidence/fixed-state-pair-acceptance.json'
        assert qualification.exists() and json.loads(qualification.read_text())['passed'], 'trajectory warning requires review and identical-weight correctness evidence before continuing'
def main():
    reduce(['primary-existing','primary-existing-recovery','primary-step-time','primary-step-time-recovery'],ROOT/'analysis')
    for stage in ['fourth-paired','heldout-a','heldout-b','ablation-no_gradient_service','ablation-linear_kernel','ablation-uniform_topology']:
        reduce([stage,stage+'-recovery'],ROOT/'analysis'/stage)
    main=['primary-existing','primary-existing-recovery','primary-step-time','primary-step-time-recovery','fourth-paired','fourth-paired-recovery']
    for variant in ('no_gradient_service','linear_kernel','uniform_topology'):main.extend(['ablation-'+variant,'ablation-'+variant+'-recovery'])
    trajectories(main,ROOT/'analysis/trajectory-audit.json')
    best_file=ROOT/'configs/strongest-existing.json'
    if best_file.exists():
        best=json.loads(best_file.read_text())['planner']
        for bank in ('heldout-a','heldout-b'):trajectories([bank,bank+'-recovery'],ROOT/f'analysis/{bank}/trajectory-audit.json',best)
    subprocess.run([sys.executable,str(ROOT/'tools/analyze_planner_decisions.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/report_progress.py')],check=True)
if __name__=='__main__':main()
