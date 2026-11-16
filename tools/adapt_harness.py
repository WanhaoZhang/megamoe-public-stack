"""Copy and retarget only the required frozen harness into this new project."""
from pathlib import Path
import argparse
import shutil

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/z00956592/megaflux/single-mb-fbw-planner-study'
OLD='/home/z00956592/megaflux/planner-executor-ep8-lab'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);a=ap.parse_args()
    for name in ('train.py','launch.sh','controller.py','gate_unified_plan.py','validate.py','gate_pool_replay.py'):
        text=(a.baseline/'tools'/name).read_text().replace(OLD,REMOTE)
        text=text.replace("CONTAINER = 'planner-executor-ep8-lab'","CONTAINER = 'single-mb-fbw-planner-study'")
        text=text.replace("'echo','eplb_fixed','step_time')","'echo','eplb_fixed','step_time',*[f'eplb_budget_{b}' for b in (0,1,2,4,8,16,32,64,128,256)])")
        if name=='gate_pool_replay.py':
            text=text.replace("'echo','eplb_fixed','step_time'),", "'echo','eplb_fixed','step_time',*[f'eplb_budget_{b}' for b in (0,1,2,4,8,16,32,64,128,256)]),")
        if name=='gate_unified_plan.py':
            text=text.replace("for name in ('moonep','native','ultra_ep','echo','eplb_fixed'):",
                "for name in ('moonep','native','ultra_ep','echo','eplb_fixed','eplb_budget_0','eplb_budget_8','eplb_budget_32',f'eplb_budget_{ep*slots}'):")
            text=text.replace('proposal=getattr(ref,name)(source,home,slots)',
                "proposal=(ref.eplb_fixed(source,home,slots,int(name.rsplit('_',1)[1])) if name.startswith('eplb_budget_') else getattr(ref,name)(source,home,slots))")
        (ROOT/'tools'/name).write_text(text,newline='\n')
    guard=Path(r'D:/github/MoE负载不均/experiments/2026-11-15-multicard-system/host_guard.py').read_text()
    guard=guard.replace("Path(__file__).resolve().parent","Path(__file__).resolve().parents[1]")
    guard=guard.replace('/tmp/moe-codesign-system-20261115-8npu.lock','/tmp/single-mb-fbw-planner-study-8npu.lock')
    (ROOT/'tools/host_guard.py').write_text(guard)
    print('adapted required harness; old project untouched')


if __name__=='__main__':main()
