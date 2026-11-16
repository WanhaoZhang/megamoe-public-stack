"""Serialize qualified diagnostics; forbid timing while this container compiles."""
import json
import subprocess
import time
from controller import TASK,CONTAINER,run_job


def phase(name):
    (TASK/'evidence/causal-stages-live.json').write_text(json.dumps(dict(stage=name,time=time.time(),complete=False),indent=2))
    print('CAUSAL_STAGE',name,flush=True)


def compiling():
    ps=subprocess.check_output(['docker','top',CONTAINER,'-eo','pid,comm'],text=True)
    return any(any(word in line for word in ('bisheng','cmake','clang')) for line in ps.splitlines()[1:])


def builds_ready():
    try:
        for mode in (0,1):
            path=TASK/f'source/native-balanced/worker-variants/owner-window{mode}'
            manifest=json.loads((path/'build.json').read_text())
            if not manifest.get('so_sha256') or not (path/'libascend_deepep_megaflux_early_bf16.so').is_file():return False
        return True
    except (FileNotFoundError,json.JSONDecodeError):return False


def main():
    gates=json.loads((TASK/'evidence/prepared-plan-gates-v3.json').read_text())
    if len(gates)!=6 or not all(g['numerical_gate_passed'] for g in gates):raise RuntimeError('Prepared gate incomplete')
    phase('incoming-dY-and-identical-assignment-repeatability')
    result=run_job('balanced','validate','diagnostic-incoming-native-self-vio5-v1',
        ['--incoming-gradient-diagnostic','--root',str(TASK),'--planner','native','--maxvio','5','--steps','3'],1800)
    if not result['qualified']:raise RuntimeError('Incoming-gradient diagnostic failed')
    phase('waiting-for-owned-CPU-builds')
    for _ in range(600):
        if builds_ready() and not compiling():break
        time.sleep(3)
    else:raise RuntimeError('Owner-window build is not complete; inspect exact builder before restarting')
    subprocess.run(['docker','exec',CONTAINER,'bash','--noprofile','--norc',str(TASK/'tools/check_owner_window.sh')],check=True)
    phase('owner-window-numerical-qualification')
    from owner_window_qualification import main as qualify
    qualify()
    if compiling():raise RuntimeError('Compiler activity before performance diagnosis')
    phase('identical-plan-and-source-quota-causal-timing')
    from prepared_plan_pilot import main as prepared
    prepared()
    if compiling():raise RuntimeError('Compiler activity before window screen')
    phase('owner-service-window-screen')
    from owner_window_screen import main as screen
    screen()
    (TASK/'evidence/causal-stages-live.json').write_text(json.dumps(dict(stage='complete',time=time.time(),complete=True),indent=2))


if __name__=='__main__':main()
