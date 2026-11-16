"""Require all builds to retire before numerical gates or wall-clock timing."""
import json
import subprocess
import time
from controller import TASK,CONTAINER
from remaining_causal_stages import compiling


def main():
    prior=json.loads((TASK/'evidence/causal-stages-live.json').read_text())
    if not prior.get('complete'):raise RuntimeError('Prior causal queue has not retired')
    for _ in range(1200):
        ready=all((TASK/f'source/native-balanced/worker-variants/redispatch-runahead{m}/build.json').is_file() for m in range(4))
        if ready and not compiling():break
        time.sleep(3)
    else:raise RuntimeError('Build not finished; inspect exact worker compiler, never relaunch on a wait timeout')
    subprocess.run(['docker','exec',CONTAINER,'bash','--noprofile','--norc',str(TASK/'tools/check_runahead.sh')],check=True)
    from runahead_qualification import main as qualify
    qualify()
    from runahead_screen import main as screen
    screen()
    (TASK/'evidence/redispatch-runahead-stage-v1.json').write_text(json.dumps(dict(complete=True,time=time.time()),indent=2))


if __name__=='__main__':main()
