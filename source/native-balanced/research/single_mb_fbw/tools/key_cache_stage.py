"""Gate source-metadata optimization after the preceding pipeline experiment."""
import json
import subprocess
import time
from controller import TASK,CONTAINER
from remaining_causal_stages import compiling


def main():
    prior=json.loads((TASK/'evidence/redispatch-runahead-stage-v1.json').read_text())
    if not prior.get('complete'):raise RuntimeError('Runahead queue has not retired')
    # These are numerical comparisons, not timing scores. Their original
    # qualified worker is immutable while the isolated cache worker compiles.
    from forced_subset_gates import main as subset_gates
    subset_gates()
    for _ in range(1200):
        ready=all((TASK/f'source/native-balanced/worker-variants/gradient-key-cache{m}/build.json').is_file() for m in (0,1))
        if ready and not compiling():break
        time.sleep(3)
    else:raise RuntimeError('Key cache build not finished; inspect exact compilation before any relaunch')
    subprocess.run(['docker','exec',CONTAINER,'bash','--noprofile','--norc',str(TASK/'tools/check_key_cache.sh')],check=True)
    from key_cache_qualification import main as qualify
    qualify()
    from key_cache_screen import main as screen
    screen()
    (TASK/'evidence/gradient-key-cache-stage-v1.json').write_text(json.dumps(dict(complete=True,time=time.time()),indent=2))


if __name__=='__main__':main()
