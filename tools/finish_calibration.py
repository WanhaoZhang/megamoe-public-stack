"""CPU-only reduction, acceptance and primitive fitting after all29 jobs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
jobs=json.loads((root/'evidence/calibration-jobs.json').read_text())
assert len(jobs)==29 and all(x['qualified'] for x in jobs)
tags=[x['tag'] for x in jobs if '--diagnostic-timer' in json.loads((root/f'evidence/{x["tag"]}/diagnostic-request.json').read_text())['arguments']]
assert len(tags)==8
arguments=[value for tag in tags for value in ('--tag',tag)]
commands=[['reduce_links.py'],['reduce_interface.py'],['reduce_diagnostics.py',*arguments],['reduce_framework.py',*arguments]]
def run(command):
    filename=root/'evidence'/('calibration-reduce-'+command[0]+'.log')
    with filename.open('w') as stream:
        result=subprocess.run([sys.executable,str(root/'tools'/command[0]),*command[1:]],stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps(dict(tool=command[0],exit=result.returncode,log=str(filename))),flush=True)
    return result.returncode
with ThreadPoolExecutor(max_workers=4) as executor:results=list(executor.map(run,commands))
assert results==[0]*4,'inspect reducer logs before accepting calibration'
subprocess.run([sys.executable,str(root/'tools/accept_calibration.py')],check=True)
subprocess.run([sys.executable,str(root/'tools/fit_primitives.py')],check=True)
print('CALIBRATION_REDUCED_ACCEPTED_AND_PRIMITIVES_FIT',flush=True)
