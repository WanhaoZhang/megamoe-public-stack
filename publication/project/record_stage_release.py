"""Record actual node ownership after this stage; never signal any process."""
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parent
controller='/home/z00956592/megaflux/step-lifetime-host-boundary-20261116/stack/tools/controller.py'
script=f'''import importlib.util,json,time
spec=importlib.util.spec_from_file_location('stage_controller',{controller!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:
 sample=module.idle();record=dict(timestamp=time.time(),devices_free=True,driver=sample,no_signals_sent=True)
except Exception as error:
 record=dict(timestamp=time.time(),devices_free=False,reason=str(error),no_signals_sent=True)
print(json.dumps(record))
'''
result=subprocess.run(['ssh','pod6-1-root','python3','-'],input=script.encode(),capture_output=True)
result.check_returncode()
record=json.loads(result.stdout)
(ROOT/'NODE-RELEASE-20261116.json').write_text(json.dumps(record,indent=2)+'\n',newline='\n')
print(json.dumps(record,indent=2))
