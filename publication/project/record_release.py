"""Read-only release observation after the bounded local pair."""
from pathlib import Path
import json
import subprocess
import remote

ROOT=Path(__file__).resolve().parent
script=f'''import importlib.util,json,time
spec=importlib.util.spec_from_file_location('local_controller',{(remote.REMOTE+'/stack/tools/controller.py')!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:
 record=dict(timestamp=time.time(),devices_free=True,driver=module.idle(),no_signals_sent=True)
except Exception as error:
 record=dict(timestamp=time.time(),devices_free=False,reason=str(error),no_signals_sent=True)
print(json.dumps(record))
'''
result=subprocess.run(['ssh',remote.HOST,'python3','-'],input=script.encode(),capture_output=True)
result.check_returncode()
value=json.loads(result.stdout)
(ROOT/'node-release.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(value,indent=2))
