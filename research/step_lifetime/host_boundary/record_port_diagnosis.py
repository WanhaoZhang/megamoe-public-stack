"""Read-only evidence for the pre-worker rendezvous failures."""
from pathlib import Path
import json
import subprocess
import remote

ROOT=Path(__file__).resolve().parent
script='''import json,pathlib,subprocess,time
lines=subprocess.check_output(['ss','-Htanp'],text=True).splitlines()
selected=[line[:360] for line in lines if ':40740 ' in line or ':33856 ' in line]
print(json.dumps(dict(timestamp=time.time(),ephemeral_range=pathlib.Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().strip(),
 selected_socket_rows=selected,read_only=True,no_signals_sent=True)))
'''
result=subprocess.run(['ssh',remote.HOST,'python3','-'],input=script,text=True,capture_output=True)
result.check_returncode()
parsed=json.loads(result.stdout)
(ROOT/'diagnostics/port-collision-evidence.json').write_text(json.dumps(parsed,indent=2)+'\n',newline='\n')
print(json.dumps(parsed,indent=2))
