"""Download the sealed stage once complete, including explicit small TimerV2 NPZs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/z00956592/megaflux/single-mb-fbw-planner-study'


def query(names):
    code='''import json,sys
from pathlib import Path
r=Path(sys.argv[1]);names=json.loads(sys.argv[2]);print(json.dumps({n:json.loads((r/n).read_text()) if (r/n).is_file() else None for n in names}))
'''
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pod5-8-root',
         'python3 -c '+shlex.quote(code)+' '+shlex.quote(REMOTE+'/evidence')+' '+shlex.quote(json.dumps(names))]
    return json.loads(subprocess.run(cmd,capture_output=True,check=True,timeout=35).stdout)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    name='fast-redispatch-stage-v1.json'
    for attempt in range(720):
        state=query([name])[name]
        if state and state.get('complete'):break
        if not a.wait:raise RuntimeError('Stage incomplete')
        if attempt%15==0:print('WAITING_FOR_SEALED_FAST_STAGE',attempt,flush=True)
        time.sleep(20)
    else:raise RuntimeError('Fast stage still incomplete; inspect its exact controller')
    names=[name,'fast-redispatch-capabilities-v1.json','fast-redispatch-qualification-v1.json',
           'fast-redispatch-screen-v1.json','light-timer-qualification-v1.json',
           'fast-redispatch-harness-deployment-v1.json']
    data=query(names)
    for name,value in data.items():
        if value is None:raise RuntimeError('Missing receipt '+name)
        path=ROOT/'evidence'/name
        if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Existing compact evidence differs')
        path.write_text(json.dumps(value,indent=2))
    # Numerical gates are retained too, rather than relying solely on exit status.
    gates=[r['tag'] for r in data['fast-redispatch-qualification-v1.json']]
    screens=[r['tag'] for r in data['fast-redispatch-screen-v1.json']]
    for tag in gates+screens:
        subprocess.run([sys.executable,str(ROOT/'tools/fetch_receipts.py'),tag],check=True)
    for row in data['light-timer-qualification-v1.json']:
        subprocess.run([sys.executable,str(ROOT/'tools/fetch_receipts.py'),'--timer-light',row['tag']],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/analyze_fast_pipeline.py')],check=True)
    print('SEALED_FAST_STAGE_COLLECTED',len(gates),len(screens),flush=True)


if __name__=='__main__':main()
