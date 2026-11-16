"""Execute only explicitly supplied scripts on this study's admitted node."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--container',action='store_true');p.add_argument('script',type=Path);a=p.parse_args()
    prefix=ROOT/'evidence'/(a.name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pod5-8-root']
    if a.container:command+=['docker','exec','-i','single-mb-fbw-planner-study','bash','--noprofile','--norc','-s']
    else:command+=['bash','--noprofile','--norc','-s']
    with prefix.with_suffix('.log').open('wb') as f:
        r=subprocess.run(command,input=a.script.read_bytes().replace(b'\r\n',b'\n'),stdout=f,stderr=subprocess.STDOUT)
    receipt=dict(command=command,returncode=r.returncode,log=str(prefix.with_suffix('.log')))
    prefix.with_suffix('.json').write_text(json.dumps(receipt,indent=2));print(receipt)
    print(prefix.with_suffix('.log').read_text(errors='replace')[-2200:])
    raise SystemExit(r.returncode)


if __name__=='__main__':main()
