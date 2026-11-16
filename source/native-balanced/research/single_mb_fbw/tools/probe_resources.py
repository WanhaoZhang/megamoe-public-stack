"""Read-only task-local server probes; no containers or processes are changed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--jump');p.add_argument('hosts',nargs='+');a=p.parse_args()
    forbidden={'pod6-1','pod6-1-root'}
    if any(h in forbidden for h in a.hosts):raise ValueError('Other experiment node excluded')
    def probe(host):
        cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ConnectionAttempts=1']
        if a.jump:cmd+=['-J',a.jump]
        cmd += [host,'hostname; npu-smi info; pgrep -af raylet']
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
            return dict(host=host,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,command=cmd)
        except subprocess.TimeoutExpired:
            return dict(host=host,status='timeout',command=cmd)
    with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(probe,a.hosts))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'resource-scans';out.mkdir(exist_ok=True)
    (out/f'{stamp}.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
