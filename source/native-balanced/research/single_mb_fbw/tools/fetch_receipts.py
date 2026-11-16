"""Fetch top-level evidence; bulky profiler/plog trees stay in-project remotely."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/z00956592/megaflux/single-mb-fbw-planner-study'


def main():
    p=argparse.ArgumentParser();p.add_argument('--timer-light',action='store_true');p.add_argument('tags',nargs='+');a=p.parse_args()
    for tag in a.tags:
        if not re.fullmatch('[a-zA-Z0-9_.-]+',tag):raise ValueError('Invalid run name')
        code="""from pathlib import Path
import sys,tarfile
p=Path(sys.argv[1]);assert (p/'status.json').is_file(),p
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for f in sorted(p.iterdir()):
  if f.is_file() and (f.suffix in ('.json','.jsonl','.npy') or f.name=='stdout.log'):
   t.add(f,arcname=f.name,recursive=False)
 if len(sys.argv)>2 and sys.argv[2]=='timer-light':
  for f in sorted((p/'timer-light').glob('rank*-*.npz')):
   if f.is_file() and not f.is_symlink():t.add(f,arcname='timer-light/'+f.name,recursive=False)
"""
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pod5-8-root',
                 'python3 -c '+shlex.quote(code)+' '+shlex.quote(REMOTE+'/evidence/'+tag)+(' timer-light' if a.timer_light else '')]
        response=subprocess.run(command,capture_output=True,check=True)
        destination=ROOT/'evidence'/tag;destination.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(response.stdout),mode='r:gz') as t:
            members=t.getmembers()
            def allowed(name):
                return (Path(name).name==name or (a.timer_light and
                    re.fullmatch(r'timer-light/rank[0-7]-[0-9]+-(forward|backward)\.npz',name)))
            if any(not f.isfile() or not allowed(f.name) for f in members):
                raise ValueError('Unexpected archive path/type')
            for f in members:
                payload=t.extractfile(f).read();path=destination/f.name;path.parent.mkdir(exist_ok=True)
                if path.exists() and path.read_bytes()!=payload:
                    raise ValueError(f'Existing local evidence differs: {path}')
                path.write_bytes(payload)
        receipt=dict(tag=tag,remote_project=REMOTE,top_level_only=not a.timer_light,
            archive_sha256=hashlib.sha256(response.stdout).hexdigest(),
            files={f.name:hashlib.sha256((destination/f.name).read_bytes()).hexdigest() for f in members})
        (destination/'fetch.json').write_text(json.dumps(receipt,indent=2))
        print(tag,len(members),flush=True)


if __name__=='__main__':main()
