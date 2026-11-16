"""Cancel only a verified task-owned controller before device admission."""
import argparse
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('tag')
args=parser.parse_args()
if not all(value.isalnum() or value in '-_.' for value in args.tag):parser.error('unsafe tag')
script='''import json,os,pathlib,signal
root=pathlib.Path('/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack')
tag=TAG
out=root/'evidence'/tag
assert (out/'queue.json').exists() and not (out/'admission.json').exists()
assert not (out/'stdout.log').exists()
found=[]
for item in pathlib.Path('/proc').iterdir():
 if not item.name.isdigit():continue
 try:
  command=(item/'cmdline').read_bytes().split(b'\\0')
  if b'tools/controller.py' in command and tag.encode() in command:
   assert (item/'cwd').resolve()==root
   assert not (out/'admission.json').exists()
   os.kill(int(item.name),signal.SIGTERM);found.append(int(item.name))
 except (FileNotFoundError,PermissionError):continue
assert len(found)==1,found
(out/'status.json').write_text(json.dumps(dict(qualified=False,started=False,
 outcome='cancelled_before_admission',reason='Replace unexecuted prototype with qualified FIFO arrival semantics',pids=found),indent=2))
print(found)
'''.replace('TAG',repr(args.tag))
subprocess.run(['ssh','pod6-1-root','python3','-'],input=script.encode(),check=True)
