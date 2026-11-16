"""Host-side own-project lease and external NPU-process watchdog.
Usage: python3 host_guard.py --container NAME -- COMMAND ...
It never kills another container or foreign process. It stops only NAME.
"""
import argparse,subprocess,fcntl,time,json,re,threading,os,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=argparse.ArgumentParser();P.add_argument('--container',required=True);P.add_argument('cmd',nargs=argparse.REMAINDER);args=P.parse_args()
out=ROOT/'receipts';out.mkdir(exist_ok=True)
lock=open('/tmp/single-mb-fbw-planner-study-8npu.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def state():
 r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15,check=True)
 # Process rows have decimal NPU ID then PID. Device summary second field is name.
 pids={int(m.group(1)) for m in re.finditer(r'^\|\s*\d+\s*\|\s*(\d+)\s*\|',r.stdout,re.M)}
 return r.stdout,pids
s,pids=state()
if pids:raise SystemExit('External NPU users at admission: '+str(pids))
receipt={'time':time.time(),'npu_smi':s,'container':args.container,'pid':os.getpid(),'cmd':args.cmd}
name=str(time.time_ns());(out/('admit-'+name+'.json')).write_text(json.dumps(receipt,indent=2))
done=threading.Event()
container_id=subprocess.check_output(['docker','inspect','--format','{{.Id}}',args.container],text=True).strip()
history={}
def start_time(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 except (FileNotFoundError,ProcessLookupError):return None
def own_processes():
 own=subprocess.check_output(['docker','top',args.container,'-eo','pid'],text=True,timeout=10)
 pids={int(x) for x in own.split() if x.isdigit()}
 for pid in pids:
  stamp=start_time(pid)
  if stamp is not None:history[pid]=stamp
 return pids
def watch():
 samples=0
 while not done.wait(3):
  try:
   before=own_processes()
   s,pids=state()
   after=own_processes();unknown=pids-(before|after);foreign=set();stale=[]
   samples+=1
   with (out/('monitor-'+name+'.jsonl')).open('a') as f:
    f.write(json.dumps({'time':time.time(),'npu_pids':sorted(pids),'own_pids':sorted(after),'npu_smi':s if samples%10==0 else None})+'\n')
   for pid in unknown:
    stamp=start_time(pid)
    if stamp is None:stale.append(pid);continue
    cgroup=Path(f'/proc/{pid}/cgroup').read_text()
    if container_id in cgroup or history.get(pid)==stamp:continue
    foreign.add(pid)
   if stale:
    with (out/('stale-driver-pids-'+name+'.jsonl')).open('a') as f:f.write(json.dumps({'time':time.time(),'exited_pids':stale})+'\n')
   if foreign:
    (out/('foreign-stop-'+name+'.json')).write_text(json.dumps({'time':time.time(),'foreign':sorted(foreign),'before':sorted(before),'after':sorted(after),'npu_smi':s},indent=2))
    subprocess.run(['docker','stop','--time','10',args.container],timeout=25)
    return
  except Exception as exc:
   if done.is_set():return
   (out/('watch-error-'+name+'.txt')).write_text(str(exc))
   subprocess.run(['docker','stop','--time','10',args.container],timeout=25)
   return
threading.Thread(target=watch,daemon=True).start()
cmd=args.cmd[1:] if args.cmd[:1]==['--'] else args.cmd
try:code=subprocess.call(cmd)
finally:done.set()
(out/('exit-'+name+'.json')).write_text(json.dumps({'time':time.time(),'returncode':code},indent=2))
raise SystemExit(code)
