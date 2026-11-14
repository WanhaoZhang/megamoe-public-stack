#!/usr/bin/env python3
"""Host-side EP8 launch gate. Fail closed; never reset devices or stop other jobs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def inspect_smi(output, allow_warning_alarm=False):
    cards = re.findall(r'^\|\s*(\d+)\s*\|\s*(Ascend950\S*)\s*\|\s*(\S+)\s*\|', output, re.M)
    empty = re.findall(r'No running processes found in NPU (\d+)\b', output)
    util = re.findall(r'^\|\s*\|\s*\|\s*NA\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*\d+', output, re.M)
    return (sorted(int(c[0]) for c in cards) == list(range(8)) and
            all(c[2] in (('OK','Warning','Alarm') if allow_warning_alarm else ('OK',)) for c in cards) and
            sorted(map(int, empty)) == list(range(8)) and len(util) == 8 and
            all(int(u) == 0 and int(mem) < 8192 for u,mem in util))


def check_devices(out, label, allow_warning_alarm=False):
    result = subprocess.run(['npu-smi','info'], capture_output=True, text=True, timeout=30)
    (out/(label+'-npu-smi.txt')).write_text(result.stdout+result.stderr)
    holders=[];logs=[];valid=True
    for manager,paths in ((False,['/dev/davinci'+str(i) for i in range(8)]),
                          (True,['/dev/davinci_manager'])):
        fuser=subprocess.run(['fuser',*paths],capture_output=True,text=True,timeout=30)
        valid=valid and fuser.returncode in (0,1)
        logs.append(fuser.stdout+fuser.stderr)
        for pid in re.findall(r'\b\d+\b',fuser.stdout):
            proc=Path('/proc',pid)
            try:
                comm=(proc/'comm').read_text().strip()
                exe=Path(os.readlink(proc/'exe')).name
            except FileNotFoundError:
                continue  # The querying process already exited.
            except OSError:
                comm=exe='unreadable'
            # npu-smi monitoring opens only the manager; never ignore a holder
            # of an actual compute device or an unidentified manager client.
            observer=manager and comm=='npu-smi' and exe=='npu-smi'
            holders.append(dict(pid=int(pid),manager=manager,comm=comm,exe=exe,observer=observer))
    (out/(label+'-fuser.txt')).write_text('\n'.join(logs))
    (out/(label+'-holders.json')).write_text(json.dumps(holders,indent=2)+'\n')
    return (result.returncode==0 and inspect_smi(result.stdout,allow_warning_alarm) and
            valid and all(h['observer'] for h in holders))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    from launch_config import add_arguments, shell_command, validate_output_name
    add_arguments(p)
    p.add_argument('--out', required=True, help='new relative run directory')
    p.add_argument('--gate-only', action='store_true')
    from build_identity import VARIANTS
    p.add_argument('--worker-variant', choices=VARIANTS, default='planner-bench')
    from routing_variants import VARIANTS as ROUTING_VARIANTS
    p.add_argument('--routing-variant',choices=ROUTING_VARIANTS)
    p.add_argument('runner_args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    root=Path(a.host_root).resolve(); mount=Path(a.mount_root).resolve()
    root.relative_to(mount)
    validate_output_name(a.out)
    x=json.loads(subprocess.check_output(['docker','inspect',a.container]))[0]
    if (x['Id'] != a.container_id or not x['State']['Running'] or
        x['Config'].get('Labels',{}).get('codex.task') != a.task_label or
        [(m['Source'],m['Destination']) for m in x['Mounts'] if m['RW']] != [(str(mount),a.mount_target)] or
        Path(a.mount_target,root.relative_to(mount)).as_posix() != a.container_root):
        raise ValueError('container identity/mount mismatch')
    out=root/'test/megamoe_planner_test/output'/a.out
    out.mkdir(parents=True,exist_ok=False)
    # Serializes our launches only. External ownership is always checked again.
    import fcntl
    with (mount/'planner-benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        eligible=check_devices(out,'pre1',a.allow_warning_alarm)
        if eligible:
            time.sleep(2)
            eligible=check_devices(out,'pre2',a.allow_warning_alarm)
        record=dict(eligible=eligible,run_exit=None,post_gate=False,
                    state='NOT_STARTED',allow_warning_alarm=a.allow_warning_alarm,
                    utc=datetime.now(timezone.utc).isoformat())
        (out/'acceptance.json').write_text(json.dumps(record,indent=2)+'\n')
        if not eligible or a.gate_only:
            print(json.dumps(dict(**record,path=str(out))))
            return 0 if eligible else 3
        argv=a.runner_args
        if argv[:1]==['--']:argv=argv[1:]
        # Output, dependency roots and endpoint belong to this launcher.
        if any(v.split('=')[0] in ('--out','--fsdp-root','--shmem-endpoint','--worker-variant','--routing-variant') for v in argv):
            raise ValueError('runner overrides a protected launch argument')
        inside=a.container_root
        inside_out=inside+'/test/megamoe_planner_test/output/'+a.out
        env=dict(PLANNER_BENCH_GATED_RUN=a.out,
                 FSDP_ROOT=a.fsdp_root,
                 MOONEP_ROOT=a.moonep_root,
                 SHMEM_HOME_PATH=a.shmem_root,
                 WORKER_NATIVE=inside+'/worker-variants/'+a.worker_variant)
        if a.routing_variant:
            env['PLANNER_NATIVE']=inside+'/planner-variants/'+ROUTING_VARIANTS[a.routing_variant]['planner_directory']
            argv=['--routing-variant',a.routing_variant,*argv]
        command=shell_command(a, [
            'env',*[k+'='+v for k,v in env.items()], 'bash',
            inside+'/test/megamoe_planner_test/run_planner_megamoe.sh',inside_out,
            '--worker-variant',a.worker_variant,*argv])
        cmd=['docker','exec',a.container,'bash','-lc',command]
        (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        with (out/'launcher.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        time.sleep(3)
        post=check_devices(out,'post',a.allow_warning_alarm)
        record.update(run_exit=result.returncode,post_gate=post,
                      state='COMPLETED' if result.returncode==0 and post else 'FAILED')
        (out/'acceptance.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(**record,path=str(out))))
        return 0 if record['state']=='COMPLETED' else 1


if __name__=='__main__':raise SystemExit(main())
