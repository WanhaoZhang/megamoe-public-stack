"""Publish local Python changes to the isolated, idle online-prototype lane."""
from pathlib import Path
import subprocess
import tarfile

root=Path(__file__).resolve().parent
stack=root/'stack'
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115'
container='temporal-replica-frontier-20261115'
controller=(stack/'tools/controller.py').read_text().replace(
    "'/home/z00956592/megaflux/planner-executor-ep8-lab'",repr(remote+'/stack')).replace(
    "CONTAINER = 'planner-executor-ep8-lab'",f'CONTAINER = {container!r}').replace(
    "(TASK / 'evidence' / 'device-jobs.lock')",
    "pathlib.Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock')")
path=root/'controller.py'
path.write_text(controller,encoding='utf-8',newline='\n')
archive=root/'frontier-overlay.tar'
with tarfile.open(archive,'w') as handle:
    handle.add(path,arcname='tools/controller.py')
    for path in (stack/'tools').glob('*.py'):
        if path.name!='controller.py':handle.add(path,arcname='tools/'+path.name)
    launcher=(stack/'tools/launch.sh').read_text().replace(
        'TASK=/home/z00956592/megaflux/planner-executor-ep8-lab',f'TASK={remote}/stack')
    path=root/'launch.sh'
    path.write_text(launcher,encoding='utf-8',newline='\n')
    handle.add(path,arcname='tools/launch.sh')
    for name in ('source/native-balanced/CMakeLists.txt',
                 'configs/temporal-gates.json','configs/temporal-study.json',
                 'configs/temporal-scheduler-freeze.json',
                 'source/native-balanced/kernels/mega_moe_bf16/mega_moe_temporal_scheduler.cpp'):
        handle.add(stack/name,arcname=name)
    timer_manifest=stack/'configs/temporal-worker-timer-freeze.json'
    if timer_manifest.exists():handle.add(timer_manifest,arcname='configs/temporal-worker-timer-freeze.json')
subprocess.run(['scp',str(archive),f'pod6-1-root:{remote}/frontier-overlay.tar'],check=True)
subprocess.run(['ssh','pod6-1-root','tar','-xf',remote+'/frontier-overlay.tar','-C',remote+'/stack'],check=True)
