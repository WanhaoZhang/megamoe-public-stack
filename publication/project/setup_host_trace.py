"""Create a sibling diagnostic lane without touching the running confirmation source."""
from pathlib import Path
import json
import shutil
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent/'host-boundary-20261116'
ROOT.mkdir(exist_ok=True)
STACK = ROOT/'stack'
if STACK.exists():
    # Resume only the just-created partial long-path checkout; never overwrite edits.
    origin = subprocess.check_output(['git','-C',str(STACK),'remote','get-url','origin'],text=True).strip()
    status = subprocess.check_output(['git','-C',str(STACK),'status','--porcelain'],text=True).splitlines()
    index = subprocess.check_output(['git','-C',str(STACK),'ls-files','-z'])
    head = subprocess.check_output(['git','-C',str(STACK),'rev-parse','HEAD'])
    expected = subprocess.check_output(['git','-C',str(BASE/'stack'),'rev-parse','HEAD'])
    if Path(origin).resolve() != (BASE/'stack').resolve() or index or head != expected or any(not row.startswith(('D  ', '?? ')) for row in status):
        raise SystemExit('Existing sibling is not the fresh partial checkout; inspect it.')
    subprocess.run(['git','-C',str(STACK),'config','core.longpaths','true'],check=True)
    subprocess.run(['git','-C',str(STACK),'config','core.autocrlf','false'],check=True)
    # Failed initial clone never wrote its index. Recreate it in this verified fresh
    # task-owned checkout; this is not a reset of any existing research workspace.
    subprocess.run(['git','-C',str(STACK),'read-tree','--reset','-u','HEAD'],check=True)
else:
    subprocess.run(['git','-c','core.longpaths=true','clone','--no-hardlinks',str(BASE/'stack'),str(STACK)],check=True)
for args in (['config','core.longpaths','true'],['config','core.autocrlf','false'],
             ['switch','-c','research/step-lifetime-host-boundary'],
             ['remote','rename','origin','frozen-base'],
             ['remote','set-url','--push','frozen-base','no-push']):
    subprocess.run(['git','-C',str(STACK),*args],check=True)
shutil.copy2(BASE/'stack/configs/lifetime-runtime-freeze.json',STACK/'configs/lifetime-runtime-freeze.json')
for directory in ('evidence','diagnostics','implementation','figures'):
    (ROOT/directory).mkdir(exist_ok=True)
remote = (BASE/'remote.py').read_text().replace('step-lifetime-priority-cache-20261116','step-lifetime-host-boundary-20261116')
remote = remote.replace('base=/home/z00956592/megaflux/step-lifetime-raw-control-20261116/stack',
                        'base=/home/z00956592/megaflux/step-lifetime-priority-cache-20261116/stack')
remote = remote.replace("controller = re.sub(r\"^CONTAINER = .*$\", f\"CONTAINER = '{CONTAINER}'\", controller, flags=re.M)",
                        "controller = re.sub(r\"^CONTAINER = .*$\", f\"CONTAINER = '{CONTAINER}'\", controller, flags=re.M)\n    controller = controller.replace('29200 + int(time.time()) % 2000', '32200 + int(time.time()) % 2000')")
(ROOT/'remote.py').write_text(remote,newline='\n')
plan = '''# Host-only boundary diagnosis

Independent sibling research/step-lifetime-host-boundary, based on b77bb06 and the explicitly frozen priority-cache worker e2868fd2. Old source, SOs and completed runs remain immutable. Same common node admission lock; isolated container/install/artifacts/port range.

Question: do the old rich-capture 188–206ms cross-rank backward entry gaps persist after warmup with the TimerOFF worker and no new device Events/snapshots/synchronizations? Measure host calls before any diagnostic resource creation; do not infer device completion from host return.

Implement explicit --host-boundary-trace, off by default. Buffer CPU timestamp records, dump after training only. Trace existing pool/operator, planner/publish, FSDP, optimizer, lifecycle/drain/P2P submission and Python GC boundaries. No tensor clone, CPU read or new device synchronization in the tracer.

Gate: small L2 H512 F512 S256 GA4 W2, three complete Adam updates and full step0 per-parameter gradient comparison against retained qualified reference. Then bounded main-shape B1/B3 controls (11 updates, inspect warmed updates5–10); no traced wall time enters performance figures. Actual worker, guarded source, initial/input/route identity, finite updates, exclusive admission and90GiB budget must all pass.

Do not implement a new gradient policy solely from pre-warmup rich-capture waits. Interpret a disappearing gap as a difference between measurement/run conditions, not proof of one specific Event-limit cause. This lane is diagnostic infrastructure, not a new planner.
'''
(ROOT/'PLAN.md').write_text(plan,newline='\n')
(ROOT/'setup-provenance.json').write_text(json.dumps(dict(base_head=subprocess.check_output(['git','-C',str(BASE/'stack'),'rev-parse','HEAD'],text=True).strip(),
    copied_freeze='configs/lifetime-runtime-freeze.json',isolated_branch='research/step-lifetime-host-boundary'),indent=2)+'\n',newline='\n')
print(ROOT)
