from pathlib import Path
import ctypes,hashlib,json,os,runpy,sys
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p/'source/integration/wuhan'))
from early_dw1_entry import verify
d=Path(os.environ['WORKER_NATIVE']);record=verify(d,1)
variant=os.environ.get('EXPERIMENT_VARIANT','base');assert variant in ('base','panel-cache')
lib=ctypes.CDLL(str(d/'libascend_deepep_megaflux_early_bf16.so'))
getter=getattr(lib,'megaflux_restore_panel_acquire_cache',None)
actual=int(getter()) if getter else 0;assert actual==int(variant=='panel-cache')
retire=getattr(lib,'megaflux_backward_dw1_retire_fill',None);assert not retire or int(retire())==0
mode='on' if int(lib.megaflux_early_bf16_timer_enabled()) else 'off'
base=p/'source/worker-variants'/('rho63-base-a2-'+mode)
for name in ('libascend_deepep_moonep_planner.so','libascend_deepep_plan_validator.so'):
 assert hashlib.sha256((d/name).read_bytes()).digest()==hashlib.sha256((base/name).read_bytes()).digest()
print('SCHEDULE_SELECTION',json.dumps(record|dict(variant=variant,panel_acquire_cache=actual,timer=mode,common_controls_byte_identical=True)),flush=True)
entry=Path(sys.argv[1]).resolve();sys.argv=[str(entry),*sys.argv[2:]]
runpy.run_path(str(entry),run_name='__main__')
