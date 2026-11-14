"""Verify both new worker builds and freeze baseline planner/validator controls."""
from pathlib import Path
import ctypes,hashlib,json,shutil
import torch,torch_npu
p=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records={}
for mode in ('off','on'):
 build=p/'panel-cache-source/worker-variants'/('rho63-panel-cache-a2-'+mode)
 base=p/'source/worker-variants'/('rho63-base-a2-'+mode)
 record=json.loads((build/'timer-capture-build.json').read_text())
 for name,h in record['sources'].items():assert sha(p/'panel-cache-source'/name)==h,name
 for name,h in record['libraries'].items():assert sha(build/name)==h,name
 assert record['backward_replicated_dw1_early']==record['restore_panel_acquire_cache']==1
 lib=ctypes.CDLL(str(build/'libascend_deepep_megaflux_early_bf16.so'))
 assert lib.megaflux_restore_panel_acquire_cache()==lib.megaflux_backward_replicated_dw1_early()==1
 assert lib.megaflux_early_bf16_timer_enabled()==int(mode=='on')
 assert lib.megaflux_worker_pipeline_flags()==0 and lib.megaflux_restore_operand_ready()==1
 assert lib.megaflux_restore_delay_cycles()==0
 dest=p/'selected-variants'/('rho63-panel-cache-a2-'+mode);dest.mkdir(parents=True,exist_ok=False)
 worker='libascend_deepep_megaflux_early_bf16.so';shutil.copy2(build/worker,dest/worker)
 for name in ('libascend_deepep_moonep_planner.so','libascend_deepep_plan_validator.so'):
  shutil.copy2(base/name,dest/name)
  assert sha(dest/name)==json.loads((base/'timer-capture-build.json').read_text())['libraries'][name]
 record['libraries']={f.name:sha(f) for f in dest.glob('*.so')}
 record['runtime_selection_provenance']=dict(worker_build=str(build),control_build=str(base),planner_policy_changed=False,binding_changed=False,source_root='panel-cache-source')
 (dest/'timer-capture-build.json').write_text(json.dumps(record,indent=2))
 original=json.loads((build/'build.json').read_text());assert original['restore_panel_acquire_cache']==1
 (dest/'build.json').write_text(json.dumps(original,indent=2));records[mode]=record
(p/'panel-cache-qualification.json').write_text(json.dumps(dict(pass_=True,device_execution=False,variants=records),indent=2))
print('PANEL_CACHE_GETTERS_AND_COMMON_CONTROLS_VERIFIED',flush=True)
