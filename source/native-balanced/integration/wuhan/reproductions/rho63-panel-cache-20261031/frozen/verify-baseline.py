from pathlib import Path
import ctypes,hashlib,json
import torch,torch_npu
p=Path(__file__).resolve().parent;r=p/'source'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records={}
for mode in ('off','on'):
 d=r/'worker-variants'/('rho63-base-a2-'+mode);record=json.loads((d/'timer-capture-build.json').read_text())
 assert record['backward_replicated_dw1_early']==1
 for name,h in record['sources'].items():assert sha(r/name)==h,name
 for name,h in record['libraries'].items():assert sha(d/name)==h,name
 lib=ctypes.CDLL(str(d/'libascend_deepep_megaflux_early_bf16.so'))
 for getter,expected in [('megaflux_backward_replicated_dw1_early',1),('megaflux_worker_pipeline_flags',0),('megaflux_backward_restore_mode',2),('megaflux_restore_operand_ready',1),('megaflux_worker_owner_rows',16),('megaflux_worker_owner_compact',0),('megaflux_early_bf16_timer_enabled',int(mode=='on'))]:
  assert int(getattr(lib,getter)())==expected,(mode,getter)
 records[mode]=record
import ascend_deepep._C as binding
ordinary=r/'ascend_deepep/lib/libascend_deepep_mega_moe_bf16.so'
assert ctypes.CDLL(str(ordinary)).mega_moe_bf16_timer_enabled()==0
assert Path(binding.__file__).resolve().parent==r/'ascend_deepep'
result=dict(pass_=True,source_commit='f6daec8a38cebe8d9596af05174edd8570179ca7',variants=records,binding_sha256=sha(Path(binding.__file__)),ordinary_sha256=sha(ordinary))
(p/'baseline-build-verification.json').write_text(json.dumps(result,indent=2));print('BASELINE_OFF_ON_BINDING_VERIFIED',flush=True)
