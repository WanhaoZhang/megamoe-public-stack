"""Check exact native artifacts without opening a distributed training job."""
import ctypes
import os
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
import ascend_deepep
from ascend_deepep.runtime_limits import require_runtime_capacity
require_runtime_capacity()
worker=ctypes.CDLL(str(Path(os.environ["WORKER_NATIVE"])/"libascend_deepep_megaflux_early_bf16.so"))
settings={}
for name in ["megaflux_max_local_experts","megaflux_early_bf16_timer_enabled","megaflux_worker_pipeline_flags","megaflux_worker_owner_rows","megaflux_backward_restore_mode","megaflux_restore_operand_ready","megaflux_backward_replicated_dw1_early"]:
    function=getattr(worker,name);function.restype=ctypes.c_int;settings[name]=int(function())
assert settings["megaflux_early_bf16_timer_enabled"]==0,settings
assert settings["megaflux_worker_pipeline_flags"]==0,settings
print(json.dumps({"actual_worker_settings":settings}),flush=True)
print(json.dumps({'module':ascend_deepep.__file__,'limits':list(ascend_deepep._C.large_ep_limits())}),flush=True)
for line in Path('/proc/self/maps').read_text().splitlines():
    if any(name in line for name in ('libshmem.so','libprofapi.so','libprofimpl.so')):
        print(line.split()[-1],flush=True)
