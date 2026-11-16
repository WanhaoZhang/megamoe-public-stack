import json
import torch
import torch_npu
from fsdp_turbo.utils.device import set_accelerator_compatible
set_accelerator_compatible()
from fsdp_turbo.distributed.expert_parallel.moonep.backend_npu import _AscendImports
imports = _AscendImports()
import npu_patch
print(json.dumps(dict(module=npu_patch.__file__, exports=dir(imports.buffer_c))))
