"""Audit the exact full-model structure on meta without allocating NPU memory."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
from mindspeed_llm.fsdp2.utils.arguments import ModelArguments, DataArguments, ParallelArguments, TrainingArguments, OptimizationArguments
from mindspeed_llm.fsdp2.utils.global_vars import set_args
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from mindspeed_llm.fsdp2.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
root = Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
flat = {}
for part in (ModelArguments(model_name_or_path=str(root / 'configs/model')), DataArguments(),
             ParallelArguments(), TrainingArguments(output_dir=str(root / 'evidence')), OptimizationArguments()):
    flat.update(asdict(part))
flat.update(use_fused_rmsnorm=True, use_fused_rotary_pos_emb=True, use_sparse_flash_attn=True,
            use_fused_lightning_indexer=True, use_fused_lightning_indexer_loss=True,
            use_fused_compressor=True, use_triton_swiglu_limit=False, use_fused_clipped_swiglu=True, use_ascend_mhc=False,
            fix_router=False, moe_grouped_gemm=True, indexer_loss_coeff=1., router_aux_loss_coef=0.)
set_args(SimpleNamespace(**flat))
values = json.loads((root / 'configs/model/config.json').read_text())
config = DeepseekV4Config(**values)
config.mlp_layer_types = ['moe'] * config.num_hidden_layers
with torch.device('meta'):
    model = DeepseekV4ForCausalLM(config)
result = dict(total_parameters=sum(p.numel() for p in model.parameters()),
              routed_parameters=sum(p.numel() for n,p in model.named_parameters() if '.experts.' in n),
              layers=len(model.model.layers), mtp=model.mtp is not None,
              experts=[dict(layer=i, gate_up=list(layer.mlp.experts.gate_up_proj.shape),
                            down=list(layer.mlp.experts.down_proj.shape),
                            shared=list(layer.mlp.shared_experts.up_proj.weight.shape))
                       for i,layer in enumerate(model.model.layers)])
print(json.dumps(result, indent=2))
(root / 'evidence/final-meta-model.json').write_text(json.dumps(result, indent=2))
