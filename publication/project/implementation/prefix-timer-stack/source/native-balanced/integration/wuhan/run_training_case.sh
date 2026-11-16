#!/usr/bin/env bash
# One explicit, frozen 8P case. Host gate_and_run.py owns device gates/exit.
set -euo pipefail
ROOT=${1:?composite source root}
OUT=${2:?new absolute evidence directory}
METHOD=${3:?A B C or D}
RHO=${4:?0 3 or 6}
ACTIVATION=${5:?recompute or uam}
STEPS=${6:?explicit training step count}
LAYERS=${7:-8}
# All defaults retain the frozen single-node comparison. EP groups remain
# contiguous inside each eight-device node; FSDP may span nodes.
NNODES=${WUHAN_NNODES:-1}
NODE_RANK=${WUHAN_NODE_RANK:-0}
EP_SIZE=${WUHAN_EP_SIZE:-4}
EXPERT_FSDP=${WUHAN_EXPERT_FSDP:-2}
LOCAL_WORLD_SIZE=${WUHAN_LOCAL_WORLD_SIZE:-8}
WORLD=$((NNODES * LOCAL_WORLD_SIZE))
MASTER_ADDR=${WUHAN_MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${WUHAN_MASTER_PORT:-29871}
python "$(dirname "$0")/validate_launch.py"
[[ "$METHOD" =~ ^[ABCD]$ && "$RHO" =~ ^(0|3|6)$ ]]
[[ "$ACTIVATION" =~ ^(recompute|uam)$ && "$STEPS" =~ ^[1-9][0-9]*$ ]]
[[ "$LAYERS" =~ ^[1-9][0-9]*$ ]]
test -f /project/assets/model/config.json
test -f /project/assets/data/alpaca_text_document.bin
test -f /project/assets/data/alpaca_text_document.idx
mkdir -p "$OUT"
archive_shmem_logs() {
    local run_rc=$?
    mkdir -p "$OUT/shmem_log"
    if [[ -d /root/shmem/log ]]; then
        cp -a /root/shmem/log/. "$OUT/shmem_log/" || true
    else
        printf '%s\n' 'No /root/shmem/log directory in this runtime.' > "$OUT/shmem_log/NOT_PRESENT"
    fi
    printf '%s\n' "$run_rc" > "$OUT/training.exit"
}
trap archive_shmem_logs EXIT
set +u
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -u
export PYTHONPATH="/project/python-runtime:$ROOT/ascend_deepep_hkq:$ROOT/FSDPTurbo:$ROOT/MindSpeed-LLM:/project/control-source-a2/MoonEP:${PYTHONPATH:-}"
VARIANT=${WUHAN_WORKER_VARIANT:-s9-ported}
[[ "$VARIANT" =~ ^[a-z0-9-]+$ ]]
export WORKER_NATIVE="$ROOT/ascend_deepep_hkq/worker-variants/$VARIANT"
export LD_LIBRARY_PATH="$WORKER_NATIVE:$ROOT/ascend_deepep_hkq/ascend_deepep/lib:${LD_LIBRARY_PATH:-}"
export TRAINING_BACKEND=mindspeed_fsdp
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp35s0f2}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp35s0f2}
export HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=240
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1 OMP_NUM_THREADS=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_PROCESS_LOG_PATH="$OUT/plog"
export WUHAN_STEP_RECORD_DIR="$OUT/steps"
if [[ "${WUHAN_CAPTURE_REPLAY:-0}" == 1 ]]; then
    export WUHAN_REPLAY_CAPTURE_DIR="$OUT/layer-capture"
else
    unset WUHAN_REPLAY_CAPTURE_DIR
fi
export ASCEND_ROUTER_DISTRIBUTION=rank_skew
export ASCEND_ROUTER_NUM_HOT_EXPERTS=2 ASCEND_ROUTER_REBALANCE_EP=0
export ASCEND_ROUTER_COUNT_ALIGN=1 ASCEND_EP_SIZE="$EP_SIZE"
export ASCEND_ROUTER_IMBALANCE="$RHO"
export ASCEND_DEEPEP_FIXED_ROUTER_SPLIT_CACHE=0
export MEGAMOE_SAVE_GMM1_OUTPUT=bf16
export ASCEND_DEEPEP_MEGAMOE_REDISPATCH=0
export ASCEND_DEEPEP_MEGAMOE_REDISPATCH_OVERLAP=0
export ASCEND_DEEPEP_MEGAMOE_REDISPATCH_VERIFY=0
export ASCEND_DEEPEP_TOKEN_CHUNK_M=999999
export ASCEND_DEEPEP_SHMEM_WINDOW_POOL=2
export ASCEND_LAUNCH_BLOCKING=0 ASCEND_DEEPEP_FWD_SYNC=0
export ASCEND_DEEPEP_BWD_SYNC=0 ASCEND_DEEPEP_BWD_STREAM_SYNC=0
export ASCEND_DEEPEP_FWD_PRE_STREAM_SYNC=0
# EFS2 Hybrid initializes after shard; all four controls must use that path.
# Setting this to 1 only changes A/B/C in the author engine and breaks pairing.
export SCRATCH_INIT_BEFORE_SHARD=0 OOM_SNAPSHOT_ENABLE=0
export FSDP_TURBO_UAM_LAST_LAYER_RESIDENT=0
export FSDP_TURBO_UAM_MHC_CPU_ANCHORS=every16
export FSDP_TURBO_UAM_MHC_ANCHOR_BACKEND=cpu
export FSDP_TURBO_UAM_PRODUCER_PREFETCH_LAYERS=1
export UAM_VALIDATE_FINITE=1 UAM_GRAD_DIAG_ACTIVE=0 UAM_SYNC_BEFORE_CLIP=0
export UAM_LOG_FWD_BWD_TIME=0
export UAM_GRAD_DIAG_STEPS= UAM_CPU_SUBMIT_TIMING=0
unset SHMEM_UID_SESSION_ID
extra=()
# Local per-node data mounts must each build GPT sample indices. The author
# shared-filesystem path remains the default; opt in for isolated test nodes.
case "${WUHAN_DATA_NO_SHARED_STORAGE:-0}" in
    0) ;;
    1) extra+=(--data.no_shared_storage True) ;;
    *) echo 'WUHAN_DATA_NO_SHARED_STORAGE must be 0 or 1' >&2; exit 2 ;;
esac
case "$METHOD" in
    A) dispatcher=mega_moe ;;
    B) dispatcher=megamoe_balanced
       extra+=(--optimization.megamoe_balanced_planner native --optimization.megamoe_balanced_replica_slots "${WUHAN_NATIVE_SLOTS:-4}") ;;
    C) dispatcher=megamoe_balanced
       extra+=(--optimization.megamoe_balanced_planner moonep --optimization.megamoe_balanced_replica_slots 2) ;;
    D) dispatcher=moonep_megamoe ;;
esac
restore=${BALANCED_RESTORE_MODE:-staged}
if [[ "$METHOD" == B || "$METHOD" == C ]]; then
    extra+=(--optimization.megamoe_balanced_backward_restore "$restore")
    if [[ -n "${WUHAN_RECV_CAPACITY:-}" ]]; then
        extra+=(--optimization.megamoe_balanced_recv_capacity "$WUHAN_RECV_CAPACITY")
    fi
fi
if [[ "$ACTIVATION" == recompute ]]; then
    extra+=(--parallel.recompute True --parallel.recompute_modules 'model.layers.{*}' 'mtp.layers.{*}.layer'
            --training.unified_activation_memory_policy "")
else
    extra+=(--parallel.recompute False --training.unified_activation_memory_policy
            "$ROOT/MindSpeed-LLM/examples/fsdp2/deepseek4/uam_config_megamoe_tuned.yaml")
fi
cd "$ROOT/MindSpeed-LLM"
command=(python -m torch.distributed.run --nproc-per-node="$LOCAL_WORLD_SIZE" --nnodes="$NNODES" --node-rank="$NODE_RANK"
    --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT"
    "$ROOT/ascend_deepep_hkq/integration/wuhan/runtime_bootstrap.py" "$PWD/train_fsdp2.py"
    examples/fsdp2/deepseek4/pretrain_deepseek4_flash_284b_4k_fsdp2_A3.yaml
    --model.model_name_or_path /project/assets/model --model.train_from_scratch True
    --model.n_routed_experts 8 --model.num_hidden_layers "$LAYERS" --model.num_experts_per_tok 2
    --data.dataset '{"file_name": "/project/assets/data/alpaca_text_document"}' --data.cutoff_len 8192
    --parallel.fsdp_size "$WORLD" --parallel.ep_size "$EP_SIZE" --parallel.ep_fsdp_size "$EXPERT_FSDP"
    --parallel.ep_dispatcher "$dispatcher"
    --training.per_device_train_batch_size 1 --training.gradient_accumulation_steps 1
    --training.output_dir "$OUT/output" --training.max_steps "$STEPS"
    --training.save_steps 0 --training.save_epochs 0 --training.save_hf_weights False
    --training.seed 42 --training.logging_steps 1 --training.profile False
    --optimization.fix_router True --optimization.router_imbalance "$RHO"
    --optimization.use_fused_rmsnorm True --optimization.use_fused_rotary_pos_emb True
    --optimization.use_sparse_flash_attn True --optimization.use_fused_lightning_indexer True
    --optimization.use_fused_lightning_indexer_loss True --optimization.use_ascend_mhc False
    --optimization.use_triton_swiglu_limit True --optimization.chunk_loss_size 1024
    --optimization.use_fused_compressor True "${extra[@]}")
python - "$OUT/launch.json" "$METHOD" "$RHO" "$ACTIVATION" "$STEPS" "$LAYERS" "$restore" "$VARIANT" "${command[@]}" <<'PY'
import hashlib, json, os, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent) if '__file__' in globals() else '.')
sys.path.insert(0, str(Path(os.environ['WORKER_NATIVE']).parents[1]/'integration/wuhan'))
from distributed_config import LaunchTopology
path, method, rho, activation, steps, layers, restore, variant, *command = sys.argv[1:]
native = Path(os.environ['WORKER_NATIVE'])
topology = LaunchTopology.from_env(os.environ)
Path(path).write_text(json.dumps(dict(method=method, rho=int(rho), activation=activation,
    steps=int(steps), layers=int(layers), experts=8, topk=2, sequence=8192,
    ep=topology.ep, expert_fsdp=topology.expert_fsdp, world_size=topology.world_size,
    topology=asdict(topology), restore=restore if method in ('B', 'C') else None,
    diagnostic_capture=os.environ.get('WUHAN_CAPTURE_REPLAY', '0') == '1',
    publish_sync=os.environ.get('WUHAN_PUBLISH_SYNC', 'device') if method in ('B', 'C') else None,
    control_tape_device=os.environ.get('WUHAN_CONTROL_TAPE_DEVICE', 'npu') if method in ('B', 'C') else None,
    data_no_shared_storage=os.environ.get('WUHAN_DATA_NO_SHARED_STORAGE', '0') == '1',
    native_slots=int(os.environ.get('WUHAN_NATIVE_SLOTS', '4')) if method == 'B' else None,
    recv_capacity_override=int(os.environ['WUHAN_RECV_CAPACITY']) if method in ('B', 'C') and os.environ.get('WUHAN_RECV_CAPACITY') else None,
    worker_variant=variant if method in ('B', 'C') else None,
    worker_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(native.glob('*.so'))} if method in ('B', 'C') else {},
    command=command), indent=2))
PY
"${command[@]}"

# Timer harnesses leave these CPU dumps. Export only after successful training;
# ordinary/Timer-OFF runs never enter this branch or import a trace exporter.
if [ -d "$OUT/named-kernel-timer" ]; then
    python3 "$ROOT/ascend_deepep_hkq/integration/wuhan/export_timer_trace.py" \
        --input "$OUT/named-kernel-timer" --out "$OUT/kernel-trace" \
        --world-size "$(( ${WUHAN_NNODES:-1} * ${WUHAN_LOCAL_WORLD_SIZE:-8} ))"
fi
