#!/usr/bin/env bash
# 双机 16 卡 backward 精度测试（2 节点 × 8 卡，EP=16）。
#
# 前置条件（两台机器都要）：
#   1. 同一 git commit 检出 + 同源编译（bash scripts/build_mega_moe_bf16.sh）
#   2. 互通：互 ping 通过，29500/8899 端口防火墙放行
#   3. 独占：npu-smi info 确认 8 卡空闲
#
# 用法：
#   机器 A（主节点，node rank 0，IP 假设 10.0.0.1）：
#     NODE_RANK=0 bash test/megamoe_bf16_test/run_backward_dual_node.sh 10.0.0.1
#   机器 B（node rank 1）：
#     NODE_RANK=1 bash test/megamoe_bf16_test/run_backward_dual_node.sh 10.0.0.1
#
#   两台几乎同时启动（torchrun 会互相等待组网，先起的一台会等另一台）。
#   MASTER_IP 是机器 A 的实体 IP；NODE_RANK 区分节点（0/1）。
#
# 可调参数（环境变量）：
#   LOCAL_DEVICES   每节点用的卡号（默认 0,1,2,3,4,5,6,7）
#   BENCH_M/K/N     shape（默认 4096/7168/6144，即 K=7168 F=3072）
#   EPR             每 rank 专家数（默认 4 → E=64 总专家，EP=16）
#   TOPK            默认 8
#   SHMEM_PORT      默认 8899
#   MASTER_PORT     默认 29500
#   SKIP_REF=1      跳过 CPU 参考计算（大 shape 双机启动时省 10-30 分钟，
#                   kernel fwd+bwd 照跑，只是不做梯度对比）
#   EXTRA_ARGS      透传给 test_mega_moe_backward.py 的额外参数
#                   （如 --check-intermediates --routing-mode random）
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MASTER_IP=${1:?用法: NODE_RANK=<0|1> bash $0 <主节点IP>}
NODE_RANK=${NODE_RANK:?必须设置 NODE_RANK=0（主节点）或 1（从节点）}
LOCAL_DEVICES=${LOCAL_DEVICES:-0,1,2,3,4,5,6,7}
BENCH_M=${BENCH_M:-4096}
BENCH_K=${BENCH_K:-7168}
BENCH_N=${BENCH_N:-6144}
EPR=${EPR:-4}
TOPK=${TOPK:-8}
SHMEM_PORT=${SHMEM_PORT:-8899}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
SKIP_REF=${SKIP_REF:-1}
if [[ "${SKIP_REF}" == "1" ]]; then
    SKIP_REF_FLAG="--skip-reference"
else
    SKIP_REF_FLAG=""
fi

SHMEM_IP_PORT="tcp://${MASTER_IP}:${SHMEM_PORT}"
export ASCEND_DEEPEP_SHMEM_IP_PORT="${SHMEM_IP_PORT}"
export SHMEM_UID_SESSION_ID="${MASTER_IP}:${SHMEM_PORT}"

echo "[dual-node] node_rank=${NODE_RANK} master=${MASTER_IP} \\
devices=${LOCAL_DEVICES} M=${BENCH_M} K=${BENCH_K} N=${BENCH_N} \\
EPR=${EPR} topK=${TOPK} EP=$((NNODES * NPROC_PER_NODE)) \\
shmem=${SHMEM_IP_PORT} extra=${EXTRA_ARGS:-}"

# torchrun is the standard launcher; the -- separator stops argparse from
# matching our script's short flags (--m/--n) as prefixes of torchrun's own
# options (--master_addr/--nnodes). The script-side aliases
# --seq/--hidden/--gmm-n are the torchrun-safe names.
exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_IP}" \
    --master_port="${MASTER_PORT}" \
    -- \
    "${HERE}/test_mega_moe_backward.py" \
    "${LOCAL_DEVICES}" \
    --seq "${BENCH_M}" --hidden "${BENCH_K}" --gmm-n "${BENCH_N}" \
    --topk "${TOPK}" --expert-per-rank "${EPR}" \
    --shmem-ip-port "${SHMEM_IP_PORT}" \
    --watchdog-secs 300 \
    ${SKIP_REF_FLAG} \
    ${EXTRA_ARGS:-}
