#!/bin/bash
# BigOpV2（router 手写 + 可插拔融合算子）精度门禁（默认复用一次 torchrun）。
# 对比 BigOpV2 vs BigOp（纯小算子 golden）：
#   - 前向 y（TEST_MODE=forward/both）
#   - 反向 dx/dw1/dw2/gate grad（TEST_MODE=backward/both）
#   - FUSED_MODE=train（默认，MegaMoeBF16Train，fwd+bwd 一体可训练）
#               =forward（MegaMoeBF16.compute，只前向）
#
# 固定多组 shape 矩阵逐组跑。精度 Failed 返回 2；运行 ERROR（例如 kernel
# launch、SHMEM、参数错误）返回 1；WARNING 只报告并返回 0。
# 反向日常硬门禁：finite + cosine>0.99 + 双百（含 BF16 小值绝对误差）。
# 双千未通过显示 WARNING，不单独令脚本退出失败。
# 默认由 run_precision_fast.sh 在同一组 HCCL/SHMEM 进程中跑完整矩阵；
# shape-dependent MegaMoe handle 每个 case 重建，避免不同布局互相污染。
# 排查跨 case 状态污染时，可设置 PRECISION_ISOLATE_CASES=1 恢复旧的逐 case
# torchrun 行为。
# 矩阵按 kernel 路径边界选取（不是随机挑）：
#   16:512:512:16:1     最小合法 shape + topk=1 退化权重 + 天然空 expert
#                       （16 路由 ÷ 16 专家，seed 固定 → 空置模式确定）
#   1000:512:512:8:2    M%256!=0 → 每 expert 行数带 128 行尾块（唯一 tile-tail 维度）
#   128:1024:2048:16:2  原默认 shape，单核排序基准
#   4096:1024:2048:16:2 M*topK=8192 恰跨路由排序单核/多核边界（SORT_API_MAX_ELEM=8160）
#   512:2048:3072:16:2  大 K/N（2048/6144）多 tile
#   256:2048:2048:16:8  topk=EPR=8 → max_recv 打满（dropless 最坏容量）
# 格式 = SEQ:HIDDEN:FFN:E:TOPK（N=2*FFN；须满足 K%256==0、N%512==0、E%EP==0、TOPK<=E）
#
# 用法：
#   bash test/bigop_fused/run_precision.sh [TEST_MODE] [FUSED_MODE]
#     TEST_MODE  = forward|backward|both（默认 both）
#     FUSED_MODE = train|forward（默认 train）
#   8 卡（自动切换 EP8 矩阵，EPR=1/2/4/8 全覆盖）：
#     PRECISION_NPROC=8 bash test/bigop_fused/run_precision.sh
#   单组快速复现（环境变量覆盖矩阵）：
#     PRECISION_CASES="128:1024:2048:16:2" bash test/bigop_fused/run_precision.sh
#   强制逐 case 隔离（排查连续调用影响）：
#     PRECISION_ISOLATE_CASES=1 bash test/bigop_fused/run_precision.sh
#   多机精度门禁（设 NODE_RANK 激活，需 MASTER_IP）：
#     机器 A: NNODES=2 NODE_RANK=0 MASTER_IP=10.0.0.1 PRECISION_NPROC=8 \
#       PRECISION_CASES="4096:7168:3072:64:8" bash test/bigop_fused/run_precision.sh
#     机器 B: NNODES=2 NODE_RANK=1 MASTER_IP=10.0.0.1 PRECISION_NPROC=8 \
#       PRECISION_CASES="4096:7168:3072:64:8" bash test/bigop_fused/run_precision.sh
#   其他可调：PRECISION_NPROC（默认 2）、PRECISION_TIMEOUT（每组超时秒，默认 900）、
#             PRECISION_MASTER_PORT（默认 29517）、ASCEND_DEEPEP_SHMEM_IP_PORT（默认 8899）、
#             MEGAMOE_SAVE_GMM1_OUTPUT（auto|none|bf16|mxfp8，默认 bf16）。
#             MXFP8 反向使用同编解码语义 golden 做硬门禁，并额外报告相对
#             精确 BF16 backward 的 INFO-only compression drift。
#             MEGAMOE_SWIGLU_LIMIT（默认 0）
HERE=$(cd "$(dirname "$0")" && pwd)

# Fast/persistent mode is the normal precision path.  Keep the original body
# below as an explicit diagnostic fallback instead of paying torchrun + HCCL +
# SHMEM initialization for every shape.
# Multi-node (NODE_RANK set) always uses the per-case isolated path below.
if [[ -z "${NODE_RANK:-}" && "${PRECISION_ISOLATE_CASES:-0}" != "1" ]]; then
    exec "${HERE}/run_precision_fast.sh" "$@"
fi

# --- Single/multi-node detection (same pattern as run_speed.sh) ---
NODE_RANK=${NODE_RANK:-}
if [[ -n "$NODE_RANK" ]]; then
    NNODES=${NNODES:-2}
    MASTER_IP=${MASTER_IP:?Multi-node requires MASTER_IP}
    SHMEM_PORT=${SHMEM_PORT:-8899}
    IP_PORT="tcp://${MASTER_IP}:${SHMEM_PORT}"
    TORCHRUN_NODE_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP}"
else
    IP_PORT=${ASCEND_DEEPEP_SHMEM_IP_PORT:-tcp://127.0.0.1:8899}
    TORCHRUN_NODE_ARGS=""
fi

TEST_MODE=${1:-both}
FUSED_MODE=${2:-train}
NPROC=${PRECISION_NPROC:-2}
PRECISION_TIMEOUT=${PRECISION_TIMEOUT:-900}
MASTER_PORT=${PRECISION_MASTER_PORT:-29517}
MEGAMOE_SAVE_GMM1_OUTPUT=${MEGAMOE_SAVE_GMM1_OUTPUT:-bf16}
MEGAMOE_SWIGLU_LIMIT=${MEGAMOE_SWIGLU_LIMIT:-0}

case "$MEGAMOE_SAVE_GMM1_OUTPUT" in
    auto|none|bf16|mxfp8) ;;
    *)
        echo "[error] MEGAMOE_SAVE_GMM1_OUTPUT must be auto, none, bf16 or mxfp8; got '$MEGAMOE_SAVE_GMM1_OUTPUT'" >&2
        exit 2
        ;;
esac

DEFAULT_CASES=(
    "16:512:512:16:1"
    "1000:512:512:8:2"
    "128:1024:2048:16:2"
    "4096:1024:2048:16:2"
    "512:2048:3072:16:2"
    "256:2048:2048:16:8"
)
# EP8 矩阵（PRECISION_NPROC=8 时启用；约束 E%8==0，其余同上）：
#   16:512:512:16:1     最小 shape + topk=1 退化权重 + 天然空 expert（EPR=2）
#   1000:512:512:8:2    M%256!=0 尾块 + EPR=1（每 rank 单专家）+ 跨 rank 路由为主
#   128:1024:2048:16:2  基准 shape（EPR=2）
#   4096:1024:2048:16:2 M*topK=8192 恰跨路由排序单核/多核边界（SORT_API_MAX_ELEM=8160）
#   128:1024:2048:32:4  EPR=4 + topk=EPR=4 → max_recv 容量压力（128*4*8=4096 行）
#   512:2048:3072:64:2  E=64（EPR=8，DeepSeek 风格 EP）+ 大 K/N（2048/6144）多 tile
EP8_CASES=(
    "16:512:512:16:1"
    "1000:512:512:8:2"
    "128:1024:2048:16:2"
    "4096:1024:2048:16:2"
    "128:1024:2048:32:4"
    "512:2048:3072:64:2"
)
if [[ -n "${PRECISION_CASES:-}" ]]; then
    read -ra CASES <<< "${PRECISION_CASES}"
elif [[ "$NPROC" -eq 8 ]]; then
    CASES=("${EP8_CASES[@]}")
else
    CASES=("${DEFAULT_CASES[@]}")
fi

# 自包含：PYTHONPATH 指向 bigop_fused 自身（golden.py / fused.py / timing.py
# 已复制进本目录），cd 到本目录，torchrun 直接用相对路径。
export PYTHONPATH="$HERE:$PYTHONPATH"
export ASCEND_DEEPEP_SHMEM_IP_PORT="$IP_PORT"
export SHMEM_UID_SESSION_ID="${IP_PORT#tcp://}"
export TEST_MODE="$TEST_MODE"
export FUSED_MODE="$FUSED_MODE"
export MEGAMOE_SAVE_GMM1_OUTPUT MEGAMOE_SWIGLU_LIMIT
cd "$HERE"

if ! python -c "import torch_npu; import ascend_deepep; print('[ok] ascend_deepep import OK')"; then
    echo "[error] ascend_deepep import FAILED (see stderr above)." >&2
    exit 1
fi

LOG="run_precision.log"
echo "[run] $(date +%H:%M:%S) BigOpV2 precision gate EP${NPROC} ip=$IP_PORT test=$TEST_MODE "
     "fused=$FUSED_MODE save_gmm1=$MEGAMOE_SAVE_GMM1_OUTPUT "
     "swiglu_limit=$MEGAMOE_SWIGLU_LIMIT cases=${#CASES[@]} "
     "timeout=${PRECISION_TIMEOUT}s" | tee "$LOG"

PASS_CNT=0
WARN_CNT=0
FAIL_CNT=0
ERROR_CNT=0
WARNING_CASES=()
FAILED_CASES=()
ERROR_CASES=()
for i in "${!CASES[@]}"; do
    case_spec="${CASES[$i]}"
    IFS=: read -r seq hidden ffn experts topk <<< "${case_spec}"
    if [[ -z "$seq" || -z "$hidden" || -z "$ffn" || -z "$experts" || -z "$topk" ]]; then
        echo "[case] bad case spec '$case_spec' (want SEQ:HIDDEN:FFN:E:TOPK)" | tee -a "$LOG"
        ERROR_CNT=$((ERROR_CNT + 1))
        ERROR_CASES+=("${case_spec}(bad-spec)")
        continue
    fi
    export BENCH_SEQ="$seq" BENCH_HIDDEN="$hidden" BENCH_FFN="$ffn" \
           BENCH_EXPERTS="$experts" BENCH_TOPK="$topk"
    echo "" | tee -a "$LOG"
    echo "[case $((i + 1))/${#CASES[@]}] seq=$seq hidden=$hidden ffn=$ffn E=$experts "
         "topk=$topk (N=$((2 * ffn)))" | tee -a "$LOG"
    T0=$SECONDS
    FAILED_BEFORE=$(grep -c '== FINAL \[Failed\]' "$LOG" || true)
    WARNING_BEFORE=$(grep -c '== FINAL \[WARNING\]' "$LOG" || true)
    timeout -k 30 "$PRECISION_TIMEOUT" torchrun ${TORCHRUN_NODE_ARGS} \
        --nproc_per_node="$NPROC" \
        --master_port="$MASTER_PORT" test_precision.py 2>&1 | tee -a "$LOG"
    RC=${PIPESTATUS[0]}   # 管道后 $? 是 tee 的退出码，必须取 PIPESTATUS
    ELAPSED=$((SECONDS - T0))
    FAILED_AFTER=$(grep -c '== FINAL \[Failed\]' "$LOG" || true)
    WARNING_AFTER=$(grep -c '== FINAL \[WARNING\]' "$LOG" || true)
    if [[ "$RC" -eq 0 ]]; then
        if [[ "$FAILED_AFTER" -gt "$FAILED_BEFORE" ]]; then
            echo "[case $((i + 1))/${#CASES[@]}] Failed  ${case_spec}  (${ELAPSED}s)" | tee -a "$LOG"
            FAIL_CNT=$((FAIL_CNT + 1))
            FAILED_CASES+=("${case_spec}")
        elif [[ "$WARNING_AFTER" -gt "$WARNING_BEFORE" ]]; then
            echo "[case $((i + 1))/${#CASES[@]}] WARNING  ${case_spec}  (${ELAPSED}s)" | tee -a "$LOG"
            WARN_CNT=$((WARN_CNT + 1))
            WARNING_CASES+=("${case_spec}")
        else
            echo "[case $((i + 1))/${#CASES[@]}] SUCCESS  ${case_spec}  (${ELAPSED}s)" | tee -a "$LOG"
            PASS_CNT=$((PASS_CNT + 1))
        fi
    else
        echo "[case $((i + 1))/${#CASES[@]}] ERROR  ${case_spec}  rc=$RC  (${ELAPSED}s)" | tee -a "$LOG"
        ERROR_CNT=$((ERROR_CNT + 1))
        ERROR_CASES+=("${case_spec}")
    fi
done

echo "" | tee -a "$LOG"
echo "[summary] ${PASS_CNT} SUCCESS / ${WARN_CNT} WARNING / ${FAIL_CNT} Failed / ${ERROR_CNT} ERROR  (test=$TEST_MODE fused=$FUSED_MODE)" | tee -a "$LOG"
if [[ "$ERROR_CNT" -gt 0 ]]; then
    echo "[summary] ERROR cases: ${ERROR_CASES[*]}" | tee -a "$LOG"
    exit 1
fi
if [[ "$FAIL_CNT" -gt 0 ]]; then
    echo "[summary] Failed cases: ${FAILED_CASES[*]}; log: $LOG" | tee -a "$LOG"
    exit 2
elif [[ "$WARN_CNT" -gt 0 ]]; then
    echo "[summary] WARNING cases: ${WARNING_CASES[*]}; diagnostic run completed normally. log: $LOG" | tee -a "$LOG"
else
    echo "[summary] ALL CASES SUCCESS. log: $LOG" | tee -a "$LOG"
fi
