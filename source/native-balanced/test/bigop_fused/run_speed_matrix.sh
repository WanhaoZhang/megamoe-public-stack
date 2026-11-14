#!/bin/bash
# run_speed 矩阵扫描 + 出图：多组 shape/配置批量跑性能，聚合 JSON ->
# summary.csv + speed_matrix.png（shape/配置敏感性分析）。
#
# 用法（单机）：
#   NPROC=8 bash test/bigop_fused/run_speed_matrix.sh [warmup] [iters]
#
# 用法（多机；两台机器同时执行相同命令）：
#   Machine A: NNODES=2 NODE_RANK=0 MASTER_IP=10.0.0.1 NPROC=8 \
#     bash test/bigop_fused/run_speed_matrix.sh 3 10
#   Machine B: NNODES=2 NODE_RANK=1 MASTER_IP=10.0.0.1 NPROC=8 \
#     bash test/bigop_fused/run_speed_matrix.sh 3 10
#
# 矩阵覆盖（格式 SEQ:HIDDEN:FFN:E:TOPK，空格分隔多组）：
#   SPEED_CASES="4096:7168:3072:64:8 4096:7168:4096:64:8" \
#     bash test/bigop_fused/run_speed_matrix.sh 3 10
#
# 环境变量：
#   NPROC             每节点卡数（默认 2）
#   NNODES            节点数（默认 2，仅多机时生效）
#   NODE_RANK         当前节点编号（0..N-1，设了就启用多机模式）
#   MASTER_IP         多机必填，主节点 IP
#   SPEED_MASTER_PORT torchrun 端口（默认 29521）
#   SPEED_MATRIX_STAMP 多机同步用时间戳（所有节点必须相同；单机自动生成）
#   BIGOP_TIMING / MEGAMOE_TRACE / BENCH_LOCAL_MEM_BYTES 透传给每组
#
# 默认矩阵（同规格 K=7168 topK=8，2 F × 2 E，seq 固定 4096）：
#   EP8：F 3072/4096、E 64(EPR8)/128(EPR16)
#   EP4：F 3072/4096、E 32(EPR8)/64(EPR16)
#   EP2：同 EP8 shape（EPR 翻倍）
# 输出：test/bigop_fused/speed_matrix/<时间戳>/
#   case_XX_<spec>.json   每组结构化结果（test_speed.py BENCH_JSON_OUT）
#   run_matrix.log        全程日志
#   summary.csv           全部配置 x 指标扁平表（plot_speed_matrix.py 生成）
#   speed_matrix.png      6 联对比图（无 matplotlib 时降级 ASCII，CSV 照出）
HERE=$(cd "$(dirname "$0")" && pwd)
NPROC=${NPROC:-${SPEED_NPROC:-2}}
WARMUP=${1:-3}
ITERS=${2:-10}
TIMEOUT=${SPEED_TIMEOUT:-1800}
MASTER_PORT=${SPEED_MASTER_PORT:-29521}

# --- 多机检测（与 run_speed.sh 一致）---
NODE_RANK=${NODE_RANK:-}
if [[ -n "$NODE_RANK" ]]; then
    NNODES=${NNODES:-2}
    MASTER_IP=${MASTER_IP:?Multi-node requires MASTER_IP}
    IP_PORT="tcp://${MASTER_IP}:${SHMEM_PORT:-8899}"
    TORCHRUN_NODE_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP}"
    WORLD_SIZE=$((NPROC * NNODES))
    MODE="multi-node nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_IP}"
else
    IP_PORT=${ASCEND_DEEPEP_SHMEM_IP_PORT:-tcp://127.0.0.1:${SHMEM_PORT:-8899}}
    TORCHRUN_NODE_ARGS=""
    WORLD_SIZE=$NPROC
    MODE="single-node"
fi

EP8_CASES=(
    "4096:7168:3072:64:8"
    "4096:7168:3072:128:8"
    "4096:7168:4096:64:8"
    "4096:7168:4096:128:8"
    "1024:7168:3072:64:8"
    "1024:7168:3072:128:8"
    "1024:7168:4096:64:8"
    "1024:7168:4096:128:8"
)
EP2_CASES=(
    "4096:7168:3072:64:8"
    "4096:7168:3072:128:8"
    "4096:7168:4096:64:8"
    "4096:7168:4096:128:8"
)
# EP4（K=7168 topK=8，EPR=E/4）：2 F × 2 E（E=128 → EPR=32 超 routing limit，改用 E=32）
EP4_CASES=(
    "4096:7168:3072:32:8"
    "4096:7168:3072:64:8"
    "4096:7168:4096:32:8"
    "4096:7168:4096:64:8"
)
# EP32（4机×8卡=32卡，EPR=E/32；E=256→EPR=8, E=512→EPR=16）
EP32_CASES=(
    "4096:7168:3072:256:8"
    "4096:7168:3072:512:8"
    "4096:7168:4096:256:8"
    "4096:7168:4096:512:8"
)
if [[ -n "${SPEED_CASES:-}" ]]; then
    read -ra CASES <<< "${SPEED_CASES}"
elif [[ "$WORLD_SIZE" -eq 32 ]]; then
    CASES=("${EP32_CASES[@]}")
elif [[ "$WORLD_SIZE" -eq 8 ]]; then
    CASES=("${EP8_CASES[@]}")
elif [[ "$WORLD_SIZE" -eq 4 ]]; then
    CASES=("${EP4_CASES[@]}")
else
    CASES=("${EP2_CASES[@]}")
fi

export PYTHONPATH="$HERE:$PYTHONPATH"
export ASCEND_DEEPEP_SHMEM_IP_PORT="$IP_PORT"
export SHMEM_UID_SESSION_ID="${IP_PORT#tcp://}"
# 默认矩阵含 seq=8192 / topK=16 两组（对称窗口约 1.8GiB），默认窗口提到 2GiB；
# 自定义矩阵窗口需求小可自行覆盖（test_speed.py 启动时仍会打印实际估算并校验）
export BENCH_LOCAL_MEM_BYTES=${BENCH_LOCAL_MEM_BYTES:-2147483648}
cd "$HERE"

if ! python -c "import torch_npu; import ascend_deepep; print('[ok] ascend_deepep import OK')"; then
    echo "[error] ascend_deepep import FAILED (see stderr above)." >&2
    exit 1
fi

# 多机时用环境变量同步时间戳（所有节点必须相同），单机自动生成
STAMP=${SPEED_MATRIX_STAMP:-$(date +%Y%m%d_%H%M%S)}
export SPEED_MATRIX_STAMP="$STAMP"  # 子进程继承，多机手动传入时保持一致
OUT_DIR="$HERE/speed_matrix/$STAMP"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_matrix.log"
echo "[run] $(date +%H:%M:%S) speed matrix [${MODE}] EP${WORLD_SIZE} cases=${#CASES[@]} warmup=$WARMUP iters=$ITERS ip=$IP_PORT" | tee "$LOG"
echo "[run] out_dir=$OUT_DIR" | tee -a "$LOG"

OK_CNT=0
FAIL_CNT=0
for i in "${!CASES[@]}"; do
    spec="${CASES[$i]}"
    IFS=: read -r seq hidden ffn experts topk <<< "$spec"
    if [[ -z "$seq" || -z "$hidden" || -z "$ffn" || -z "$experts" || -z "$topk" ]]; then
        echo "[case] bad spec '$spec' (want SEQ:HIDDEN:FFN:E:TOPK)" | tee -a "$LOG"
        FAIL_CNT=$((FAIL_CNT + 1))
        continue
    fi
    json_out="$OUT_DIR/case_$(printf '%02d' "$i")_${spec//:/_}.json"
    echo "" | tee -a "$LOG"
    echo "[case $((i + 1))/${#CASES[@]}] $spec -> $(basename "$json_out")" | tee -a "$LOG"
    T0=$SECONDS
    BENCH_SEQS="$seq" BENCH_HIDDEN="$hidden" BENCH_FFN="$ffn" \
    BENCH_EXPERTS="$experts" BENCH_TOPK="$topk" BENCH_WARMUP="$WARMUP" \
    BENCH_ITERS="$ITERS" BENCH_JSON_OUT="$json_out" BENCH_OUT_DIR="$OUT_DIR" \
    timeout -k 30 "$TIMEOUT" torchrun ${TORCHRUN_NODE_ARGS} \
        --nproc_per_node="$NPROC" \
        --master_port="$MASTER_PORT" test_speed.py 2>&1 | tee -a "$LOG"
    RC=${PIPESTATUS[0]}   # 管道后 $? 是 tee 的退出码，必须取 PIPESTATUS
    ELAPSED=$((SECONDS - T0))
    if [[ "$RC" -eq 0 ]]; then
        echo "[case $((i + 1))/${#CASES[@]}] OK   $spec (${ELAPSED}s)" | tee -a "$LOG"
        OK_CNT=$((OK_CNT + 1))
    else
        echo "[case $((i + 1))/${#CASES[@]}] FAIL $spec rc=$RC (${ELAPSED}s)" | tee -a "$LOG"
        FAIL_CNT=$((FAIL_CNT + 1))
    fi
done

echo "" | tee -a "$LOG"
echo "[summary] ${OK_CNT} OK / ${FAIL_CNT} FAIL  (EP${WORLD_SIZE} [${MODE}] warmup=$WARMUP iters=$ITERS)" | tee -a "$LOG"

# 聚合出图（仅 node 0 执行；多机时其他节点跳过）
if [[ -z "$NODE_RANK" || "$NODE_RANK" == "0" ]]; then
    if ls "$OUT_DIR"/case_*.json >/dev/null 2>&1; then
        python "$HERE/plot_speed_matrix.py" --dir "$OUT_DIR" 2>&1 | tee -a "$LOG" \
            || echo "[warn] plot_speed_matrix.py 失败，可稍后手动重跑（CSV/PNG 未生成）" | tee -a "$LOG"
    else
        echo "[warn] 无 case_*.json 产出，跳过聚合出图" | tee -a "$LOG"
    fi
fi
echo "[summary] out: $OUT_DIR" | tee -a "$LOG"
[[ "$FAIL_CNT" -eq 0 ]]
