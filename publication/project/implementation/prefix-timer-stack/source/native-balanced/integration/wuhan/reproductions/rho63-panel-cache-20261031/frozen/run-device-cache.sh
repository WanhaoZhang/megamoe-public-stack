#!/usr/bin/env bash
set -euo pipefail
OUT=${1:?};MODE=${2:?};TIMER=${3:-off};RHO=${4:-3};shift 4
set +u;source /usr/local/Ascend/cann-9.2.0/set_env.sh;set -u
export PYTHONPATH=/project/python-runtime:/project/source:/project
export EXPERIMENT_VARIANT=${EXPERIMENT_VARIANT:-base}
if [[ "$EXPERIMENT_VARIANT" == base ]];then
 export WORKER_NATIVE=/project/source/worker-variants/rho63-base-a2-$TIMER
elif [[ "$EXPERIMENT_VARIANT" == candidate ]];then
 export WORKER_NATIVE=/project/selected-variants/rho63-retire-fill-a2-$TIMER
elif [[ "$EXPERIMENT_VARIANT" == panel-cache ]];then
 export WORKER_NATIVE=/project/selected-variants/rho63-panel-cache-a2-$TIMER
else exit 2;fi
export EXPECTED_EARLY_DW1=1 WUHAN_PUBLISH_SYNC=device WUHAN_CONTROL_TAPE_DEVICE=npu WUHAN_POOL_AUDIT=0
export MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS=all
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TASK_QUEUE_ENABLE=2
export HCCL_EXEC_TIMEOUT=240 HCCL_CONNECT_TIMEOUT=180
export MEGAFLUX_SHMEM_PORT=32860
export GLOO_SOCKET_IFNAME=enp35s0f2 HCCL_SOCKET_IFNAME=enp35s0f2
export LD_LIBRARY_PATH=$WORKER_NATIVE:/project/source/ascend_deepep/lib:/project/dependencies/sdk/memfabric_hybrid/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib:/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
unset WUHAN_REPLAY_CAPTURE_DIR ASCEND_LAUNCH_BLOCKING
mkdir -p "$OUT/plog";export ASCEND_PROCESS_LOG_PATH=$OUT/plog
N=8;ENTRY=/project/source/integration/wuhan/runtime_bootstrap.py
case "$MODE" in
 verify) python3 /project/verify-baseline.py;exit;;
 native-golden)
  N=2;args=(/project/source/test/megamoe_bf16_test/test_mega_moe_backward.py 0,1 --seq 129 --hidden 512 --gmm-n 1024 --topk 2 --expert-per-rank 2 --routing-mode balanced --save-gmm1-output bf16 --swiglu-limit 1.5 --check-intermediates --shmem-ip-port tcp://127.0.0.1:32860)
  ;;
 small)
  N=4;export WUHAN_POOL_AUDIT=1
  args=(/project/bridge_entry.py --out "$OUT" --planner moonep --restore panel --tokens 257 --ffn 256 --experts 8 --slots 2 --require-replicas --expect-operand-ready 1 --expect-delay-cycles 0 --limit 10 "$@")
  ;;
 reference) args=(/project/ordinary-reference.py --capture /project/inputs/rho$RHO --out "$OUT");;
 capture)
  export EXPERIMENT_RHO=$RHO EXPERIMENT_ATTEMPT=${EXPERIMENT_ATTEMPT:-evidence/baseline-node73-a1}
  t=0;ref=();if [[ "$TIMER" == on ]];then t=1;fi
  if [[ -n "${EXPERIMENT_REFERENCE:-}" ]];then ref=(--reference "$EXPERIMENT_REFERENCE");
  elif [[ "$TIMER" == on ]];then ref=(--reference /project/$EXPERIMENT_ATTEMPT/rho$RHO-base-off);fi
  args=(/project/capture-current.py --out "$OUT" --timer "$t" --capture /project/inputs/rho$RHO --repeat 3 --shmem-port 32860 --save-outputs "${ref[@]}")
  ;;
 perf)
  args=(/project/performance-online.py --out "$OUT" --capture /project/inputs/rho$RHO --restore panel --warmup 10 --repeat 30 --phases backward fb)
  ;;
 *)exit 2;;
esac
torchrun --nnodes=1 --nproc-per-node="$N" --master-port=32850 "$ENTRY" --native-only /project/panel-cache-selection-entry.py "${args[@]}"
