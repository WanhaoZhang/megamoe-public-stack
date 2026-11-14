# 原capture复现说明

这些命令针对已授权Ascend950八卡环境，在**全新项目目录和容器**运行。原始capture/SDK/镜像/依赖须先从[索引](ARTIFACTS.md)取得并核对SHA；本目录不提供权重、机器访问方式或凭据。机器、驱动、依赖字节或输入改变后需重新验收，不能承诺逐毫秒复现。

## 1. 准备独立布局并构建

设`REPO`为包含本目录的当前integration仓库，`EXP`为不存在的新实验目录。以下是Linux示例；不要把EXP指向既有实验。

```bash
set -euo pipefail
REPO=/absolute/path/to/ascend_deepep_hkq_wuhan
EXP=/absolute/path/to/new-rho63-reproduction
BUNDLE=$REPO/integration/wuhan/reproductions/rho63-panel-cache-20261031
mkdir "$EXP"
git -C "$REPO" worktree add --detach "$EXP/source" f6daec8a38cebe8d9596af05174edd8570179ca7
git -C "$REPO" worktree add --detach "$EXP/panel-cache-source" 9d036d2b5ffe6b16e25666957124eed3cbf5b407
cp "$BUNDLE/frozen/"* "$EXP/"
```

按实际冻结环境补齐`EXP/dependencies`、`EXP/python-runtime`、两个源码checkout中的CATLASS/SHMEM依赖以及matching PyTorch/CANN。原实验使用独立复制的SDK和runtime、CANN9.2只读挂载，不能混用另一版本SO。输入放在`EXP/inputs/rho6`和`rho3`，各有8个PT与JSON。核对本目录`evidence/FINGERPRINTS.json`中的每个file/tensor指纹。

用本任务自己的新容器，将EXP挂载为`/project`，CANN为`/usr/local/Ascend/cann-9.2.0`，matching Torch路径见冻结`run-device-cache.sh`。不要复用或停止其他任务容器。冻结launcher的网络接口`enp35s0f2`和端口32850/32860对应实测节点；换节点时应只修改新项目里的副本，记录新SHA和编号，重新准入。所有shell在Linux使用LF。

`build-baseline.sh`先跑CPU契约，再构建C1=1的OFF/ON、普通BF16和matching binding；`build-panel-cache.sh`只开启本优化，不叠加retire-fill。在新容器内执行：

```bash
bash /project/build-baseline.sh
bash /project/build-panel-cache.sh
# 基线Host binding/lib、SHMEM/CATCCOS等运行依赖必须与新candidate checkout匹配。
# qualify需要实际libshmem_utils.so及Torch依赖搜索路径：
set +u; source /usr/local/Ascend/cann-9.2.0/set_env.sh; set -u
export PYTHONPATH=/project/python-runtime:/project/source:/project
export LD_LIBRARY_PATH=/project/source/ascend_deepep/lib:/project/dependencies/sdk/memfabric_hybrid/lib:/project/dependencies/sdk/shmem/lib:$LD_LIBRARY_PATH
python3 /project/verify-baseline.py
python3 /project/qualify-panel-cache.py
```

qualification核对源SHA/实际getters并在`selected-variants/rho63-panel-cache-a2-{off,on}`复制候选worker和**基线相同字节**的planner/validator，原构建目录保持冻结。9d新宏默认0；只有候选库为1。不要凭目录名假定库已选择。

## 2. 两次准入与逐进程门禁

以下在宿主root运行，`CONTAINER`必须是本任务的新容器名，已确认其`/project`为EXP。整个批次保留同一任务锁；现有节点占用/健康/fuser失败即停止，不自动换编号重试，不reset或kill外部进程。

```bash
CONTAINER=your-own-new-container
RUN=$EXP/evidence/repro-a1
mkdir -p "$EXP/evidence"
mkdir "$RUN"
# 本shell保留任务锁至批次结束；每个case另有gate_and_run内部锁。
exec 9>"$EXP/.device-campaign.lock"
flock -n 9
export RUN REPO
python3 - <<'PY'
from pathlib import Path
import os,sys,time
sys.path.insert(0,str(Path(os.environ['REPO'])/'integration/wuhan'))
from gate_and_run import gate
out=Path(os.environ['RUN']);boot=Path('/proc/sys/kernel/random/boot_id').read_text()
assert gate(out,'admission1')
time.sleep(120)
assert Path('/proc/sys/kernel/random/boot_id').read_text()==boot
assert gate(out,'admission2')
PY
```

定义每个有限作业的门禁函数。worker收据写`RUN/NAME`，门禁写`RUN/evidence/NAME`，两者都必须通过：

```bash
run_case() {
  local name=$1 variant=$2 mode=$3 timer=$4 rho=$5
  shift 5
  local ref=${1:-}
  if (( $# )); then shift; fi
  python3 "$REPO/integration/wuhan/gate_and_run.py" \
    --project "$RUN" --name "$name" -- \
    docker exec -e EXPERIMENT_VARIANT="$variant" \
      -e EXPERIMENT_ATTEMPT=evidence/repro-a1 \
      -e EXPERIMENT_REFERENCE_ATTEMPT=evidence/repro-a1 \
      -e EXPERIMENT_REFERENCE="$ref" \
      -e EXPERIMENT_PERFORMANCE_REFERENCE="/project/evidence/repro-a1/rho$rho-base-off" \
      "$CONTAINER" timeout --signal=TERM --kill-after=20s 1200s \
      bash /project/run-device-cache.sh "/project/evidence/repro-a1/$name" "$mode" "$timer" "$rho" "$@"
}
```

## 3. 先正确性，再TimerON

先跑独立EP2数学参考、EP4普通/`--checkpoint`两层pool测试。small入口会跑balanced/hot/rotate三次SGD，污染pool权重后验证恢复；base与panel-cache的`updates[*].{before,after,outputs}`必须逐位一致，空owner梯度必须为零。

```bash
run_case ordinary-independent-golden base native-golden off 6
run_case small-base base small off 6
run_case small-base-checkpoint base small off 6 "" --checkpoint
run_case small-cache panel-cache small off 6
run_case small-cache-checkpoint panel-cache small off 6 "" --checkpoint
```

随后每rho运行普通完整参考，再做OFF/ON三次五输出：

```bash
for rho in 6 3; do
  run_case "reference-rho$rho" base reference off "$rho"
  run_case "rho$rho-base-off" base capture off "$rho"
  run_case "rho$rho-base-on" base capture on "$rho" "/project/evidence/repro-a1/rho$rho-base-off"
  run_case "rho$rho-panel-cache-off" panel-cache capture off "$rho" "/project/evidence/repro-a1/rho$rho-base-off"
  run_case "rho$rho-panel-cache-on" panel-cache capture on "$rho" "/project/evidence/repro-a1/rho$rho-panel-cache-off"
done
```

普通参考按pool边界将捕获的概率转为BF16；旧capture的旧数值hash只保留元数据，当前C1必须满足当前普通核参考原判据。候选对C1、ON对OFF和每次重复必须严格逐位相同；不允许放宽阈值。检查每rank JSON、三行iterations JSONL与`ordinary_reference_comparisons`，不能只看gate的eligible。

使用**当前仓库**的导出器（已包含累计AIC_DO_WAIT解码），离线导出每rank96轨道，默认不显示wait：

```bash
for rho in 6 3; do
  for variant in base panel-cache; do
    python3 "$REPO/integration/wuhan/export_timer_trace.py" \
      --input "$RUN/rho$rho-$variant-on" --world-size 8 \
      --out "$EXP/analysis/rho$rho-$variant-no-waits"
    python3 "$REPO/integration/wuhan/export_timer_trace.py" \
      --input "$RUN/rho$rho-$variant-on" --world-size 8 --include-waits \
      --out "$EXP/analysis/rho$rho-$variant-with-waits"
  done
done
```

要求同轨duration overlap0、原始kind覆盖守恒和WGRAD/DGRAD/publish/owner done坐标相同。不要跨rank相减计算通信延迟；累计wait不是完整区间。原始NPY和空隙保留。

## 4. 在线MoonEP的正式TimerOFF

`performance-online.py`保留每次F的真实在线MoonEP；B使用这次F保存的placement。它校验当前已通过的base-OFF收据及输入hash，测完整autograd B、直接FB，包含home refresh/restore/owner/finish。没有计时内hash/Timer/profiler，也没有将F和B相加。

```bash
for rho in 6 3; do
  i=0
  for variant in base panel-cache panel-cache base base panel-cache panel-cache base; do
    printf -v name 'rho%d-%02d-%s' "$rho" "$i" "$variant"
    run_case "$name" "$variant" perf off "$rho"
    i=$((i+1))
  done
done
```

每个进程warm10/measure30；保存全部八rank原始样本、计时前后五输出hash、实际SO身份、前后门禁。用实际已核验库SHA重算（下面历史SHA仅适用于历史SO，重编译应填新SHA）：

```bash
python3 "$BUNDLE/summarize.py" --root "$RUN" \
  --baseline-worker-sha 593f7c7bfa34c216976c70942b2f82af51935d1a8c03e406a889c938ad6d63f6 \
  --candidate-worker-sha 2d52c337a7e14bf931e687779eb389a4595bb4c197543d3fb436e290b08e7c39 \
  --out "$EXP/summary-new.json"
```

该CPU工具独立重读16组×8rank×30样本和门禁，计算四配对收益、P90、控制波动及保留条件。本次已对历史原始八rank收据重跑并匹配报告；它不代替数学参考或NPU执行。

## 5. 正常退休与回退

每个case失败即停止队列，保留编号；修正后另起新编号和新目录，不能覆盖失败结果。批次结束后复查健康/进程/root-fuser，确认本任务所有device进程正常退出、容器仅剩本任务服务后再停止自己的容器。保持原输入/库/原始图冻结。

回退用原C1库或重建`MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=0`。9d的代码已通过固定两capture验收；本目录复制的是实测device/build脚本，文档编排与CPU汇总/解码检查不冒充另一轮全新设备复现。
