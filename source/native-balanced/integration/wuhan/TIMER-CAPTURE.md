> 当前计划校验默认融合（保留兼容参数），统一实现及旧框架迁移见 [FUSED-PLAN-DEFAULT.md](FUSED-PLAN-DEFAULT.md)。

# 完整 AscendTimerV2 打点：从构建到两张图

本入口将原算子 TimerV2、schema6 source/restore 坐标打点、MoonEP planner 计时、配对五输出校验和 JSON 导出串起来。所有采集脚本均在本仓库；不需要历史实验目录里的 prepare 脚本、旧 PID 或特定容器名。

## 环境与准备

当前目标是 Ascend950/A5 单机2/4/8卡；详细source分区打点仅支持EP2/4/8。默认复现八卡、两个EP4组（拓扑记为EP4×expert-FSDP2）。入口执行在线planner和完整专家前反向，不执行真实FSDP参数all-gather/reduce-scatter、Attention或optimizer。不要将分组或算子回放当成完整训练吞吐验收。

依赖：Linux、Bash、匹配的CANN/Bisheng、torch/torch_npu、NumPy、CANN SHMEM Python包，以及本仓库已编译的 `_C` 和基础BF16依赖库。验证环境为CANN9.2.0、Python3.12、torch/torch_npu2.10；版本支持范围以实际构建结果为准。独立采集不依赖FSDPTurbo、MindSpeed或外部MoonEP的npu_patch模块。

首次拉取要初始化依赖，并按[根README](../../README.md)安装基础binding、BF16库；已有本仓库环境可以直接使用：

```bash
git checkout codex/moonflux-8p-integration
git pull --ff-only
git submodule update --init --recursive
# 先 source 本机对应版本的 CANN set_env.sh。
# SHMEM_HOME_PATH 指向实际SDK根；其下应有 shmem/include 和 shmem/src。
export SHMEM_HOME_PATH=/absolute/path/to/matching-shmem-sdk
```

安装环境的 `shmem` Python包和 `_C` 必须使用兼容的同一套SHMEM运行时。本入口在加载 `_C` 前显式提升已选中的Python SHMEM库，并硬检查进程只有一个libshmem映像；不要通过复制多份动态库来绕过检查。

## 第一步：构建独立 TimerOFF/ON 配对库

在仓库根目录执行，prefix每次使用新名字，已存在的目录不会覆盖：

```bash
MAX_JOBS=2 bash scripts/build_megaflux_timer_capture.sh timer-rho6-a1
```

产物：

- `worker-variants/timer-rho6-a1-off/`
- `worker-variants/timer-rho6-a1-on/`

两目录各有独立worker SO、planner SO及 `timer-capture-build.json`。固定S9/flags0、owner16、panel restore、operand-ready1、direct阈值0、其他候选关闭；区别为Timer开关。普通生产默认仍关闭planner计时，保留同事的direct/fused/大EP功能。

不要把TimerON planner复制到生产目录。其计时需要扩展header；Python显式 `timer=True` 分配足够空间，误配会报错。单独CMake构建需同时启用 `ASCEND_DEEPEP_ENABLE_TIMER=ON` 和 `MEGAFLUX_MOONEP_TIMER=ON`；后者默认OFF。

## 第二步：没有真实捕获文件时，直接跑同 shape 合成负载

先确认设备空闲、健康；网络环境需要时设置本机实际 `HCCL_SOCKET_IFNAME`。执行：

```bash
REPO=$(pwd)
bash integration/wuhan/run_timer_capture.sh \
  --variants "$REPO/worker-variants/timer-rho6-a1" \
  --out /absolute/path/to/experiments/timer-rho6-a1 \
  --world-size 8 --ep 4 \
  --tokens 8192 --hidden 7168 --ffn 3072 \
  --experts 8 --topk 2 --slots 2 --capacity 20480 \
  --rho 6 --hot-experts 2 --limit 10 --repeat 3
```

默认参数就是上述配置：每卡X/dY `[8192,7168]`、IDs/prob `[8192,2]`、W1 `[2,6144,7168]`、W2 `[2,7168,3072]`；W1为gate/up合并。X/dY/W1/W2为BF16，prob为FP32传入，算子按原bridge转换；保存BF16中间量。

合成router在EP rank0的前两个专家集中负载，`rho = hot-card-load / cold-card-load - 1` 为舍入前目标。各token的TopK专家互异，配额规则和实际分配写进回执。它复现shape与受控偏斜，不承诺与之前框架fixed router生成的每个token顺序相同。

入口先运行TimerOFF，再以同样seed/输入运行TimerON。从第1次到第repeat次均检查五输出；TimerON逐位对齐TimerOFF。合成负载的检查是有限值、重复性和观察器不改变结果，不代替CPU数学golden或整网收敛验证。

## 第三步：有真实训练捕获时，精确回放输入与期望输出

无需额外prepare脚本，把捕获目录传入同一个入口：

```bash
REPO=$(pwd)
bash integration/wuhan/run_timer_capture.sh \
  --variants "$REPO/worker-variants/timer-rho6-a1" \
  --out /absolute/path/to/experiments/timer-real-main-a1 \
  --world-size 8 --capture /absolute/path/to/layer-capture --repeat 3
```

捕获格式为本仓库 `WuhanReplayCapture`：每rank有 `rankN.pt` 和 `rankN.json`，包含config、X/IDs/prob/W1/W2/dY、完整placement及Y/dX/dp/dW1/dW2期望SHA。先核对文件及各tensor SHA，再按捕获config创建EP分组和池；命令行shape在capture模式下不覆盖捕获配置。捕获必须为panel/MoonEP，world与实际启动一致。对每次调用验证在线placement与捕获一致、五输出与捕获期望逐位一致，再比较TimerOFF/ON。

历史主层输入来自step10 `model.model.layers.5.mlp.experts`；原始tensor约4GB，属于实验数据，不上传源码仓库。同事若要逐位重现这次历史输入，需取得这套rank文件；只有shape即可使用上一节合成模式。

## 输出、判断与排错

最终必须看到 `TIMER_CAPTURE_PASS` 各rank回执和根目录 `ACCEPTANCE.json` 的pass为true：

```text
OUT/
  control/                  # TimerOFF，日志、各rank与逐次五输出回执
  timer/                    # TimerON，同上；rankN/step0/ 下保存原始NPY
  kernel-trace/
    all_8_rank_forward_trace.json
    all_8_rank_backward_trace.json
    trace-export-audit.json
  ACCEPTANCE.json
```

原始NPY包括planner_pairs、前反向原TimerV2 buffer、stage_header/indices/pairs。图含权重搬移/ready、source提交/释放请求、反向restore、partial、owner SUM及计算。每rank32AIC+64AIV，共96条逻辑轨道；逐条保留原始周期，同轨道正时长事件零重叠，外层范围用边界标记。详细口径见[导出说明](TIMER-TRACE.md)。前向或反向未收齐、未知schema、重复坐标、同轨道冲突时导出失败。

- 输出目录存在：改用新目录；不覆盖旧结果。
- `schema6 TimerON SO` / `schema1 planner`：加载了旧库或混用了OFF/ON目录，重新构建并检查LD_LIBRARY_PATH。
- `requires timer=True`：TimerON planner被普通executor加载，应恢复TimerOFF目录；不要扩大普通header来掩盖误用。
- 缺CATLASS头：先初始化submodule或补齐对应依赖。
- 五输出/placement不一致：查看首个失败iteration回执和两侧run.log；入口会停止，不自动放宽阈值或导出“成功图”。
- `RunAicpuIndOpThreadInit`：保留完整日志并检查HCCL/CANN和设备状态；本轮曾在首次HCCL初始化遇到该错误，清理确认后新attempt通过，不据此自动归因于MegaMoE算子或循环重试。
- 如需独立启动，用 `torchrun ... runtime_bootstrap.py --native-only capture_kernel_timer.py --help` 查看参数；正常使用优先采用wrapper，确保OFF/ON隔离。

两个进程之间不复用runtime或SHMEM状态。图的rank时间原点独立，不能用跨rank横坐标差计算通信时延；TimerON和包含hash/CPU回读的运行不能作为正式性能数据。实际训练中扩展采集时应选择目标层和安全采集窗口，不能每层每步无条件同步/回读。

## CPU检查

```bash
python integration/wuhan/test_timer_capture_cpu.py
python integration/wuhan/test_timer_trace_cpu.py
```

本次交付验证范围及固定源码/动态库身份见[TIMER-CAPTURE-VALIDATION.md](TIMER-CAPTURE-VALIDATION.md)。
