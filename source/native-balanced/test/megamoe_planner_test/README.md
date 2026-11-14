> 计划校验已经统一为融合实现，路由实验仅比较 strict-fused/direct-fused 的 A/B/B/A 顺序；历史 generic 校验对照不再是可执行模式。参见 [默认校验说明](../../integration/wuhan/FUSED-PLAN-DEFAULT.md)。

# Planner + MegaMoe 真正 EP8 benchmark

本入口只运行单机 WORLD=EP=8 的单层 online 前反向；不会启动整网训练。

## 对照链路

| backend | 内容 |
|---|---|
| `original` | 原始 MegaMoe F/B，无 planner |
| `moonep-reference` | 原 MoonEP planner、prefetch、owner SUM 组件串接 MegaMoe F/B |
| `local-serial` | 本仓 planner + 独立权重复制 + MegaMoe F/B + 独立 owner SUM |
| `s9` | 同一 planner + S9 融合执行 |

所有链路使用相同输入、初始权重、路由概率和接收容量。完整 F/B 包含规划、home 权重发布、反向恢复、owner SUM 和完成检查；不含 FSDP all-gather/reduce-scatter。没有实现 same-plan replay。

## 构建和依赖

只在自己独立的源码目录和容器内构建。准备固定版本的 CANN、SHMEM、FSDPTurbo、原 MoonEP，以及本仓 gitlink 固定的 catlass/catccos；不要更新共享环境。原 MoonEP 的 Python/native 包需要已经可导入。

```bash
source /path/to/cann/set_env.sh
export SHMEM_HOME_PATH=/path/to/pinned/shmem/install
bash test/megamoe_planner_test/build_benchmark.sh
python -m unittest discover -s test/megamoe_planner_test -p test_contract_cpu.py -v
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python test/megamoe_planner_test/offline/check_parallel_cpu.py
```

构建关闭 Timer、worker 调度、row-ready 和 owner compact 实验，固定 panel restore=2、operand-ready=1、delay=0。库文件及源码摘要写入构建回执；已有冻结构建不会被覆盖。请在新源码目录重新构建，而不是修改已用于测量的构建目录。

## 启动

在宿主机上设置下列参数。路径均为示例，需要替换；`container-id` 和 `codex.task` 标签必须匹配自己已有容器，启动器不会创建、停止或修改任何容器。

```bash
LAUNCH=(
  --host-root /data/my-task/source
  --container-root /project/source
  --mount-root /data/my-task --mount-target /project
  --container my-container --container-id FULL_CONTAINER_ID
  --task-label MY_TASK_LABEL
  --cann-env /path/in/container/cann/set_env.sh
  --fsdp-root /project/deps/FSDPTurbo
  --moonep-root /project/deps/MoonEP
  --shmem-root /project/deps/shmem/install
)
python3 test/megamoe_planner_test/gate_and_run.py "${LAUNCH[@]}" --out readiness --gate-only
python3 test/megamoe_planner_test/campaign.py "${LAUNCH[@]}"   --name balanced-run1 --precision-workers 4 --   --tokens 4096 --hidden 7168 --ffn 3072 --experts 32 --slots 4 --topk 6   --route balanced_random --planner moonep --warmup 10 --repeat 30
```

每次设备运行都重新检查健康、设备进程、设备文件持有者和独占目录标识。默认仅接受 OK；若机器负责人明确允许，额外传 `--allow-warning-alarm`，仍须空闲。失败立即退出，不自动重试或清理其他任务。每次换新 `--name`，保留失败证据。

四类场景使用同一 shape：

| 场景 | 参数 | 定义 |
|---|---|---|
| 严格均衡 | `--route balanced_random` | 打乱顺序，但专家总计数严格均衡 |
| 自然 random | `--route random` | 各 rank 独立随机、每 token TopK 不重复，轻度不均衡 |
| 1.5 倍 | `--route controlled --load-ratio 1.5` | 最大卡 token 数 / 平均值 = 1.5 |
| 2 倍 | `--route controlled --load-ratio 2` | 最大卡 token 数 / 平均值 = 2 |

`scenarios.json` 另列小形状等可选输入，是否支持以启动器参数和布局检查为准。容量在规划前按固定输入离线校准；这不是规划后动态分配内存。

## 正确性与计时

先运行四组精度，检查 Y/dX/dP/dW1/dW2、冷专家、被覆盖共享池的恢复和重复调用。保持原精度阈值并追加 relative-L2<=0.05；同计划串行/S9 还要求计划一致。CPU 并行 checker 只加速离线张量比较，不改变阈值或 NPU 性能。

性能运行只接受同输入、同源码、同依赖及同加载库的成功精度回执。每个样本先取八 rank 最大时间，再统计 median/p95/std；四链路正序和反序单独报告，不把双向样本混合掩盖漂移。

`--mode diagnostic` 的阶段边界同步和可选 profiler 用于归因，不能当正式性能数据，也不能把重叠阶段相加。

## 可选原路由策略（统一融合校验）

使用方法、历史收益、保留或撤销标准见 [交付说明](../../docs/planner-megamoe-delivery.md)。
在上述 baseline 构建之后运行 `bash test/megamoe_planner_test/build_routing_variants.sh`；
再以同一组 `LAUNCH` 参数调用 `routing_campaign.py --name NEW_NAME --precision-workers 4 -- SHAPE_ARGS`。
此入口只比较 strict-fused、direct-fused，按 A/B/B/A 顺序测路由策略；计划校验始终融合，无单独开关。
