# S0 共享专家反向正确性实验

本入口只测试同一 launch 的 shared serial prefix + 原 routed backward。它保留生产 routed forward，以独立 PyTorch forward 产生本地 BF16 `X/rawGU/A`，不替换同事的前向实现，也不代表整网、TP、bias、redispatch 或性能验收。

## 参考与判据

- 固定外部 `dY`；先运行原 `pool.backward`，再以同一冻结 placement、新控制 epoch 运行 `backward_replica_shared`。`dp/dWr1/dWr2` 必须逐位一致。
- 独立参考采用 PyTorch `Linear → FP32 clamp/SiLU → BF16 A → Linear` 的 autograd。Gate 只做上界 clamp，up 做双侧 clamp，等号位置导数保留；边界 fixture 包含 BF16 `±1` 及两侧相邻表示值。
- Gate/up 使用两个独立 X leaf，使两次 Linear 的 `dX` 各自先舍入 BF16，再相加并舍入 BF16；最后与原 routed BF16 `dX` 相加并舍入 BF16。这是专家输入的 `dx_direct`，未加入框架 router 路径梯度。
- `dx_direct/dWg/dWu/dWd` 直接沿用 `test_bridge_npu.metrics` 的原 backward 判据，不修改容差。连续调用所有七梯度还必须逐位重复一致。
- 检查 `X/rawGU/A`、shared/routed weights、`dY/prob`、counts、dispatch 和四个 placement 保存张量调用前后哈希不变。原 routed state 仅排除已存在的 256 个 int32 backward progress counter，检查其余每个字节；CPU contract test 与现有 layout parser 交叉核对该精确范围。
- 非法 dtype、非连续 X、rawGU/A 错误形状以及 clip limit 的 `0/-1/NaN/Inf/1e-300`（FP32 下溢）必须在 native launch 前抛错。非法调用后继续下一次有效 epoch，验证池仍可使用。
- 加载的 worker SO 必须报告 `megaflux_shared_expert_backward_version()==1`、TimerOFF；记录实际 SO SHA256。失败保留输出目录，重新运行使用新编号。

## 运行矩阵

| 编号 | World/EP | M/H/routed F/shared Fs | limit | 场景 | 主要覆盖 |
|---|---|---|---|---|---|
| s0-ep4-tail | 8/4 | 257/512/512/256 | 1 | balanced,hot,rotate,boundary | 两个 EP 组、尾行、Fs≠F、WGrad 零 tile 核心、空 owner/slot、精确 clip 边界、3 epoch |
| s0-ep8-tail | 8/8 | 257/512/512/256 | 1 | balanced,hot,rotate,boundary | EP8、多个活动 replica、相同独立判据 |
| s0-ep2-tail | 8/2 | 257/512/512/256 | 1 | balanced,hot,rotate,boundary | 四个 EP 组、EP2 路径 |
| s0-wide | 8/4 | 129/1024/512/768 | .5 | hot,rotate | 不同宽度与累计 K 尾行、强 clipping、WGrad 零 tile 核心 |
| s0-limit6 | 8/8 | 257/512/512/256 | 6 | balanced,hot,rotate | 正常大阈值，禁止用 limit0 替代无 clip |

M/Fs 范围和尺寸拒绝由 host/binding 静态测试另行覆盖。这里执行的八 rank 非法输入套件不启动任何故意不匹配的集合通信。hot/rotate 会检查确有活动 replica；若 planner 未复制则该项失败，不把它计为 replica 通过。

## 复现命令

先由实验执行者完成机器健康、进程和 root fuser 准入，并提供隔离的 CANN/torch_npu/SHMEM 环境、当前 binding 和 capability1/TimerOFF worker 库。以下命令不承担准入和机器清理功能，不应在未准入节点运行。`ROOT/WORKER_NATIVE/RUN_ROOT` 必须指向本次隔离目录。

```bash
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$WORKER_NATIVE:$ROOT/ascend_deepep/lib:${LD_LIBRARY_PATH:-}"
export WUHAN_PUBLISH_SYNC=device WUHAN_CONTROL_TAPE_DEVICE=npu
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=2
unset WUHAN_REPLAY_CAPTURE_DIR
python3 "$ROOT/integration/wuhan/test_shared_expert_backward_contract.py"
torchrun --nnodes=1 --nproc-per-node=8 --master-port=32400 \
  "$ROOT/integration/wuhan/runtime_bootstrap.py" --native-only \
  "$ROOT/integration/wuhan/test_shared_expert_backward_npu.py" \
  --out "$RUN_ROOT/s0-ep4-tail-a1" --ep 4 --tokens 257 --hidden 512 \
  --ffn 512 --shared-ffn 256 --limit 1 --repeat 3 --shmem-port 32410
```

其他矩阵行复用命令，更换全新 `--out`、两组端口及对应参数。wide 行追加 `--cases hot rotate`；limit6 行追加 `--cases balanced hot rotate`。默认同时运行 host 非法输入套件。`--skip-invalid` 仅适合已单独完成该套件后的复测，应在报告注明。

每个 case 在第一次 iteration 前用 `open('xb')` 写入 `rankN-CASE-inputs.pt`，保存真实 CPU tensor `X/Wg/Wu/Wd/dY/rawGU/A`、原 routed `W1/W2/logical_ids/topk_weights`，以及独立 oracle 的 `dx_shared/dWg/dWu/dWd` 四输出。metadata 记录 rank/EP/shape、基础与 rank seed、case 顺序、math/ABI version、两个 clip limit、SO 与测试/参考源码指纹；不保存全环境、凭据或 device 指针。输入文件不是仅用 seed 代替的数据描述。

第一次 routed forward 后、两个 backward 前，用同样的防覆盖方式写入 `rankN-CASE-routed-saved.pt`，包含 counts、完整 opaque `training_saved_state`、dispatch、四个冻结 placement tensor、forward status 和 `Y_routed`。同时写入 `rankN-CASE-artifacts.json`，记录两份 PT 的相对文件名、实际文件 SHA256 和字节数，因而即使随后 kernel 失败仍能找到输入/保存态证据。逐 iteration 与最终 rank 收据也引用这两份文件指纹。所有 PT 都是 CPU tensors 和普通 metadata，可用 `torch.load(path, map_location='cpu', weights_only=True)` 读取；重放保存态须新建合法执行 plan/epoch，不复用旧地址或序列化的 forward epoch 直接发 kernel。

每 rank 另写 `rankN.json` 总收据和 `iterations-rankN.jsonl` 逐场景/逐 epoch 数值、逐位、只读、placement/copies 和执行 counts 证据。仅八个总收据均存在、`pass_=true`、SO SHA 一致、artifact 文件 SHA 校验通过且 torchrun 正常退出，才计为当前矩阵行通过。PT 与哈希均在数值检查流程中生成，不进入任何性能计时。当前入口不测量性能，不输出性能收益。
