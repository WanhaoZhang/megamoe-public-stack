# 普通 BF16 物理机单机 / 多机前向验证

单机和多机共用原来的 **`test_mega_moe.sh` → `test_mega_moe.py`**。参考
`xx_ops_transformer_hs/tests/megamoe/operater_cascade/mega_moe.py` 的
`server_num / server_index / master_ip / rank_per_dev` 组织方式，
每台物理机各运行一次脚本。无需 torchrun、共享目录或手动传 bin/json。

当前多机模式只允许普通 BF16 前向；原有单机 BF16/MGroup/UDMA 入口保持兼容。
两台各 8 卡时 EP=16，每卡 4 个专家，
总共 64 个专家；TopK=16。原有 `test_mega_moe.sh` 的单机用法保持不变。

2026-09-07 验证反馈：切换 TCPStore 旧后端后，用户反馈普通 BF16 双机可跑通，
本轮 shape 为 `M=512, N=8192, K=7168, EP=16`。此反馈不覆盖反向或长循环。
反向开发复用说明见 [反向适配与双机测试参考](../../docs/MegaMoe_BF16_反向适配与双机测试参考.md)。

## 1. 修改原 py 顶部的配置

直接修改 `test_mega_moe.py` 顶部的 `SERVER_NUM`、`SERVER_INDEX` 和 `MASTER_IP`。
设备列表仍使用原 sh 的第一个参数，不需要改内部辅助模块 `physical_server_support.py`。

两台使用相同代码、构建匹配的 SO/Python 扩展，在原来能跑单机 BF16 的
CANN/SHMEM/Python 环境中启动。除本机编号和本机设备编号外，测试参数应一致。

| 配置 | 单机 | 双机主机 | 双机从机 |
|---|---|---|---|
| `SERVER_NUM` | `1` | `2` | `2` |
| `SERVER_INDEX` | `0` | `0` | `1` |
| `MASTER_IP` | `"127.0.0.1"` | 主机可访问的实际 IPv4 地址 | 同一主机地址，不填从机地址 |
| sh 第一个参数 | `0,1,2,3,4,5,6,7` | 本机 8 个设备 | 本机 8 个设备 |
| `SHMEM_PORT` | `8899` | `8899` | `8899` |
| `CONTROL_PORT` | `8900` | `8900` | `8900` |

从机填写 `SERVER_INDEX=1`，不需要把设备列表改为 `8,9,...,15`。
程序会将从机本地设备 0–7 映射为全局 rank 8–15。
主机的 8899 和 8900 两个 TCP 端口应空闲且可被从机访问。
8900 仅用于测试协调；算子的设备通信仍使用现有 SHMEM 路径。
脚本不会把“主机 TCP 可达”当成“SHMEM 设备互联已验证”。

## 2. 首次验证：小 shape、一次精度

默认 shape、mode 和循环规则沿用原入口，**不会自动切换成小 shape**。
大小 shape 都手动修改原来的 `mega_moe_cases.csv`，两台保持相同内容。
例如首次验证可填 `M=128, N=1024, K=512`；双机仍为 EP16 / 64 专家 / TopK16。
两台机器的代码仓根目录执行同一条命令（先主机，再从机）：

```bash
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode precision --warmup 0 --loops 1 --repeat 1
```

相同命令在 `SERVER_NUM=1` 时跑单机，在 `SERVER_NUM=2` 时跑双机。
直接使用 Python 也始终是原来的 `test_mega_moe.py`，参数相同。
`--shmem-ip-port` 或环境变量 `ASCEND_DEEPEP_SHMEM_IP_PORT` 显式设置时会覆盖
顶部的 `MASTER_IP` / `SHMEM_PORT`，多机协调也使用这个实际生效的主机 IP；
通常直接修改顶部配置即可，避免同时保留旧的地址覆盖值。

主机等待另一台加入属于正常行为。默认协调等待上限 `TIMEOUT_SECONDS=1800` 秒，
包括等待另一台 CPU golden 完成的时间；它不是 NPU kernel 的超时设置。
测试结果和错误直接打印到屏幕。

重点看：

```text
[TOPOLOGY] server=0/2 ... global_ranks=[0,...,7] EP=16 experts=64 ...
[TOPOLOGY] server=1/2 ... global_ranks=[8,...,15] EP=16 experts=64 ...
[MULTINODE] ... phase=generate all_servers=PASS
[MULTINODE] ... phase=compute all_servers=PASS
[SAVED_STATE] rank=... packed_topk_weights PASS ...
[PRECISION][FUSED-GOLDEN] rank=... y=...
[PRECISION][FUSED-GOLDEN] rank=... expert_token_nums=...
[SUCCESS][GLOBAL] ordinary BF16 EP=16, all servers passed; ...
```

两台均收到最后的全局成功信息才算本轮通过。本机所有 rank 完成不等于全局通过。
任何一台的生成、运行或校验异常会在协调阶段报告给另一台；进程/网络突然退出时，
另一台可能等到协调超时。kernel 内部超时仍由运行时处理。

## 3. 默认大 shape 和性能

小 shape 通过后，两台将同一份 `mega_moe_cases.csv` 手动改回
`M=4096, N=8192, K=7168`，启动命令不变：

```bash
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode precision --warmup 0 --loops 1 --repeat 1
```

两机各自按全局 EP16 生成完整输入和 golden，CPU 内存、磁盘与生成耗时会增加。
均衡路由下每 rank 接收容量仍为 `M*TopK=65536`，SHMEM 分配按全局 EP 重新计算。

精度通过后两台使用相同的 `--mode both --loops 10`，再按需要增加循环数。
本机计时摘要标为 `scope=local`；脚本自动汇总两机的 rank 统计，打印
`[PERF][GLOBAL] ... median_of_rank_medians_us=...`，并在每台输出目录保存
`python_results/performance_global.csv`。全局中位数从所有 rank 的中位数直接计算，
不会先求每机中位数再平均。Timer trace 保留在各机本地，只含本机 rank。

使用原来的 `--enable-timer` 参数之前，需要匹配的带 timer 构建。这个测试入口没有修改 kernel ABI；
若当前安装已匹配代码版本，仅新增启动脚本不要求额外重编。

## 4. 数据与校验边界

- 两机使用相同 shape、seed 和生成算法，独立生成一致的全局输入与 golden。
- 启动算子前核对两机测试参数、NumPy 版本和公共测试/生成脚本的逻辑摘要。
  摘要排除可编辑的机器配置，因此两台 `SERVER_INDEX` 不同不会被误判为代码不一致。
- 每机只加载本机 rank 输入，只校验本机 y、专家计数和持久 packed TopK 权重。
- packed 权重的参考行序依赖所有源 rank 的真实设备排序，不能用 CPU 排序替代。
  脚本通过 TCPStore 自动交换排序索引；默认大 shape 下，每台发布约 2 MiB，
  不传输完整 saved-state、激活或专家权重。
- TCPStore 显式使用旧后端（`use_libuv=False`）。2026-09-07 的 CPU 本机探测中，
  主机 PyTorch 2.7.1 的默认 libuv 后端通过 1 KiB、但 256 KiB 传输超时；
  两机的旧后端均通过这两个大小。该设置用于绕过已复现的后端传输问题，
  不代表双机算子精度已验证；完整测试仍需两机均出现全局成功信息。
- 每台使用独立输出子目录；同时生成多个不同任务时应使用不同端口。

本地 CPU 自测覆盖 rank 映射、配置不一致、跨机错误传播、排序索引交换、
权重错误检出和全局性能汇总；不能替代远端 TCPStore、SHMEM 和 NPU 双机验证。

```bash
python3 test/megamoe_bf16_test/test_physical_launcher.py
```
