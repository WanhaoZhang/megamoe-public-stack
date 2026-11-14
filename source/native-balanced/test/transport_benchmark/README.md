# DeepEP UDMA / SDMA / MTE 搬运 Benchmark

该目录独立测量 Ascend950 上 DeepEP/MegaMoe 关心的四类搬运路径，不调用正式 MegaMoe 算子：

测量口径参考 `catccos_xixing_remote_read` 仓 `costmodel_accuracy_validation` 分支的 A5 hardware benchmark，以及当前 SHMEM 的 `mte_perftest`/UDMA/SDMA 示例。UDMA Scalar 带宽默认参数与 [SHMEM官方Ascend950 Scalar perftest](https://gitcode.com/cann/shmem/wiki/perftest-udma-scalar-d2d-ascend950) 对齐：`PIPE_S`、`block=1`、`warmup=100`、`loops=1000`、`batch=0`。这里只对齐测量流程；实际仍构建 DeepEP 子模块锁定的 SHMEM commit，因此绝对结果不要求与官方页面完全相同。

| 路径 | READ | WRITE | 完成动作 |
|---|---|---|---|
| 远端 UDMA | `aclshmemx_udma_get_nbi` | `aclshmemx_udma_put_nbi` | `aclshmemx_udma_quiet(peer)` |
| 远端 UDMA多QP逐行 | `aclshmemx_udma_qp_get_nbi` | `aclshmemx_udma_qp_put_nbi` | 每个QP分别执行`aclshmemx_udma_qp_quiet(peer, qp_idx)` |
| 本地 SDMA | `aclshmemx_sdma_get_nbi` | Ascend950 不支持，测试中明确 `SKIP` | `aclshmemx_sdma_quiet` |
| 远端 MTE | `aclshmemx_mte_get_nbi` | `aclshmemx_mte_put_nbi` | `aclshmemx_mte_quiet` |
| 本地 MTE | GM → UB | UB → GM | `PipeBarrier<PIPE_ALL>` |

这里的 READ/WRITE 按数据方向命名：READ 表示从源读取到本地目标，WRITE 表示把本地源写到目标。

## 1. 测量口径

### 原生 RTT

RTT固定测64 B，并额外保留512 B诊断值。每次只执行一条原生搬运并等待对应引擎完成：

| 路径 | READ RTT口径 | WRITE RTT口径 |
|---|---|---|
| 远端 MTE | 远端GM → 本地UB → `PipeBarrier` | 本地UB → 远端GM → `PipeBarrier` |
| 远端 UDMA | 远端GM → 本地GM → `udma_quiet` | 本地GM → 远端GM → `udma_quiet` |
| 本地 SDMA | 本地GM → 本地GM → `sdma_quiet` | Ascend950不支持 |
| 本地 MTE | 本地GM → UB → `PipeBarrier` | UB → 本地GM → `PipeBarrier` |

Remote MTE RTT与旧A5 `remoteReadRttNs` 的核心口径一致，不包含UB→本地GM、`mte_quiet`和整Cache Flush。

RTT预热后使用一个总计时窗口连续执行全部循环，再除以循环次数；不会在每次64 B搬运前后各读一次timer。汇总中的RTT `min_ns/max_ns` 因此与该平均值相同。

### 带宽

带宽模式连续提交 `batch_size` 个 NBI 操作，然后统一 quiet：

```text
NBI × batch_size → quiet → NBI × batch_size → quiet ...
```

`batch_size=0` 与 SHMEM 官方 `udma_perftest` 一致，表示一次提交本轮全部 NBI，最后只执行一次对应完成动作；不是“每条 NBI 后完成”。所有路径默认固定执行1000次，并统一输出十进制GB/s。

UDMA 带宽同时输出两种流量口径：

- `remote_udma`：单向测试，只由 rank 0 向 rank 1 搬运，对应官方 PUT/GET 的单向口径。
- `remote_udma_bi`：双向/环形并发测试，所有 rank 同时向下一个 rank 搬运，对应并发业务压力；它不能与官方单向 PUT 表格直接比较。

远端MTE同样区分`remote_mte`单向和`remote_mte_bi`双向/环形并发。SHMEM的MTE perftest没有`batch=16`设计，而是预热后连续执行测量循环，并在末尾用流水同步收口，因此本测试不再人为插入该默认值。所有路径统一使用`warmup=100`和`loops=1000`；`batch`只控制NBI路径，本地MTE在每个64 KiB chunk后通过`PipeBarrier`完成。当前SHMEM没有可直接对齐的Ascend950 SDMA性能perftest，此处统一的是本测试内部测量口径。

NBI路径的warmup会先连续提交100条NBI，再执行一次quiet；计时区间是1000条NBI加最后一次quiet。UDMA与官方Scalar perftest的默认流程一致；其他路径采用相同循环数以便横向比较。

默认带宽测试以1个AIV block为主，避免把单核消息大小、并发核数和聚合总流量混在一起。远端MTE额外测试2个AIV block，用来同时保留单核结果和接近链路峰值的结果；本地MTE测试1、4和16个AIV block，用来观察本地搬运能否随并发核数打满。

- 远端UDMA和本地SDMA默认使用1个AIV block。
- 远端MTE默认同时测试1和2个AIV block；每个block独立搬运完整的1 MiB消息。
- 本地MTE默认测试1 MiB和16 MiB，每种大小分别测试1、4和16个AIV block。
- `model_parameters.csv` 会从有效测点中选择峰值，并记录对应的`bandwidth_aiv_blocks`。
- 如需研究其他并发规模，可显式设置对应的blocks环境变量覆盖默认值。
- Ascend950 当前不支持 `aclshmemx_sdma_put_nbi` 的 SDMA WRITE。建模本地 GM→GM 大块搬运时使用 SDMA GET 方向，脚本会明确输出 `SKIP`，不会把它当作失败或用其他结果冒充。
- 所有 rank 同时运行时，汇总中的“所有 rank 合计”只是并发流量总和；单链路建模优先使用单 rank 结果。

## 2. 构建

加载 CANN 环境，在仓库根目录执行：

```bash
bash test/transport_benchmark/build.sh
```

脚本会先构建当前 DeepEP 固定版本的 SHMEM，然后生成：

```text
test/transport_benchmark/install/bin/deepep_transport_benchmark
```

## 3. 运行

使用两张可用卡运行：

```bash
bash test/transport_benchmark/run.sh 6,7
```

同一个`run.sh`会执行原有搬运benchmark、固定64 KiB的双向普通UDMA loop sweep，以及Remote MTE BI Token-row sweep，不再运行UDMA QP测试。UDMA loop sweep要求恰好使用两张卡；如果传入更多设备，原有benchmark和Remote MTE BI Token-row sweep照常运行，UDMA loop sweep明确`SKIP`。

### 双向普通UDMA 64 KiB Loop Sweep

该测试只测双向普通UDMA，不测单向，也不调用QP专用接口。两张卡同时使用`PIPE_S`、1个AIV和普通QP 0向对端搬运，默认扫描：

```text
messageBytes = 64 KiB
loops        = 8,16,32,64,128,256,512
执行顺序     = udma_put_nbi/get_nbi × loops → udma_quiet(peer)
```

每个loops测点使用独立的SHMEM进程，计时只覆盖kernel内连续提交的NBI和最后一次quiet，不包含SHMEM初始化、kernel launch及Host校验。结果同时输出rank0、rank1各自GB/s和双向合计GB/s，并报告本次扫描中第一个达到双向峰值95%的loops。详细CSV和独立汇总仍保存在本次run目录的`udma_loop_sweep/`子目录；同一张表也会自动合并到主`summary.md`的UDMA章节，不参与Cost Model峰值选择。

### Remote MTE BI Token-row Sweep

该测试由`run.sh`自动执行，所有rank同时按环形peer进行Remote MTE READ/WRITE。默认测试：

```text
单行字节 = 2 KiB、4 KiB、8 KiB
总行数   = 16、64、256、1024
AIV数    = 1、2
warmup   = 每个AIV预热100次单行NBI
计时组数 = 100
```

每次`aclshmemx_mte_get_nbi()`或`aclshmemx_mte_put_nbi()`只搬一行，并按行号访问连续且互不重叠的地址。1个AIV负责全部行；2个AIV平分固定总行数，例如总行数64时每个AIV每组提交32次NBI。每个AIV先预热100次单行搬运，预热后quiet一次；正式计时执行100个完整行组，每组结束时各AIV都quiet，再进入下一组：

```text
单rank单组字节 = 单行字节 × 总行数
每AIV loops     = 总行数 ÷ AIV数
GB/s            = 单rank单组字节 × 计时组数 ÷ rank makespan秒数 ÷ 10^9
rank makespan   = max(all end_cycle) - min(all start_cycle)
```

详细结果保存在本次run目录的`remote_mte_bi_row_sweep/`子目录，并自动合并到主`summary.md`的MTE章节。汇总会报告每个`操作 × 行字节 × AIV数`组合中，第一个达到本次扫描峰值95%的总行数；该扫描不参与Cost Model峰值选择。

MTE与UDMA QP的独立汇总都会在结果表前说明单行字节、总行数、每AIV行数、配置/活跃QP数、计时组数、单组字节、Rank数、聚合带宽及校验字段，便于直接对照两种搬运方式。对于公平对比，建议两者都使用两张卡运行，并比较MTE双向表与UDMA QP的`read_bi/write_bi`表。

### 单向与双向UDMA QP逐行测试

QP测试与主benchmark分开运行：

```bash
bash test/transport_benchmark/run_udma_qp.sh 6,7
```

该测试固定使用两张卡和`PIPE_S`。`read/write`保持rank0发起的原单向口径；`read_bi/write_bi`表示两张卡同时工作，其中PUT由rank0与rank1同时向对端写，GET由两边同时从对端读。默认依次测试`QP/AIV=1,4,16,28`；每个AIV独占一个QP，并按以下方式分配行：

```text
AIV/QP q负责：row q、q+QP_COUNT、q+2×QP_COUNT……
每次QP PUT/GET只搬一行
每个QP提交完本组中自己负责的行后quiet一次
```

默认对单向、双向分别测试`2 KiB/4 KiB/8 KiB × 16/64/256/1024行`。每个AIV先预热100次单行搬运，预热后quiet一次；正式计时100个完整行组，每个QP在进入下一组前完成自己的上一组。当配置QP数大于总行数时，SHMEM仍按请求值配置QP，但kernel只启动`min(QP数, 总行数)`个AIV，空闲QP不参与计时。每个AIV独立记录所属QP的起止cycle，Host使用所有参与AIV的最早开始到最晚结束作为本rank makespan；不使用要求全部物理AIV参与的`SyncAll`，因此QP数可以小于28。计时不包含Host校验和kernel launch。单向结果报告rank0发起带宽；双向结果分别记录两个rank的makespan并将两边带宽相加：

```text
单rank GB/s = 单行字节 × 行数 × 计时组数 / rank makespan秒数 / 10^9
双向合计 GB/s = rank0 GB/s + rank1 GB/s
```

每个QP数测点都会新启动一对rank进程，并在SHMEM初始化前通过`aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA, qp_count)`配置。每个AIV分别记录起止cycle，单rank计时采用所有参与AIV的`max(end_cycle) - min(start_cycle)`，覆盖多AIV进入测量区间的启动错位；`max_core_cycles`仅作为旧的“最慢单核自身耗时”对照值。汇总会报告每个`操作 × 行字节 × QP数`组合中，第一个达到本次扫描峰值95%的总行数。结果放在独立的`output/udma_qp_<timestamp>_<pid>/`目录，不会混入普通transport的Cost Model汇总。

默认测点：

```text
RTT：64 B，额外保留512 B诊断值
UDMA带宽：256 B～1 MiB按2倍递增，额外包含896 KiB
远端MTE带宽：1 MiB
本地MTE带宽：1 MiB、16 MiB，分别测试1、4、16个AIV block
```

运行期间可在另一个终端查看实时进度：

```bash
tail -f test/transport_benchmark/output/run_*/rank0.log
```

结果保存在：

```text
test/transport_benchmark/output/run_<时间>_<PID>/
├── config.txt
├── rank0.log
├── rank1.log
├── results.csv
├── model_parameters.csv
├── summary_mte_udma_<YYYYMMDD_HHMMSS>.md
├── udma_loop_sweep/
│   ├── results.csv
│   └── summary.md
└── remote_mte_bi_row_sweep/
    ├── results.csv
    └── summary.md
```

`config.txt` 同时记录 DeepEP、SHMEM commit，方便后续模型参数追溯。

`model_parameters.csv`和`summary_mte_udma_<YYYYMMDD_HHMMSS>.md`会自动汇总64 B原生RTT与峰值带宽；RTT参数取本次已测有效记录中的最大值，用作保守建模，512 B仅保留为诊断值。主汇总将UDMA、SDMA和MTE带宽拆成三张独立表，并把64 KiB loop sweep放在UDMA表之后。带宽单位统一保持十进制`GB/s`：

```text
1 GB/s = 10^9 B/s
```

主汇总开头包含完整字段说明、Blocks计算示例和`model_parameters.csv`逐字段解释，可直接用于结果评审或对外讲解。

- `bandwidth_aiv_blocks`：达到所列峰值GB/s时，单rank参与并发搬运的AIV核数。

当前 BF16 UDMA 默认用例为 `M=4096, N=6144, K=3584, TopK=16, EP=2`，Golden 均匀路由下：

```text
BF16 token行 = K * 2 B = 3584 * 2 B = 7 KiB
Combine完整row-group = 128 * 7 KiB = 896 KiB
Dispatch每个peer/expert = 2048 * 7 KiB = 14 MiB
```

Combine 在 peer 边界或最后一个 row-group 处可能小于896 KiB。UDMA/SDMA保留小消息到1 MiB的曲线；远端MTE使用1 MiB代表点，本地MTE额外使用16 MiB检查大块搬运是否打满。实际非均匀路由下，Dispatch 大小为`该源rank发给该目标expert的token数 × K × 2 B`，不会固定为14 MiB。

## 4. 常用自定义参数

运行脚本通过环境变量覆盖默认值。例如只做较短的定位测试：

```bash
export DEEPEP_BENCH_UDMA_BANDWIDTH_SIZES=4096,16384,65536
export DEEPEP_BENCH_REMOTE_MTE_BANDWIDTH_SIZES=1048576
export DEEPEP_BENCH_LOCAL_MTE_BANDWIDTH_SIZES=1048576,16777216
export DEEPEP_BENCH_WARMUP=20
export DEEPEP_BENCH_LOOPS=100

bash test/transport_benchmark/run.sh 6,7
```

其他参数：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `DEEPEP_BENCH_UDMA_BANDWIDTH_SIZES` | `256`～`1048576`按2倍递增，另含`917504` | UDMA和SDMA带宽消息大小；与MTE分开配置 |
| `DEEPEP_BENCH_REMOTE_MTE_BANDWIDTH_SIZES` | `1048576` | 远端MTE带宽消息大小；默认测1 MiB |
| `DEEPEP_BENCH_LOCAL_MTE_BANDWIDTH_SIZES` | `1048576,16777216` | 本地MTE带宽消息大小；默认测1 MiB和16 MiB |
| `DEEPEP_BENCH_WARMUP` | 100 | 所有transport和mode统一使用的预热次数 |
| `DEEPEP_BENCH_LOOPS` | 1000 | 所有transport和mode统一使用的计时循环次数 |
| `DEEPEP_BENCH_REMOTE_MTE_BLOCKS` | `1,2` | 远端MTE带宽AIV block数；同时保留单block和双block结果 |
| `DEEPEP_BENCH_LOCAL_MTE_BLOCKS` | `1,4,16` | 本地MTE带宽AIV block数；观察并发扩展能力 |
| `DEEPEP_BENCH_LOCAL_SDMA_BLOCKS` | `1` | 本地SDMA带宽AIV block数；默认单block |
| `DEEPEP_BENCH_BATCH_SIZE` | 0 | NBI带宽两次完成动作之间的提交数；0表示本轮全部NBI提交后只完成一次 |
| `DEEPEP_BENCH_UDMA_PIPE` | `pipe_s` | UDMA WQE 使用 `PIPE_S` 或 `PIPE_MTE3` |
| `DEEPEP_TRANSPORT_BENCHMARK_TIMEOUT` | 1800 | 每个 rank 的超时时间（秒） |
| `DEEPEP_BENCH_RUN_UDMA_LOOP_SWEEP` | `1` | 是否继续运行双向普通UDMA 64 KiB loop sweep；`0`表示跳过 |
| `DEEPEP_BENCH_RUN_REMOTE_MTE_BI_ROW_SWEEP` | `1` | 是否继续运行Remote MTE BI Token-row sweep；`0`表示跳过 |

Remote MTE BI Token-row sweep使用：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `DEEPEP_BENCH_REMOTE_MTE_ROW_SIZES` | `2048,4096,8192` | 每次MTE NBI搬运的一行字节数 |
| `DEEPEP_BENCH_REMOTE_MTE_TOTAL_ROWS` | `16,64,256,1024` | 单rank单组固定总行数 |
| `DEEPEP_BENCH_REMOTE_MTE_ROW_BLOCKS` | `1,2` | 参与搬运的AIV数；总行数在AIV间均分 |
| `DEEPEP_BENCH_REMOTE_MTE_WARMUP_OPERATIONS` | `100` | 每个AIV在计时前执行的单行NBI次数，之后quiet一次 |
| `DEEPEP_BENCH_REMOTE_MTE_MEASURED_GROUPS` | `100` | 正式计时的完整行组数；每组结束后quiet |

UDMA QP逐行测试使用独立环境变量：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `DEEPEP_BENCH_QP_ROW_SIZES` | `2048,4096,8192` | 单次QP PUT/GET搬运的行字节数 |
| `DEEPEP_BENCH_QP_ROW_COUNTS` | `16,64,256,1024` | 一组内连续搬运的总行数 |
| `DEEPEP_BENCH_QP_COUNTS` | `1,4,16,28` | 依次测试的QP数/AIV block数；每个AIV独占一个QP |
| `DEEPEP_BENCH_QP_COUNT` | 未设置 | 兼容单个QP数的旧参数；设置后覆盖默认QP列表 |
| `DEEPEP_BENCH_QP_WARMUP_OPERATIONS` | `100` | 每个AIV在计时前执行的单行QP PUT/GET次数，之后quiet一次 |
| `DEEPEP_BENCH_QP_MEASURED_GROUPS` | `100` | 计时区间重复执行的完整行组数；每个QP完成本组负责的行后quiet |

普通UDMA双向loop sweep使用：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `DEEPEP_BENCH_UDMA_LOOP_COUNTS` | `8,16,32,64,128,256,512` | 64 KiB NBI在一次quiet前的连续提交次数 |
| `DEEPEP_BENCH_UDMA_LOOP_WARMUP` | `100` | 每个loops测点在计时前连续提交的预热NBI数；预热后quiet一次 |

比较 UDMA WQE 提交方式：

```bash
DEEPEP_BENCH_UDMA_PIPE=pipe_s \
  bash test/transport_benchmark/run.sh 6,7

DEEPEP_BENCH_UDMA_PIPE=pipe_mte3 \
  bash test/transport_benchmark/run.sh 6,7
```

比较不同 quiet 周期时可显式设置 batch。比如每16次或128次NBI执行一次quiet：

```bash
DEEPEP_BENCH_BATCH_SIZE=16 \
  bash test/transport_benchmark/run.sh 6,7

DEEPEP_BENCH_BATCH_SIZE=128 \
  bash test/transport_benchmark/run.sh 6,7
```

默认 `DEEPEP_BENCH_BATCH_SIZE=0` 表示本轮所有NBI提交完后只执行一次对应完成动作。主benchmark的所有transport和mode默认均执行100次warmup和1000次计时循环；Remote MTE BI Token-row sweep的loops由`总行数÷AIV数`决定。

## 5. 结果解释

- `rtt_avg_ns`：本次已测有效64 B记录中最慢一次的平均完成时间。
- `gbps`：单 rank 上所有参与 block 的聚合有效带宽。
- `total_cycles`：原始测量值，避免换算口径变化后无法重新分析。

注意：

1. UDMA/SDMA 是 GM→GM；本地 MTE READ/WRITE 是 GM↔UB 原子搬运，语义并不完全相同。
2. 远端 MTE接口内部用 UB 完成 GM→GM，因此它可以直接和 UDMA 的远端 GM→GM结果对比。
3. 当前校验在计时区间外检查每个目标块的首尾32字节；本地 MTE READ 只能检查回写的头32字节。
4. 首轮设备验证若某一种引擎报507015/507035，先保留该 rank 的 plog 和 `/root/shmem/log`，不要直接用其他路径的结果替代。
