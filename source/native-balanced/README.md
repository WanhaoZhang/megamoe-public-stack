# ascend-deepep

[MegaMoE Balanced 源码时序图与八卡实验导读](docs/megamoe-balanced-visual/README.md)。下载仓库后在本地浏览器打开 `docs/megamoe-balanced-visual/index.html`，点击源码链接可跳到 VS Code 对应行。

MegaFlux/MegaMoE 训练算子的完整打点入口：[TimerV2 配对构建、八卡ρ=6采集与两张全rank图](integration/wuhan/TIMER-CAPTURE.md)。包含同shape合成负载和真实训练capture回放，不依赖历史实验目录脚本。

`ascend-deepep` 提供接近 DeepEP 使用方式的 Python `ElasticBuffer`。C++ Runtime 负责 CANN SHMEM 生命周期、对称内存和 runtime workspace 管理；普通算子使用 `libascend_deepep_kernels.so`，MegaMoe W8A8/W4A8 分别独立编译为 `libascend_deepep_mega_moe.so` 和 `libascend_deepep_mega_moe_w4a8.so`。

## 目录结构

```text
ascend-deepep/
├── 3rdparty/
│   ├── shmem/                    # CANN SHMEM submodule
│   ├── catlass/                  # CATLASS submodule
│   └── catccos/                  # CATCCOS submodule
├── ascend_deepep/                # Python API 与随包分发的 kernel .so
│   ├── __init__.py
│   ├── buffer.py
│   └── lib/
├── cmake/                        # Bisheng/AscendC kernel 编译函数
├── csrc/
│   ├── ops/                      # 每个 Python 侧算子的参数校验、输出分配、tiling 构造
│   │   ├── dispatch.*
│   │   ├── combine.*
│   │   └── allgather_matmul.*
│   ├── shmem/                    # SHMEM host 生命周期封装
│   ├── elastic_buffer.*          # ElasticBuffer runtime façade
│   ├── launcher.*                # 调用 libascend_deepep_kernels.so 导出的 launcher
│   └── pybind.cpp
├── include/ascend_deepep/kernels/
│   ├── types.h                   # host/device 共享 tiling ABI
│   └── launch.h                  # kernel shared library 导出的 C ABI
├── kernels/                      # Bisheng 编译的 AscendC/CatCCOS launcher 与 kernel
│   ├── dispatch.cpp
│   ├── combine.cpp
│   ├── allgather_matmul.cpp
│   ├── mega_moe/
│       ├── mega_moe.cpp           # W8A8 融合核入口
│       ├── mega_moe_stub.cpp      # 保留的 W4A8 风格底层调试 runner
│       └── include/               # W8A8 的全部 kernel 实现与 tiling
│   └── mega_moe_w4a8/
│       ├── mega_moe_w4a8.cpp      # W4A8 融合核入口
│       ├── mega_moe_w4a8_stub.cpp # 不经过 DeepEP 注册的独立 runner
│       └── include/               # W4A8 kernel 实现与 host/device tiling
├── scripts/
│   ├── build_shmem.sh
│   ├── build_mega_moe.sh
│   ├── build_mega_moe_w4a8.sh
│   └── build.sh
└── test/
    ├── test_dispatch.py
    ├── test_allgather_matmul.py
    └── megamoe_w8a8_test/
        ├── mega_moe_w8a8_golden.py
        ├── mega_moe_w8a8_cascade.py
        ├── convert_mega_moe_y_to_txt.py
        ├── test_mega_moe.py
        ├── test_mega_moe.sh
        ├── mega_moe_cases.csv
        └── megamoe_w8a8_independent_test/
    └── megamoe_w4a8_test/         # 注册路径与独立 C++ 路径测试
```

## 获取代码

```bash
git clone --recursive <repository-url>
cd ascend-deepep
```

已有仓库时：

```bash
git submodule update --init --recursive
```

当前 submodule：

```text
3rdparty/shmem    https://gitcode.com/cann/shmem.git
3rdparty/catlass  https://gitcode.com/cann/catlass.git
3rdparty/catccos  https://gitcode.com/lishaoxun/catccos.git
```

## 构建

先加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

默认按 A2/910B 构建：

```bash
bash scripts/build.sh
```

如需构建 Ascend950/A5，使用 `-soc_type Ascend950`：

```bash
bash scripts/build.sh -soc_type Ascend950
```

`scripts/build.sh` 会根据 `soc_type` 映射 kernel 架构：

```text
默认 / 未传 -soc_type      -> a2 / dav-c220
-soc_type Ascend950       -> a5 / dav-c310
```

`scripts/build.sh` 只构建 dispatch、combine 和 allgather_matmul，并依次：

1. 初始化/构建 `3rdparty/shmem` 到 `3rdparty/shmem/install`；
2. 使用 Bisheng CCE 编译普通算子，生成 `libascend_deepep_kernels.so`；
3. 将 kernel shared library 复制到 `ascend_deepep/lib/`；
4. 构建并安装 `ascend_deepep._C` PyTorch extension。

MegaMoe device kernel 不参与 `scripts/build.sh` 编译；`scripts/build.sh`
只会随 `_C` 扩展编译用于注册 `torch.classes.AscendDeepEP.MegaMoe`、
`torch.classes.AscendDeepEP.MegaMoeBF16` 和
`torch.classes.AscendDeepEP.MegaMoeW4A8` 的轻量 host adapter。三套独立
device kernel 分别由 `scripts/build_mega_moe.sh`、
`scripts/build_mega_moe_bf16.sh` 和 `scripts/build_mega_moe_w4a8.sh` 编译。

如果只想重建 SHMEM：

```bash
bash scripts/build_shmem.sh
```

## 运行测试

Dispatch/combine call-chain 示例：

```bash
python test/test_dispatch.py --ep-size 4 --mode call-chain
```

CatCCOS `allgather_matmul` 示例：

```bash
python test/test_allgather_matmul.py \
  --ep-size 4 \
  --num-bytes 1052770304 \
  --m 128 \
  --n 256 \
  --k 256
```

W8A8 `mega_moe` 仅支持 Ascend950/A5，并使用独立编译脚本和独立 kernel
动态库。先通过通用构建安装本项目 `_C` 扩展和基础 kernel 库，再运行独立
MegaMoe 构建；独立脚本只编译 MegaMoe device kernel 和调试 stub，并重新
安装包含 `torch.classes.AscendDeepEP.MegaMoe` 注册的 Python 扩展：

```bash
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe.sh -soc_type Ascend950
```

脚本会以 `ASCEND_HOME_PATH` 的父目录为范围自动查找 Bisheng 需要的
`rt.h`。如果 runtime headers 安装在其他位置，可以显式指定：

```bash
bash scripts/build_mega_moe.sh \
  --runtime-include-dir /path/to/directory-containing-rt.h
```

MegaMoe 脚本只构建 `ascend_deepep_mega_moe` 和 `mega_moe_stub`，输出分别为
`ascend_deepep/lib/libascend_deepep_mega_moe.so` 与
`ascend_deepep/bin/mega_moe_stub`，不会重新编译其他算子的 device kernel。
Python 侧与其他算子一致，通过 `ascend_deepep.MegaMoe` 和
`ascend_deepep.ShmemInstance` 调用；真正执行时按需加载独立 MegaMoe
kernel 动态库。

测试 shape 放在 `test/megamoe_w8a8_test/mega_moe_cases.csv`，CSV 只包含
`M,N,K` 三列，其余测试参数统一在
`test/megamoe_w8a8_test/test_mega_moe.py` 中设置，不再区分 smoke 和
standard。CSV 可包含多行，脚本会依次执行全部 shape。测试脚本的第一个
参数是逗号分隔的 device ID 列表，EP 数量由 device 数量确定。脚本会设置
`ASCEND_RT_VISIBLE_DEVICES` 和
`SHMEM_UID_SESSION_ID`，然后为每个 rank 启动 Python 进程。
`mega_moe_w8a8_golden.py` 按照 MXFP8 dispatch、GMM1、SwiGLU、再次
MXFP8 量化、GMM2 和 combine 流程生成输入与 golden；Python 进程通过
`ascend_deepep.MegaMoe.compute()` 启动融合 kernel，同时通过
`mega_moe_w8a8_cascade.py` 执行 CANN 小算子级联。默认逐 rank 比较
融合输出、级联输出与 CPU golden：

```bash
bash test/megamoe_w8a8_test/test_mega_moe.sh 0,1
```

当前精度判定恢复为最初的融合算子输出与 CPU golden 比较，并使用 W4A8
风格的绝对误差保护：当
`abs(output - golden) < 0.01` 时直接使用绝对误差；超过 `0.01` 后，再以
`abs(output - golden) / max(abs(output), abs(golden), 2^-14 / 0.01)`
计算融合算子相对 golden 的偏离。偏离超过 0.01 的元素比例超过 5% 时测试
失败；每个 rank 都会额外打印偏移最大的元素索引、golden、融合输出、绝对
误差和偏移值。expert token 数要求与 golden 完全一致。小算子级联启动和三方偏离比较
代码保留为注释，后续可直接恢复。
`--absolute-error-threshold` 和 `--max-error-ratio` 可分别覆盖默认的 0.01
误差阈值和 5% 错误比例。保留的级联实现采用
`npu_moe_distribute_dispatch_v2`、
`npu_grouped_matmul_swiglu_quant_v2`、`npu_grouped_matmul` 和
`npu_moe_distribute_combine_v2`。通过
`--cascade-gmm1-mode split` 可以将中间阶段进一步拆成
`npu_grouped_matmul` 和 `cann_ops_nn.swiglu_group_quant`；该模式要求安装
`cann_ops_nn`。`--data-dir` 可指定输入、golden 和实际输出目录，
运行数据默认保留在日志打印的 `/tmp/mega_moe_python_test.*` 目录，也可通过
`--data-dir` 指定固定目录。注释保留的级联路径使用独立 HCCL 通信域；
恢复该路径后，可通过 `--hccl-init-method` 修改初始化地址。

性能测试默认参数在 `test/megamoe_w8a8_test/test_mega_moe.py` 中设置，
支持通过 `--warmup`、
`--loops` 和 `--repeat` 覆盖。脚本输出各 rank 的平均耗时，
并在每个 shape 数据目录的 `python_results/performance.csv` 保存汇总结果。只运行性能、
不生成或比较 golden 时：

```bash
bash test/megamoe_w8a8_test/test_mega_moe.sh 0,1 --performance-only
```

也可以直接运行
`python3 test/megamoe_w8a8_test/test_mega_moe.py 0,1`，并通过
`--mode both|precision|performance`、`--shmem-ip-port`、`--csv-path`、
`--seed`、`--absolute-error-threshold` 和 `--max-error-ratio` 覆盖当前
融合算子与 golden 测试的默认参数。
`mega_moe_stub` 仍保留用于不经过 PyTorch 的底层调试。

### MegaMoe BF16

当前为训练接口迁移过渡版本：三个 BF16 变体的 raw 前向返回
`(y, expert_token_nums, training_saved_state)`。`MegaMoeBF16Train`、
`compute_train` 和各 backward 入口暂时明确报错，待其他实现者完成反向适配后启用。
字段布局、生命周期和待改调用点见
[前向持久状态与反向适配](docs/MegaMoe_BF16_前向持久Workspace第一版与反向适配.md)。

BF16 前向使用独立的 `ascend_deepep.MegaMoeBF16` 算子和
`libascend_deepep_mega_moe_bf16.so`，不会覆盖现有 W8A8/W4A8。构建方式：

```bash
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
```

该路径的输入、两段权重、Dispatch/Combine 通信、GMM 输出和 SwiGLU 输出均为
BF16；GMM 的 L0C 累加与 SwiGLU 内部运算使用 FP32。接口形状、workspace
公式和当前 shape 的逐项内存值见
`kernels/mega_moe_bf16/README.md`。

W4A8 使用单独目录和动态库，不覆盖现有 W8A8。注册路径的构建与测试为：

```bash
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_w4a8.sh -soc_type Ascend950
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh 0,1,2,3,4,5,6,7
```

Python 入口为 `ascend_deepep.MegaMoeW4A8`。原 W4A8 独立 launch 方式也已
保留，不加载 `_C.so`，直接链接专用 kernel 动态库：

```bash
cd test/megamoe_w4a8_test/megamoe_w4a8_independent_test
bash build.sh
bash test_mega_moe_w4a8.sh 0,1,2,3,4,5,6,7
```

W4A8 测试继续使用原工程的 `M,K,N,TOPK,expertPerRank,EP` CSV 和 FP4/E8M0
数据生成逻辑；每次运行一个 shape，输入、CPU golden、实际输出和各 rank
日志都会保留。

`test_allgather_matmul.py` 会将每个 rank 的输入 `a` 写入 `--exchange-dir`，再由各 rank 读取全部 `a_rank_*.pt` 构造 golden：

```python
a_gather = torch.cat(a_blocks, dim=0)
golden = a_gather @ b
```

默认会执行精度校验。可选参数：

```bash
--skip-check-output     # 跳过 golden 校验
--rtol 1e-3 --atol 1e-3
--keep-exchange-files   # 保留 a_rank_*.pt
--save-outputs          # 保存 out_rank_*.pt 和 golden_rank_*.pt
```

多进程场景下，所有 rank 使用相同的 `ASCEND_DEEPEP_SHMEM_IP_PORT`。测试脚本默认使用 `tcp://127.0.0.1:8899`，也可以显式指定：

```bash
python test/test_dispatch.py --shmem-ip-port tcp://127.0.0.1:26666
```

## Python API

```python
from ascend_deepep import ElasticBuffer

buffer = ElasticBuffer(
    group,
    num_bytes=2 ** 30,
    explicitly_destroy=True,
)

recv_x, recv_topk_idx, recv_topk_weights, handle = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
    num_max_tokens_per_rank=num_max_tokens_per_rank,
    expert_alignment=1,
)

(out,) = buffer.combine(
    expert_output,
    handle,
    topk_weights=topk_weights,
)

agmm_out = buffer.allgather_matmul(a, b)

buffer.barrier()
buffer.destroy()
```

## 添加算子

当前工程支持两种方式：自写 AscendC 算子，以及引入 CatCCOS 中已有的算子模板。

### 方式一：自加 AscendC 算子

适合自己实现 device kernel，或者先做一个轻量 baseline。

1. 在 `include/ascend_deepep/kernels/types.h` 增加该算子的 tiling/参数 ABI。
2. 在 `include/ascend_deepep/kernels/launch.h` 声明导出的 C ABI，例如：

   ```cpp
   extern "C" ASCEND_DEEPEP_EXPORT void my_op_kernel_do(...);
   ```

3. 在 `kernels/my_op.cpp` 中实现：

   - AscendC `__global__ __aicore__` kernel；
   - `my_op_kernel_do` host launcher，内部用 `kernel<<<block_dim, nullptr, stream>>>(...)` 启动 kernel。

4. 在 `CMakeLists.txt` 的 `ascend_deepep_add_kernel_library(... SOURCES ...)` 中加入 `kernels/my_op.cpp`。
5. 在 `csrc/launcher.h/.cpp` 增加 `LaunchMyOpKernel(...)`，调用 `my_op_kernel_do`。
6. 在 `csrc/ops/my_op.h/.cpp` 中实现 Python 侧算子逻辑：

   - tensor device/type/shape 校验；
   - 输出 tensor 分配；
   - tiling 构造；
   - 调用 `LaunchMyOpKernel(...)`。

7. 在 `csrc/elastic_buffer.cpp` 中增加一个薄转发方法。
8. 在 `csrc/elastic_buffer.h` 和 `csrc/pybind.cpp` 暴露 Python API。
9. 在 `setup.py` 和 `MANIFEST.txt` 中加入新的 `csrc/ops/my_op.cpp/.h` 以及 `kernels/my_op.cpp`。

### 方式二：引入 CatCCOS 算子

适合复用 `3rdparty/catccos/include/catccos/dgemm/kernel` 下的模板，例如当前的 `allgather_matmul`。

原则：

- 复用本工程已有的 `3rdparty/catlass` 和 `3rdparty/shmem`；
- 不在 runtime 中重复 `aclshmemx_init_attr` / `shmem_finalize`；
- 不直接依赖 `catccos/examples/*` 中的 host registry；
- 在 `kernels/*.cpp` 中本地定义需要的 CatCCOS config，引用 CatCCOS/CATLASS 的库级头文件。

接入步骤：

1. 在 `CMakeLists.txt` 的 kernel target 中加入 CatCCOS include：

   ```cmake
   3rdparty/catccos/include
   3rdparty/catccos/utils
   3rdparty/catccos/tools
   ```

   并按目标架构设置宏，例如 A2：

   ```cmake
   COMPILE_DEFINITIONS
       CATLASS_ARCH=2201
   ```

2. 在 `kernels/my_catccos_op.cpp` 中：

   - include CatCCOS/CATLASS 库级头；
   - 定义本工程需要的 CatCCOS config；
   - 构造 `DeviceOp::Arguments`；
   - 导出 `my_catccos_op_kernel_do(...)`。

3. C++ runtime 侧仍然按自加算子的方式接入：

   - `include/ascend_deepep/kernels/types.h`
   - `include/ascend_deepep/kernels/launch.h`
   - `csrc/launcher.*`
   - `csrc/ops/my_catccos_op.*`
   - `elastic_buffer` / `pybind`

4. SHMEM 资源从 `ShmemInstance` 获取：

   ```cpp
   shmem.SymmetricBuffer()
   shmem.FftsAddr()
   ```

   不要在 CatCCOS wrapper 中重新初始化或 finalize SHMEM。

当前 `allgather_matmul` 是这种方式的最小示例：

```text
kernels/allgather_matmul.cpp
csrc/ops/allgather_matmul.cpp
test/test_allgather_matmul.py
```
