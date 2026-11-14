# BigOp fused 16P / 32P / 64P 多机运行

`scripts/batch_launch.sh` 位于 `train_zwh` 分支。每台机器使用 8 张卡：

- 2 机 × 8 卡 = 16P
- 4 机 × 8 卡 = 32P
- 8 机 × 8 卡 = 64P

下面的 `NODE*_IP` 只是假名，运行前替换成实际节点 IPv4；文档不包含真实 IP。

## 1. 代码准备

在控制机执行，代码目录需要被所有节点以相同路径访问：

```bash
export SHARE_ACCOUNT="REPLACE_WITH_ACCOUNT"
export CODE_DIR="/mnt/share/${SHARE_ACCOUNT}/ascend_deepep_hkq"

cd /mnt/share/${SHARE_ACCOUNT}
git clone --branch train_zwh --recursive "REPLACE_WITH_REPO_URL" ascend_deepep_hkq
cd "$CODE_DIR"
git submodule update --init --recursive
```

如果代码已经 clone：

```bash
git -C "$CODE_DIR" switch train_zwh
git -C "$CODE_DIR" submodule update --init --recursive
```

## 2. 公共环境变量

先设置一次：

```bash
cd "$CODE_DIR"

export CODE_DIR="/mnt/share/${SHARE_ACCOUNT}/ascend_deepep_hkq"
export REMOTE_USER="REPLACE_WITH_REMOTE_USER"
export SSH_AUTH=key

export ASCEND_ENV="/path/to/cann/set_env.sh"
export CONDA_BASE="/path/to/anaconda3"
export CONDA_ENV=megamoe_zwh
export PRELOAD_LIBSTDCXX=1

export NPROC=8
export SEQ=2048
export WARMUP=3
export ITERS=10
export BENCH_HIDDEN=7168
export BENCH_LOCAL_MEM_BYTES=2147483648
export MEGAMOE_SAVE_GMM1_OUTPUT=none
export BIGOP_TIMING=1

# 复用各节点已有代码和构建产物时使用这组配置
export SYNC_GIT=0
export ALLOW_SHA_MISMATCH=1

# 默认网卡；特殊节点单独在 GLOO_IFNAME_MAP 中覆盖
export GLOO_IFNAME_DEFAULT="REPLACE_WITH_DEFAULT_INTERFACE"
export GLOO_IFNAME_MAP=""
```

首次构建时去掉 `--skip-build`，例如：

```bash
bash scripts/batch_launch.sh --prepare-only 2
```

后续已有构建产物时使用 `--skip-build`。

## 3. 16P：2 机

```bash
export NODE0_IP="REPLACE_WITH_NODE0_IPV4"
export NODE1_IP="REPLACE_WITH_NODE1_IPV4"
export HOST_ORDER="${NODE0_IP},${NODE1_IP}"

# 先确认 cards=16、ranks=0-7 和 8-15
bash scripts/batch_launch.sh --plan 2
```

Shape A：E=256、TopK=6：

```bash
BENCH_FFN=3072 BENCH_EXPERTS=256 BENCH_TOPK=6 \
bash scripts/batch_launch.sh --skip-build 2
```

Shape B：E=512、TopK=8：

```bash
BENCH_FFN=3072 BENCH_EXPERTS=512 BENCH_TOPK=8 \
bash scripts/batch_launch.sh --skip-build 2
```

## 4. 32P：4 机

```bash
export NODE0_IP="REPLACE_WITH_NODE0_IPV4"
export NODE1_IP="REPLACE_WITH_NODE1_IPV4"
export NODE2_IP="REPLACE_WITH_NODE2_IPV4"
export NODE3_IP="REPLACE_WITH_NODE3_IPV4"
export HOST_ORDER="${NODE0_IP},${NODE1_IP},${NODE2_IP},${NODE3_IP}"

# 先确认 cards=32、ranks=0-7、8-15、16-23、24-31
bash scripts/batch_launch.sh --plan 4
```

运行：

```bash
BENCH_FFN=3072 BENCH_EXPERTS=256 BENCH_TOPK=6 \
bash scripts/batch_launch.sh --skip-build 4
```

或：

```bash
BENCH_FFN=3072 BENCH_EXPERTS=512 BENCH_TOPK=8 \
bash scripts/batch_launch.sh --skip-build 4
```

## 5. 64P：8 机

```bash
export NODE0_IP="REPLACE_WITH_NODE0_IPV4"
export NODE1_IP="REPLACE_WITH_NODE1_IPV4"
export NODE2_IP="REPLACE_WITH_NODE2_IPV4"
export NODE3_IP="REPLACE_WITH_NODE3_IPV4"
export NODE4_IP="REPLACE_WITH_NODE4_IPV4"
export NODE5_IP="REPLACE_WITH_NODE5_IPV4"
export NODE6_IP="REPLACE_WITH_NODE6_IPV4"
export NODE7_IP="REPLACE_WITH_NODE7_IPV4"
export HOST_ORDER="${NODE0_IP},${NODE1_IP},${NODE2_IP},${NODE3_IP},${NODE4_IP},${NODE5_IP},${NODE6_IP},${NODE7_IP}"

# 先确认 cards=64、8 个节点各分配 8 个 rank
bash scripts/batch_launch.sh --plan 8
```

如果 `NODE4` 的网卡名不是默认值：

```bash
export GLOO_IFNAME_MAP="${NODE4_IP}=eth2"
```

Shape A：

```bash
BENCH_FFN=3072 BENCH_EXPERTS=256 BENCH_TOPK=6 \
bash scripts/batch_launch.sh --skip-build 8
```

Shape B：

```bash
BENCH_FFN=3072 BENCH_EXPERTS=512 BENCH_TOPK=8 \
bash scripts/batch_launch.sh --skip-build 8
```

64P 当前配置使用 `SEQ=2048`，不要直接调大。

## 6. 运行顺序

每种规模都按这个顺序执行：

```bash
# 1. 设置 HOST_ORDER
# 2. 检查计划
bash scripts/batch_launch.sh --plan 2   # 或 4 / 8

# 3. 确认 cards、rank 和网卡无误后，去掉 --plan 正式运行
bash scripts/batch_launch.sh --skip-build 2   # 或 4 / 8
```

控制端日志默认在 `./launch_logs_<RUN_ID>/`，每个节点的性能日志在共享代码目录下的 `test/bigop_fused/`。
