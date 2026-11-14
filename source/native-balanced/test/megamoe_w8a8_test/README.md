# MegaMoe W8A8 端到端测试

本目录验证融合 `ascend_deepep.MegaMoe` W8A8 路径，包括精度、性能、
小算子级联对比以及可选的 Timer V2 流水打点。默认用例来自
`mega_moe_cases.csv`，当前为 `M=512, N=3072, K=5120, TopK=6`；命令中
提供的设备数量即 EP 大小。

## 构建与普通测试

在仓库根目录加载 CANN 环境后执行：

```bash
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe.sh -soc_type Ascend950
bash test/megamoe_w8a8_test/test_mega_moe.sh 6,7
```

默认测试同时执行融合算子和小算子级联精度对比。当前级联路径中的
`MoeDistributeDispatch` 只接受 `M <= 512`；测试更大的 M 时应跳过级联：

```bash
bash test/megamoe_w8a8_test/test_mega_moe.sh 6,7 \
  --csv-path /tmp/w8a8_ep2.csv --skip-cascade
```

只测试性能时也不会执行级联路径：

```bash
bash test/megamoe_w8a8_test/test_mega_moe.sh 6,7 \
  --mode performance
```

## Timer V2 打点

Timer 是编译期开关，构建和测试必须同时带 `--enable-timer`：

```bash
bash scripts/build_mega_moe.sh \
  -soc_type Ascend950 --enable-timer

bash test/megamoe_w8a8_test/test_mega_moe.sh 6,7 \
  --mode precision --warmup 0 --loops 1 --enable-timer
```

如果测试带了 `--enable-timer`，但 kernel 未按 Timer 模式重新构建，Host
会明确报错，不会生成无效打点数据。重新执行不带 `--enable-timer` 的构建
命令会将 Timer 编译开关恢复为关闭状态。

## 输出目录

未指定 `--data-dir` 时，每次运行的输入、输出、Golden、性能和 Timer 文件
保存在：

```text
test/megamoe_w8a8_test/output/run_<时间>_<PID>/
```

Timer 原始数据、解析结果和流水 JSON 分别位于每个 Shape 目录下：

```text
python_results/rank_<rank>_npu_time.csv
python_results/timer/rank_<rank>_npu_time_parsed.csv
python_results/timer_json/
```

JSON 文件名包含 M、N、K、TopK、EP 和 W8A8 路径，例如：

```text
rank_0_npu_time_512_3072_5120_6_2_w8a8.json
all_rank_trace_512_3072_5120_6_2_w8a8.json
```

JSON 可使用 Perfetto 或 `chrome://tracing` 打开。独立 C++ 测试见
`megamoe_w8a8_independent_test/README.md`。
