# MegaMoe W8A8 independent C++ test

This directory builds and runs MegaMoe without importing `ascend_deepep`,
loading `_C.so`, or using `torch.classes.AscendDeepEP.MegaMoe`.

The standalone executable links `libascend_deepep_mega_moe.so` directly and
calls `mega_moe_kernel_do`. Python is used only to generate CPU input/golden
data and compare binary outputs; it is not part of the operator launch path.
The data tools require `numpy` and CPU-capable `torch`, but do not require
`torch_npu`, `_C.so`, or an installed `ascend_deepep` package.

## Build

Source the CANN environment, then run:

```bash
bash build.sh
```

If `rt.h` is outside the detected CANN tree:

```bash
bash build.sh --runtime-include-dir /path/that/contains/rt.h
```

The isolated artifacts are installed under `install/bin` and `install/lib`.
The script does not run `pip install`, build `_C.so`, or require the common
`libascend_deepep_kernels.so` target.

## Test

```bash
bash test_mega_moe.sh 0,1
```

Optional arguments include:

```bash
bash test_mega_moe.sh 0,1 --csv-path test_shapes.csv --warmup 0 --loops 1 --timeout 600 --data-dir /tmp/mega_moe_independent
```

The CSV contains only `M,N,K`. `TOPK=6`, `expertPerRank=16`, local symmetric
memory size, and other runner parameters are set by `independent_data.py`.
Inputs, CPU golden files, C++ outputs, and per-rank logs are retained in the
reported data directory.
