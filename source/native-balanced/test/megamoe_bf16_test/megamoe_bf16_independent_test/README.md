# MegaMoe BF16 independent C++ test

This directory builds and runs MegaMoe BF16 without importing
`ascend_deepep`, loading `_C.so`, or using
`torch.classes.AscendDeepEP.MegaMoeBF16`.

The standalone executable links `libascend_deepep_mega_moe_bf16.so`
directly and calls `mega_moe_bf16_kernel_do`. Python is used only to generate
BF16 input/golden data and compare binary outputs; it is not part of the
operator launch path. The data utility requires `numpy`, but does not require
`torch`, `torch_npu`, `_C.so`, or an installed `ascend_deepep` package.

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

Build the independent UDMA variant without changing the legacy artifacts:

```bash
bash build_udma.sh
```

It installs `mega_moe_bf16_udma_stub` and its kernel library under
`install_udma/`.

## Test

```bash
bash test_mega_moe.sh 0,1
```

Run the UDMA variant through the same input, CPU-golden, and comparison flow:

```bash
bash test_mega_moe_udma.sh 0,1
```

Optional arguments include:

```bash
bash test_mega_moe.sh 0,1 \
  --csv-path test_shapes.csv \
  --warmup 0 --loops 1 --timeout 600 \
  --data-dir /tmp/mega_moe_bf16_independent
```

The CSV contains only `M,N,K`. `TOPK=6`, `expertPerRank=16`, a 1 GiB SHMEM
base floor, and other runner parameters are set by
`independent_data.py`. BF16 cases require `K % 256 == 0` and `N % 512 == 0`.
The independent generator only produces balanced round-robin routing, reserves
`align128(M * TOPK)` routed rows per rank for both BF16 variants, and validates
the generated routing against that capacity before writing runner inputs. The
UDMA standalone path adds its `maxRecvTokenNum * K * sizeof(BF16)` activation
area to that legacy floor before 2 MiB alignment.
Inputs, CPU golden files, C++ outputs, and per-rank logs are retained in the
reported data directory.

For the default `M=512,N=3072,K=5120` case, each rank has 720 MiB of BF16
expert weights. The runner allocates corresponding host and device buffers in
addition to the tiling-reported workspace and symmetric peer buffer.
