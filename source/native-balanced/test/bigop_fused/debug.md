```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer
```

```bash
BENCH_LOCAL_MEM_BYTES=2147483648 NPROC=8 \
BENCH_HIDDEN=7168 BENCH_FFN=4096 BENCH_EXPERTS=32 BENCH_TOPK=16 \
V2_LAYOUT_TIMING=1 MEGAMOE_TRACE=1 \
bash test/bigop_fused/run_speed.sh 4096 3 1
```

```bash
BENCH_LOCAL_MEM_BYTES=2147483648 NPROC=8 \
BENCH_HIDDEN=7168 BENCH_FFN=4096 BENCH_EXPERTS=128 BENCH_TOPK=16 \
V2_LAYOUT_TIMING=1 MEGAMOE_TRACE=1 \
bash test/bigop_fused/run_speed.sh 4096 3 1
```

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
```
