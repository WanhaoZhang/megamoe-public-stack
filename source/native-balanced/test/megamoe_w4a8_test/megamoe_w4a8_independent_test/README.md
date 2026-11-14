# MegaMoe W4A8 independent test

This path keeps the original standalone launch mode after the W4A8 kernel is
migrated into DeepEP. The executable links the dedicated W4A8 kernel library
and calls `mega_moe_w4a8_kernel_do` directly; it does not import
`ascend_deepep`, load `_C.so`, or use the Torch class registry.

The migrated CPU generator preserves the W4A8 FP4 packing, E8M0 scale layout
and golden calculation. By default it follows the HaiSi fast path: generate
packed E2M1 codes directly, use unit scales, and decode the same codes for the
CPU golden. Set `MEGA_MOE_W4A8_FAST_RANDOM_WEIGHT=0` to restore the legacy
FP32-to-MXFP4 preprocessing path, or set `MEGA_MOE_W4A8_RANDOM_SEED` to change
the default seed `831`. The generator requires CPU `torch`, `numpy`,
`ml_dtypes`, `en_dtypes`, `packaging` and `psutil`.

```bash
source /path/to/CANN/set_env.sh
bash build.sh
bash test_mega_moe_w4a8.sh 0,1,2,3,4,5,6,7
```

The CSV format remains `M,K,N,TOPK,expertPerRank,EP`. The original generator
handles one case per invocation, so use a separate CSV/run for each shape.
Build products remain under `install/`; input, output, golden and rank logs are
retained under `output/` by default.

Pass `--enable-timer` to allocate the migrated AscendTimerV2 buffer. It writes
`rank_<rank>_npu_time.csv`; the test script then generates
`rank_<rank>_npu_time_parsed.csv` with per-core fixed-stage and per-expert
timings. Without that flag, the runner passes a null timer pointer and retains
the original independent-test behavior.

The standalone runner defaults to `--warmup 10 --loops 100` in
`test_mega_moe_w4a8.sh`. It reports the average end-to-end time
(kernel + `aclrtSynchronizeStream` + `aclshmem_barrier_all`) over the
`--loops` measured loops, excluding warmup. The warmup count is sized to
consume the cold-start outliers (the first few launches are typically
larger) so they do not enter the timed average. Override with
`--warmup`, `--loops`.
