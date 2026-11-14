# MegaMoe W4A8 in DeepEP

Two launch paths are intentionally retained:

- `test_mega_moe_w4a8.sh` exercises the registered
  `ascend_deepep.MegaMoeW4A8` class and DeepEP SHMEM workspace lifecycle.
- `megamoe_w4a8_independent_test/` builds and launches the same device kernel
  through a standalone C++ executable, without the DeepEP Python extension.

Build the registered path with `bash scripts/build_mega_moe_w4a8.sh`, then run
`bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh 0,1,2,3,4,5,6,7`.
See the independent-test README for the standalone build and run commands.

## Selecting a case from `test_shapes.csv`

`test_shapes.csv` may contain multiple rows (one per shape). The registered
test runs **one case per invocation**; select the row with `--case-index N`
(0-based, default `0`):

```bash
# Run the first row (M=64 by default)
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh 0,1,2,3,4,5,6,7

# Run the fourth row (0-based index 3, M=512 with the shipped CSV)
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --case-index 3
```

The case actually run is printed as `[INFO] case_index=... case=...` at
startup so you can confirm which row was selected.

## Fast W4 weight generation

The precision generator now defaults to the same fast-weight strategy as the
HaiSi test: it generates valid packed E2M1 storage codes directly and sets all
MX block scales to 1. The CPU golden decodes those exact codes, so the NPU
weight input and golden remain consistent while the expensive full-FP32 weight
construction and FP32-to-MXFP4 LUT quantization are avoided.

The default seed is `831`. Override it with `MEGA_MOE_W4A8_RANDOM_SEED`, or
temporarily restore the legacy FP32-to-MXFP4 generator for A/B comparison:

```bash
MEGA_MOE_W4A8_RANDOM_SEED=1234 \
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --case-index 3

MEGA_MOE_W4A8_FAST_RANDOM_WEIGHT=0 \
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --case-index 3
```

This validates the MegaMoe kernel with already-quantized W4 inputs. The legacy
mode is still needed only when the test itself must cover the host-side
FP32-to-MXFP4 preprocessing algorithm or its original weight distribution.

## Skipping the CPU golden calculation

Pass `--skip-golden` to regenerate runtime inputs and quantized W4 weights
without running the CPU Dispatch/GMM1/SwiGLU/GMM2/Combine golden calculations.
The registered kernel still runs, but the final precision comparison is
skipped:

```bash
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --skip-golden
```

## Fast performance testing with random data (`--dummy-data`)

The CPU golden generator still performs Dispatch/GMM1/SwiGLU/GMM2/Combine and
can be expensive for large M. For performance testing — where numerical
accuracy is irrelevant — pass `--dummy-data` to generate random bytes with the
correct dtype/shape instead.
`input_expert_ids` is constrained to `[0, moe_expert_num)` to avoid kernel
out-of-bounds. This skips both the W4 quantization generator and the CPU
golden calculation, and disables the precision comparison:

```bash
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --dummy-data --reuse-data --case-index 3
```

`--dummy-data` writes the same `rank{rank}_{name}_w4a8_{M}_{K}_{N}_{EP}.bin`
files as the real generator, so it composes with `--reuse-data`: the first
run generates random data, subsequent runs reuse it without regenerating.

## Using externally generated expert ids

Use `--expert-ids-source` to replace only `input_expert_ids` while keeping the
DeepEP test's other inputs and weights unchanged. The source may be:

- a `.pt`/`.pth` tensor with shape `[EP, M, TOPK]` or `[EP*M, TOPK]`;
- a dict containing that tensor under `expert_ids`, or a list of `EP` rank tensors;
- one raw int32 `.bin` with shape `[EP, M, TOPK]`;
- a directory containing one `rank{rank}_input_expert_ids*.bin` per rank.

A HaiSi `golden_cache/<key>/input_data.pt` path is also accepted directly when
`gen_input_and_golden.py` exists above it in the same source tree. This loads
the complete cache into host memory, so the tensor-only export below is the
recommended form.

After HaiSi creates `input_data`, export only its expert ids:

```python
torch.save(
    torch.stack([value.cpu().to(torch.int32) for value in input_data.expert_ids]),
    "/tmp/haisi_expert_ids.pt",
)
```

Then run the registered DeepEP path with the same `M`, `TOPK`, `EP`, and total
expert count:

```bash
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --case-index 3 --reuse-data \
    --expert-ids-source /tmp/haisi_expert_ids.pt
```

The loader validates shape, integer dtype, and the range
`[0, EP*expertPerRank)`, converts the data to the kernel's int32 layout, and
prints the minimum and maximum expert token counts. Because changing the route
invalidates the locally generated output reference, this option automatically
skips CPU golden generation and precision comparison.

### Balanced expert ids (`--balanced-ids`, default)

With `--dummy-data` the MoE dispatch stage routes tokens to experts based on
`input_expert_ids`. The unbalanced path builds hot, warm, globally cold, and
rank-specific cold expert weights, then
samples TOPK experts without replacement. This intentionally creates a
long-tail workload and can make measured time unstable across experts.

`--balanced-ids` (default on with `--dummy-data`) makes each expert receive
an equal share of tokens: a global stream of `M*EP*TOPK` ids is built so each
of the `moe_expert_num` experts appears the same number of times (differing
by at most 1 when the total does not divide evenly), then sliced per rank.
The algorithm mirrors `gen_input_and_golden_w4a8.py`'s `is_topk_balance`
path: each round emits `arange(r, r+N) % N` (a permutation shifted by 1 each
round to avoid predictability), rounds are concatenated and truncated to the
target length. With the shipped `test_shapes.csv` (M=64, TOPK=6, 256 experts)
every expert gets exactly 6 tokens.

```bash
# Default: balanced ids (stable, comparable perf numbers)
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --dummy-data --reuse-data --case-index 3

# Reproduce the hot/warm/cold unbalanced distribution
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --dummy-data --no-balanced-ids --case-index 3
```

## Timer collection (`--enable-timer`)

Pass `--enable-timer` to the registered test to call `compute_with_timer()`.
Without an explicit `--data-dir`, raw `rank_<rank>_npu_time.csv` files are
written under `output/timer_registered`:

```bash
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --enable-timer
```

The registered test defaults to `--warmup 20 --loops 100` in
`test_mega_moe_w4a8.py`. It reports the average synchronized wall-clock time
of the `--loops` measured loops, excluding warmup. The warmup count is sized
to consume the cold-start outliers (the first few launches are typically
larger) so they do not enter the timed average. Registered output and raw
timer CSV files contain data from the last measured loop only.

Override the timing parameters, reuse an existing data directory, or skip the
CPU golden calculation:

```bash
bash test/megamoe_w4a8_test/test_mega_moe_w4a8.sh \
    0,1,2,3,4,5,6,7 --warmup 30 --loops 200 --reuse-data --skip-golden
```

Parse the default eight-rank timer output with:

```bash
python3 test/megamoe_w4a8_test/parse_timer_csv.py
```

Parsed CSV files are written under `output/timer_registered/timer`. Convert
them to per-rank and merged Chrome Trace JSON under
`output/timer_registered/timer_json` with:

```bash
python3 test/megamoe_w4a8_test/trace_parser.py
```

The original `compute()` API remains unchanged and passes a null timer pointer.

## msprof performance collection (`run_perf.py`)

`run_perf.py` collects kernel-level performance through `msprof` instead of
the in-process wall-clock timer. For each case it:

1. Runs `msprof --application="bash test_mega_moe_w4a8.sh ... --case-index N"
   --output=./prof`.
2. Locates the latest `./prof/PROF_*/mindstudio_profiler_output/task_time_*.csv`.
3. Filters rows whose `kernel_name` contains `mega_moe_w4a8`, sorts by
   `task_start`, drops the first `--warmup` samples (cold-start), and
   averages the next `--loops` `task_time(us)` values.
4. Appends the `test_shapes.csv` columns plus a `perf_avg_us` column to
   `perf_result.csv`.

By default `run_perf.py` iterates over **all** rows of `--shapes-csv` and
defaults to `--dummy-data` for fast random data. Use `--case-index N` to run
a single row:

```bash
# Iterate over every case in test_shapes.csv (default: --dummy-data)
python3 test/megamoe_w4a8_test/run_perf.py

# Collect performance for a single case (0-based index 3)
python3 test/megamoe_w4a8_test/run_perf.py --case-index 3

# Custom devices / warmup / loops
python3 test/megamoe_w4a8_test/run_perf.py --devices 0,1,2,3 \
    --warmup 20 --loops 100 --case-index 3

# Use fast W4 inputs plus the CPU golden instead of dummy data
python3 test/megamoe_w4a8_test/run_perf.py --no-dummy-data --case-index 3

# Only parse an existing ./prof directory (requires --case-index,
# since multiple cases share one prof dir and would overwrite each other)
python3 test/megamoe_w4a8_test/run_perf.py --skip-run --case-index 3
```

Result file (`perf_result.csv`) columns are the `test_shapes.csv` headers
(`M,K,N,TOPK,expertPerRank,EP`) followed by `perf_avg_us`. The file is
appended to on every run; delete it to start a fresh table. `prof/` is
cleared before each `msprof` invocation so the latest `PROF_*` directory
always belongs to the current case.

All paths in `run_perf.py` are relative to the project root
(`ascend_deepep_hkq`); the script `chdir`s there on startup, so it can be
invoked from any directory.
