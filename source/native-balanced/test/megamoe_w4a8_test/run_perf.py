#!/usr/bin/env python3
"""MegaMoe W4A8 性能数据采集脚本。

流程（对每个 case 循环执行）：
  1. 通过 msprof 拉起 test_mega_moe_w4a8.sh 跑一轮性能测试，
     透传 --case-index 选中当前 case。
     默认用 --dummy-data 生成随机数据（性能测试不验证精度，跳过慢的 W4 量化）。
  2. 从 ./prof/<PROF_*>/mindstudio_profiler_output/task_time_*.csv 中
     过滤 kernel_name 含 mega_moe_w4a8 的行。
  3. 按 task_start 排序后去掉前 warmup 条，剩余 loops 条的 task_time(us)
     取平均，作为当前 case 的性能数据。
  4. 将 test_shapes.csv 的列 + 性能数据追加写入性能结果文件。

默认遍历 --shapes-csv 的所有行；用 --case-index N 只跑第 N 行（0-based）。
所有路径相对项目根（ascend_deepep_hkq），脚本会自动 chdir 到项目根。

用法示例：
    # 遍历 test_shapes.csv 的所有 case 采集性能（默认 --dummy-data --balanced-ids）
    python test/megamoe_w4a8_test/run_perf.py

    # 只采集某个 case 的性能（例如第 4 行，0-based 即 M=512）
    python test/megamoe_w4a8_test/run_perf.py --case-index 3

    # 指定 device 与 warmup/loops
    python test/megamoe_w4a8_test/run_perf.py --devices 0,1,2,3 \
        --warmup 20 --loops 100 --case-index 3

    # 复现不均衡分布（热点/温点/冷点 expert ids）
    python test/megamoe_w4a8_test/run_perf.py --no-balanced-ids --case-index 3

    # 用真实 W4 量化数据跑性能（慢，加 --no-dummy-data）
    python test/megamoe_w4a8_test/run_perf.py --no-dummy-data --case-index 3

    # 只解析已有 prof 数据，不重新跑 msprof（必须指定 --case-index）
    python test/megamoe_w4a8_test/run_perf.py --skip-run --case-index 3
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
from pathlib import Path


# 脚本位于 test/megamoe_w4a8_test/ 下，项目根 = 脚本父目录的父目录。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# 相对项目根的默认路径。
DEFAULT_SHAPES_CSV = "test/megamoe_w4a8_test/megamoe_w4a8_independent_test/test_shapes.csv"
DEFAULT_TEST_SCRIPT = "test/megamoe_w4a8_test/test_mega_moe_w4a8.sh"
DEFAULT_PROF_DIR = "./prof"
DEFAULT_RESULT_FILE = str(SCRIPT_DIR / "perf_result.csv")

KERNEL_KEYWORD = "mega_moe_w4a8"


def find_task_time_csv(prof_dir: Path) -> Path:
    """在 prof_dir 下选一个 PROF_* 目录，返回其 task_time_*.csv 路径。

    多个 PROF_* 目录时选最新修改的；同级有多个 task_time_*.csv 时选最新的。
    """
    prof_dirs = sorted(
        prof_dir.glob("PROF_*/mindstudio_profiler_output"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not prof_dirs:
        raise RuntimeError(
            f"no PROF_*/mindstudio_profiler_output found under {prof_dir}"
        )
    selected_prof = prof_dirs[0]
    task_csvs = sorted(
        selected_prof.glob("task_time_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not task_csvs:
        raise RuntimeError(
            f"no task_time_*.csv found under {selected_prof}"
        )
    print(f"[INFO] using profiler output: {task_csvs[0]}")
    return task_csvs[0]


def extract_kernel_times(task_csv: Path, keyword: str) -> list[float]:
    """读取 task_time CSV，返回 kernel_name 含 keyword 的 task_time(us) 列表。

    CSV 表头: Device_id,kernel_name,kernel_type,stream_id,task_id,
              task_time(us),task_start(us),task_stop(us)
    按task_start升序排序，保证去掉前warmup条是时间最早(含cold-start)的。
    """
    rows = []
    with task_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("kernel_name") or "").strip()
            if keyword not in name:
                continue
            try:
                t_time = float(row["task_time(us)"])
                # task_start 末尾可能带 tab，strip 处理。
                t_start = float((row.get("task_start(us)") or "0").strip().rstrip("\t"))
            except (ValueError, KeyError):
                continue
            rows.append((t_start, t_time))
    rows.sort(key=lambda x: x[0])
    return [t for _, t in rows]


def compute_perf(times: list[float], warmup: int, loops: int) -> float:
    """去掉前 warmup 条，取后续 loops 条的 task_time 平均值。"""
    total = warmup + loops
    if len(times) < total:
        raise RuntimeError(
            f"not enough {KERNEL_KEYWORD} samples: got {len(times)}, "
            f"need warmup({warmup}) + loops({loops}) = {total}"
        )
    steady = times[warmup:warmup + loops]
    return sum(steady) / len(steady)


def read_shapes_csv(shapes_csv: Path) -> tuple[list[str], list[dict]]:
    """读取 test_shapes.csv，返回 (表头列表, 每行dict列表)。"""
    with shapes_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows


def append_result(
    result_file: Path, headers: list[str], case_row: dict, perf_us: float
) -> None:
    """将 case_row 各列 + 性能列追加写入结果文件。

    表头 = shapes 表头 + ['perf_avg_us']。文件不存在时先写表头。
    """
    perf_col = "perf_avg_us"
    out_headers = list(headers) + [perf_col]
    write_header = not result_file.exists()
    # 文件已存在但表头不一致也重写表头。
    if not write_header:
        with result_file.open(newline="", encoding="utf-8") as f:
            existing = csv.reader(f)
            first = next(existing, None)
        if first != out_headers:
            write_header = True
    with result_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(out_headers)
        row = [case_row.get(h, "") for h in headers] + [f"{perf_us:.3f}"]
        writer.writerow(row)
    print(f"[INFO] appended result to {result_file}: {row}")


def run_one_case(
    args, headers: list[str], case_row: dict, case_index: int
) -> None:
    """对单个 case 执行：msprof 采集 → 解析 task_time → 追加结果。"""
    print(f"\n{'=' * 60}")
    print(f"[INFO] case_index={case_index} case={case_row}")
    print(f"{'=' * 60}")

    # Step 1: msprof 拉起测试脚本。
    if not args.skip_run:
        prof_dir = Path(args.prof_dir)
        # 每次采集前清空旧 prof，避免选到上一次的 PROF 目录。
        if prof_dir.exists():
            import shutil
            shutil.rmtree(prof_dir, ignore_errors=True)
        prof_dir.mkdir(parents=True, exist_ok=True)

        test_args = [
            f"bash {args.test_script} {args.devices}",
            f"--warmup {args.warmup}",
            f"--loops {args.loops}",
            f"--case-index {case_index}",
        ]
        if args.dummy_data:
            # 性能测试用随机数据，跳过慢的 W4 量化与 golden。
            # 配合 --reuse-data 可在数据已存在时直接复用，避免重复生成。
            test_args.append("--dummy-data --reuse-data")
            # 均匀 expert ids：每个 expert token 数相等，性能数据稳定可比。
            if args.balanced_ids:
                test_args.append("--balanced-ids")
            else:
                test_args.append("--no-balanced-ids")
        elif args.reuse_data:
            test_args.append("--reuse-data --skip-golden")
        else:
            test_args.append("--skip-golden")
        application = " ".join(test_args)
        msprof_cmd = [
            "msprof",
            f"--application={application}",
            f"--output={args.prof_dir}",
        ]
        print(f"[INFO] running: {' '.join(msprof_cmd)}")
        subprocess.run(msprof_cmd, check=True)

    # Step 2: 定位 task_time CSV。
    task_csv = find_task_time_csv(Path(args.prof_dir))

    # Step 3: 过滤 + 去前 warmup + 取 loops 平均。
    times = extract_kernel_times(task_csv, KERNEL_KEYWORD)
    print(f"[INFO] {KERNEL_KEYWORD} samples: {len(times)}")
    perf_us = compute_perf(times, args.warmup, args.loops)
    print(f"[PERF] case_index={case_index} warmup={args.warmup} loops={args.loops} "
          f"perf_avg_us={perf_us:.3f}")

    # Step 4: 追加写入结果文件。
    append_result(Path(args.result_file), headers, case_row, perf_us)
    print(f"[SUCCESS] case_index={case_index} performance data recorded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices",
        default="1,2,3,4",
        help="comma-separated device IDs passed to test_mega_moe_w4a8.sh",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--loops", type=int, default=100)
    parser.add_argument(
        "--shapes-csv",
        default=DEFAULT_SHAPES_CSV,
        help="test_shapes.csv path (relative to project root)",
    )
    parser.add_argument(
        "--test-script",
        default=DEFAULT_TEST_SCRIPT,
        help="test_mega_moe_w4a8.sh path (relative to project root)",
    )
    parser.add_argument(
        "--prof-dir",
        default=DEFAULT_PROF_DIR,
        help="msprof output dir (relative to project root)",
    )
    parser.add_argument(
        "--result-file",
        default=DEFAULT_RESULT_FILE,
        help="performance result CSV output path",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="skip msprof execution, only parse existing prof data",
    )
    parser.add_argument(
        "--dummy-data",
        dest="dummy_data",
        action="store_true",
        default=True,
        help="use random dummy data instead of the slow W4 quantization "
             "generator (default: enabled, recommended for perf testing)",
    )
    parser.add_argument(
        "--no-dummy-data",
        dest="dummy_data",
        action="store_false",
        help="disable dummy data and run the full W4 quantization generator",
    )
    parser.add_argument(
        "--balanced-ids",
        dest="balanced_ids",
        action="store_true",
        default=True,
        help=(
            "with --dummy-data: each expert gets an equal share of tokens "
            "(counts differ by at most 1) for stable comparable perf numbers "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--no-balanced-ids",
        dest="balanced_ids",
        action="store_false",
        help=(
            "with --dummy-data: use a hot/warm/cold weighted expert-id "
            "distribution"
        ),
    )
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="pass --reuse-data to the test script (reuse existing data)",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=None,
        help=(
            "0-based row index into --shapes-csv to run a single case; "
            "omit to iterate over all rows"
        ),
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    print(f"[INFO] working dir: {Path.cwd()}")

    shapes_csv = Path(args.shapes_csv)
    headers, rows = read_shapes_csv(shapes_csv)
    if not rows:
        raise RuntimeError(f"no data rows in {shapes_csv}")

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(rows):
            raise RuntimeError(
                f"--case-index {args.case_index} out of range; "
                f"CSV has {len(rows)} rows"
            )
        indices = [args.case_index]
    else:
        indices = list(range(len(rows)))
    print(f"[INFO] will run {len(indices)} case(s): {indices}")

    # --skip-run 模式下只能解析单个 case 的 prof 数据，
    # 因为多个 case 共用同一个 prof 目录会互相覆盖。
    if args.skip_run and len(indices) > 1:
        raise RuntimeError(
            "--skip-run can only parse one case; specify --case-index N"
        )

    for case_index in indices:
        run_one_case(args, headers, rows[case_index], case_index)

    print(f"\n[DONE] {len(indices)} case(s) processed; results: {args.result_file}")


if __name__ == "__main__":
    main()
