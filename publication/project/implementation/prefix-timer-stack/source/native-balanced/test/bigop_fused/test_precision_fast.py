#!/usr/bin/env python3
"""BigOpV2 precision matrix using one persistent torchrun/HCCL/SHMEM session.

This is the fast development runner.  Unlike run_precision.sh, cases are
not isolated per case: a kernel hang or launch failure terminates the whole
matrix.  The shape-dependent MegaMoe handle is recreated per case so its
private symmetric workspace is cleared before a different layout is used.
"""

import gc
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu

import test_precision as precision
from fused import build_bigop_v2


ACLSHMEM_DATA_OP_UDMA = 0x08
LOCAL_MEM_BYTES = 1024 * 1024 * 1024


def stage_diag(case, stage: str) -> None:
    """Opt-in per-rank breadcrumbs; does not add device/distributed barriers."""
    if os.environ.get("PRECISION_STAGE_DIAG", "0") == "1":
        print(f"[fast diag rank{os.environ.get('RANK', '?')}] "
              f"{time.monotonic():.6f} case={case} {stage}", flush=True)


@dataclass(frozen=True)
class PrecisionCase:
    seq: int
    hidden: int
    ffn: int
    experts: int
    topk: int

    @classmethod
    def parse(cls, value: str) -> "PrecisionCase":
        fields = value.split(":")
        if len(fields) != 5:
            raise ValueError(
                f"bad case {value!r}; expected SEQ:HIDDEN:FFN:E:TOPK"
            )
        return cls(*(int(field) for field in fields))

    def validate(self, world: int) -> None:
        if self.hidden % 256 != 0 or (2 * self.ffn) % 512 != 0:
            raise ValueError(
                f"{self}: requires HIDDEN % 256 == 0 and 2*FFN % 512 == 0"
            )
        if self.experts % world != 0:
            raise ValueError(
                f"{self}: experts={self.experts} is not divisible by EP={world}"
            )
        if not 1 <= self.topk <= self.experts:
            raise ValueError(f"{self}: TOPK must be in [1, {self.experts}]")
        if not 1 <= self.seq <= 16384:
            raise ValueError(f"{self}: SEQ must be in [1, 16384]")

    def __str__(self) -> str:
        return f"{self.seq}:{self.hidden}:{self.ffn}:{self.experts}:{self.topk}"


def read_cases() -> list[PrecisionCase]:
    raw = os.environ.get("PRECISION_CASES", "").split()
    if not raw:
        raise ValueError("PRECISION_CASES is empty")
    return [PrecisionCase.parse(item) for item in raw]


def run_one_case(
    case: PrecisionCase,
    cfg: precision.TestConfig,
    v2,
) -> int:
    precision.configure_shape(
        case.seq, case.hidden, case.ffn, case.experts, case.topk
    )
    ctx = precision.V2Context(0, 0, None, None, None, 0)
    stage_diag(case, "setup begin")
    precision.setup(ctx, init_process_group=False)
    stage_diag(case, "setup returned")

    if ctx.rank == 0:
        print("[fast stage] precision golden(s) ...", flush=True)
    need_backward = cfg.test_mode in ("backward", "both")
    stage_diag(case, "golden begin")
    ref, golden, exact_golden, golden_mode = precision.build_precision_goldens(
        ctx, need_backward)
    torch_npu.npu.synchronize()
    stage_diag(case, "golden device done")
    if ctx.rank == 0:
        print(
            f"[fast stage] golden done; backward hard-gate={golden_mode}; "
            f"exact_bf16_drift_report={'on' if exact_golden is not None else 'off'}",
            flush=True,
        )

    precision_failures = 0
    runtime_error = False
    has_warning = False
    if ctx.rank == 0:
        print("[fast stage] fused forward ...", flush=True)
    stage_diag(case, "fused forward begin")
    x_v, y_v2 = precision.run_v2_forward(v2, ctx)
    stage_diag(case, "fused forward returned; device sync begin")
    torch_npu.npu.synchronize()
    stage_diag(case, "fused forward device done")
    if ctx.rank == 0:
        print("[fast stage] fused forward done", flush=True)
    if cfg.test_mode in ("forward", "both"):
        if not precision.report_forward(ctx, y_v2, ref):
            precision_failures += 1

    if cfg.test_mode in ("backward", "both"):
        if ctx.rank == 0:
            print("[fast stage] fused backward ...", flush=True)
        try:
            stage_diag(case, "fused backward begin")
            got = precision.run_v2_backward(y_v2, x_v, ctx)
            stage_diag(case, "fused backward returned")
        except Exception as exc:
            if cfg.fused_mode == "train":
                print(
                    f"[rank{ctx.rank}] [ERROR] fused backward raised: {exc}",
                    flush=True,
                )
                runtime_error = True
            else:
                print(
                    f"[rank{ctx.rank}] [SKIP] backward unavailable ({exc}) -- "
                    "forward-only operator has no autograd graph",
                    flush=True,
                )
        else:
            if ctx.rank == 0:
                print("[fast stage] fused backward done", flush=True)
            stage_diag(case, "backward report begin")
            backward_ok, backward_warning = precision.report_backward(ctx, got, golden)
            precision.report_mxfp8_drift(ctx, golden, exact_golden)
            stage_diag(case, "backward report done")
            has_warning |= backward_warning
            if not backward_ok:
                precision_failures += 1

    stage_diag(case, "case device sync begin")
    torch_npu.npu.synchronize()
    stage_diag(case, "case device sync done")
    if runtime_error:
        return 3
    if precision_failures:
        return 2
    return 1 if has_warning else 0


def main() -> int:
    from ascend_deepep import ShmemInstance
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    cfg = precision.read_config()
    cases = read_cases()
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    for case in cases:
        case.validate(world)

    torch.npu.set_device(rank % torch.npu.device_count())
    dist.init_process_group(backend="hccl", world_size=world, rank=rank)

    shmem = None
    failed_cases = []
    warning_cases = []
    runtime_error_cases = []
    try:
        shmem = ShmemInstance(rank, world, cfg.ip_port)
        shmem.set_data_op_engine_type(ACLSHMEM_DATA_OP_UDMA)
        shmem.init(LOCAL_MEM_BYTES)

        if rank == 0:
            print(
                f"[fast] one torchrun/HCCL/SHMEM session: EP={world}, "
                f"cases={len(cases)}, test={cfg.test_mode}, "
                f"fused={cfg.fused_mode}; MegaMoe handle is reset per shape",
                flush=True,
            )

        for index, case in enumerate(cases, start=1):
            stage_diag(case, "case entry HCCL barrier begin")
            dist.barrier()
            stage_diag(case, "case entry HCCL barrier done")
            if rank == 0:
                print(f"\n[fast case {index}/{len(cases)}] {case}", flush=True)

            # MegaMoeBF16Train owns a 1 GiB aclshmem_malloc allocation whose
            # internal offsets depend on K/N/EPR/max_recv.  Reusing the same
            # handle across unlike shapes can reinterpret stale data under a
            # new layout.  Recreate only this cheap handle; torchrun, HCCL and
            # the ShmemInstance remain persistent.
            stage_diag(case, "handle create begin")
            v2 = build_bigop_v2(shmem, LOCAL_MEM_BYTES, mode=cfg.fused_mode)
            stage_diag(case, "handle create done")
            try:
                local_status = run_one_case(case, cfg, v2)
                status_tensor = torch.tensor(
                    [local_status], dtype=torch.int32, device="npu"
                )
                stage_diag(case, "status all_reduce begin")
                dist.all_reduce(status_tensor, op=dist.ReduceOp.MAX)
                case_status_value = int(status_tensor.item())
                stage_diag(case, "status all_reduce done")
                case_runtime_error = case_status_value == 3
                case_failed = case_status_value == 2
                case_warning = case_status_value == 1
                if case_runtime_error:
                    runtime_error_cases.append(str(case))
                elif case_failed:
                    failed_cases.append(str(case))
                elif case_warning:
                    warning_cases.append(str(case))

                # Finish every device/SHMEM consumer before freeing this
                # shape's private symmetric workspace.
                torch_npu.npu.synchronize()
                stage_diag(case, "cleanup SHMEM barrier begin")
                shmem.barrier()
                stage_diag(case, "cleanup SHMEM barrier done; HCCL barrier begin")
                dist.barrier()
                stage_diag(case, "cleanup HCCL barrier done")
            finally:
                stage_diag(case, "handle destroy begin")
                v2.destroy()
                stage_diag(case, "handle destroy done")

            if rank == 0:
                case_status = ("ERROR" if case_runtime_error else
                               ("Failed" if case_failed else
                                ("WARNING" if case_warning else "SUCCESS")))
                print(
                    f"[fast case {index}/{len(cases)}] "
                    f"{case_status} {case}",
                    flush=True,
                )
            del status_tensor
            gc.collect()
            torch.npu.empty_cache()
            stage_diag(case, "case cleanup done")
    finally:
        if shmem is not None:
            shmem.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        if runtime_error_cases:
            print(f"[fast summary] ERROR: {' '.join(runtime_error_cases)}", flush=True)
        elif failed_cases:
            print(f"[fast summary] Failed: {' '.join(failed_cases)}", flush=True)
        elif warning_cases:
            print(f"[fast summary] WARNING: {' '.join(warning_cases)}", flush=True)
        else:
            print(f"[fast summary] ALL {len(cases)} CASES SUCCESS", flush=True)
    # Keep the worker at zero for numerical Failed/WARNING so the shell wrapper
    # can print the complete matrix. run_precision_fast.sh converts a final
    # Failed summary to exit 2; execution errors remain nonzero here.
    return 1 if runtime_error_cases else 0


if __name__ == "__main__":
    sys.exit(main())
