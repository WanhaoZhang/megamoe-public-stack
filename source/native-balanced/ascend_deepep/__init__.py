import ctypes
import os
from pathlib import Path


_SHMEM_LIBS = (
    "libshmem_utils.so",
    "aclshmem_bootstrap_config_store.so",
    "libshmem.so",
    "aclshmem_bootstrap_uid.so",
)


def _loaded_lib_paths(soname: str) -> list[str]:
    """Absolute paths of ``soname`` already mapped into this process."""
    paths: list[str] = []
    try:
        lines = open("/proc/self/maps", encoding="utf-8", errors="replace")
    except OSError:
        return paths
    with lines:
        for line in lines:
            if soname not in line:
                continue
            path = line.split()[-1]
            if path.startswith("/") and path not in paths:
                paths.append(path)
    return paths


def _load_package_shmem_libs():
    lib_dir = Path(__file__).resolve().parent / "lib"
    required_libs = [
        "libshmem_utils.so",
        "aclshmem_bootstrap_config_store.so",
        "libshmem.so",
    ]
    optional_libs = [
        "aclshmem_bootstrap_uid.so",
        "libascend_deepep_mega_moe.so",
        "libascend_deepep_mega_moe_w4a8.so",
        "libascend_deepep_mega_moe_bf16.so",
        "libascend_deepep_mega_moe_bf16_mgroup.so",
        "libascend_deepep_mega_moe_bf16_udma.so",
    ]
    # MoonEP's buffer-backed heap is already this process's libshmem.so.
    # Opening the copy under ascend_deepep/lib by absolute path would create a
    # second image; kernel DT_NEEDED then resolves the SONAME to the one that
    # is already resident.
    share_existing_shmem = bool(_loaded_lib_paths("libshmem.so"))
    mode = getattr(os, "RTLD_NOW", 0) | getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in required_libs + optional_libs:
        if share_existing_shmem and lib_name in _SHMEM_LIBS:
            continue
        lib_path = lib_dir / lib_name
        if not lib_path.exists():
            if lib_name in required_libs:
                raise ImportError(f"required SHMEM shared library not found: {lib_path}")
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=mode)
        except OSError as exc:
            raise ImportError(f"failed to load SHMEM shared library {lib_path}: {exc}") from exc


_load_package_shmem_libs()

from . import _C as _C  # noqa: F401,E402

import torch

ShmemInstance = torch.classes.AscendDeepEP.ShmemInstance
Dispatch = torch.classes.AscendDeepEP.Dispatch
Combine = torch.classes.AscendDeepEP.Combine
AllGatherMatmul = torch.classes.AscendDeepEP.AllGatherMatmul
MegaMoe = torch.classes.AscendDeepEP.MegaMoe
MegaMoeBF16 = torch.classes.AscendDeepEP.MegaMoeBF16
MegaMoeBF16MGroup = torch.classes.AscendDeepEP.MegaMoeBF16MGroup
MegaMoeBF16UDMA = torch.classes.AscendDeepEP.MegaMoeBF16UDMA
MegaMoeW4A8 = torch.classes.AscendDeepEP.MegaMoeW4A8

__all__ = [
    "ShmemInstance",
    "Dispatch",
    "Combine",
    "AllGatherMatmul",
    "MegaMoe",
    "MegaMoeBF16",
    "MegaMoeBF16MGroup",
    "MegaMoeBF16Train",
    "MegaMoeBF16UDMA",
    "MegaMoeW4A8",
]


class _MegaMoeBF16TrainFunction(torch.autograd.Function):
    """Internal autograd bridge for the MegaMoe BF16 kernels.

    forward:  y, counts = f(op, shmem, x, topk_ids, topk_weights, w1, w2)
    backward: computes grad_x, grad_topk_weights, grad_weight1, grad_weight2.
    topk_ids is discrete and remains non-differentiable.
    """

    @staticmethod
    def forward(
        ctx,
        op,
        shmem,
        x,
        topk_ids,
        topk_weights,
        w1,
        w2,
        max_recv,
        swiglu_limit,
        save_gmm1_output,
        retain_dispatch_output,
        diagnostic_holder,
    ):
        if os.environ.get("ASCEND_DEEPEP_DEBUG_PTR", "0") == "1":
            print(
                f"[MegaMoe-fwd] op={id(op)} w1_ptr={w1.data_ptr()} w2_ptr={w2.data_ptr()} "
                f"w1_dtype={w1.dtype} w1_is_shared_with_param={w1._base is not None} "
                f"x_ptr={x.data_ptr()} max_recv={max_recv} swiglu_limit={swiglu_limit}",
                flush=True,
            )
        capture_forward_timer = (
            diagnostic_holder is not None
            and diagnostic_holder.get("capture_forward_timer", False)
        )
        if capture_forward_timer:
            (
                y,
                counts,
                training_saved_state,
                dispatch_output,
                forward_transient_workspace,
                forward_timer,
            ) = op.compute_train_with_timer(
                shmem, x, topk_ids, topk_weights, w1, w2, max_recv,
                float(swiglu_limit), save_gmm1_output
            )
            diagnostic_holder["forward_timer"] = forward_timer
        else:
            (y, counts, training_saved_state, dispatch_output,
             forward_transient_workspace) = (
                op.compute_train(
                    shmem, x, topk_ids, topk_weights, w1, w2, max_recv,
                    float(swiglu_limit), save_gmm1_output
                )
            )
        # torch.classes instances do not support arbitrary Python attributes.
        # Pass the saved context to diagnostic code through an explicit holder.
        if diagnostic_holder is not None:
            diagnostic_holder["forward_transient_workspace"] = (
                forward_transient_workspace
            )
            diagnostic_holder["training_saved_state"] = training_saved_state
            diagnostic_holder["dispatch_output"] = dispatch_output
        ctx.op = op
        ctx.shmem = shmem
        ctx.max_recv = max_recv
        ctx.swiglu_limit = float(swiglu_limit)
        ctx.save_gmm1_output = save_gmm1_output
        ctx.retain_dispatch_output = bool(retain_dispatch_output)
        # CPU-side mirror of the saved-value format consumed by backward.
        from ctypes import c_float
        disabled_limit = float.fromhex("0x1.fffffep+127")
        normalized_limit = c_float(swiglu_limit).value if swiglu_limit > 0 else disabled_limit
        resolved_mode = save_gmm1_output
        if resolved_mode == "auto":
            resolved_mode = "mxfp8" if normalized_limit == disabled_limit else "bf16"
        has_clip_masks = resolved_mode == "mxfp8" and normalized_limit != disabled_limit
        ctx.saved_gate_up_metadata = {
            "layout_version": 2 if has_clip_masks else 1,
            "clip_mask_kind": "three_bitplanes_v1" if has_clip_masks else "none",
            "mode": resolved_mode, "normalized_limit": normalized_limit,
        }
        if diagnostic_holder is not None:
            diagnostic_holder["saved_gate_up_metadata"] = ctx.saved_gate_up_metadata.copy()
        ctx.diagnostic_holder = diagnostic_holder
        # Dispatch output is independently allocated. Retain it only for a
        # backward implementation that consumes packed X/TopK weights.
        if ctx.retain_dispatch_output:
            ctx.save_for_backward(
                topk_weights, w1, w2, counts, training_saved_state,
                dispatch_output
            )
        else:
            ctx.save_for_backward(
                topk_weights, w1, w2, counts, training_saved_state
            )
        ctx.mark_non_differentiable(counts)
        return y, counts

    @staticmethod
    def backward(ctx, grad_y, grad_counts=None):
        if not ctx.retain_dispatch_output:
            raise RuntimeError(
                "the current MegaMoe BF16 backward consumes retained "
                "dispatch_output; set retain_dispatch_output=True until the "
                "redispatching backward implementation is integrated"
            )
        (
            topk_weights,
            w1,
            w2,
            counts,
            training_saved_state,
            dispatch_output,
        ) = ctx.saved_tensors
        if os.environ.get("ASCEND_DEEPEP_DEBUG_PTR", "0") == "1":
            print(
                f"[MegaMoe-bwd] op={id(ctx.op)} w1_ptr={w1.data_ptr()} w2_ptr={w2.data_ptr()} "
                f"w1_dtype={w1.dtype} grad_y_ptr={grad_y.data_ptr()} "
                f"grad_y_dtype={grad_y.dtype} grad_y_contig={grad_y.is_contiguous()} "
                f"saved_ptr={training_saved_state.data_ptr()}",
                flush=True,
            )
        if os.environ.get("ASCEND_DEEPEP_BWD_SYNC", "0") == "1":
            try:
                cur = torch.npu.current_stream()
                print(
                    f"[MegaMoe-bwd] stream={cur.npu_stream if hasattr(cur, 'npu_stream') else cur} "
                    f"syncing before fused backward",
                    flush=True,
                )
            except Exception:  # pragma: no cover - diagnostic only
                pass
            torch.npu.synchronize()
        capture_workspace = (
            ctx.diagnostic_holder is not None
            and ctx.diagnostic_holder.get("capture_backward_workspace", False)
        )
        capture_timer = (
            ctx.diagnostic_holder is not None
            and ctx.diagnostic_holder.get("capture_backward_timer", False)
        )
        if capture_timer:
            dx, dtopk, dw1, dw2, timer = ctx.op.backward_with_timer(
                ctx.shmem,
                grad_y.contiguous(),
                topk_weights,
                w1,
                w2,
                counts,
                training_saved_state,
                dispatch_output,
                ctx.max_recv,
                ctx.swiglu_limit,
                ctx.save_gmm1_output,
            )
            ctx.diagnostic_holder["backward_timer"] = timer
        elif capture_workspace:
            dx, dtopk, dw1, dw2, backward_workspace = ctx.op.backward_debug(
                ctx.shmem,
                grad_y.contiguous(),
                topk_weights,
                w1,
                w2,
                counts,
                training_saved_state,
                dispatch_output,
                ctx.max_recv,
                ctx.swiglu_limit,
                ctx.save_gmm1_output,
            )
            ctx.diagnostic_holder["backward_workspace"] = backward_workspace
        else:
            dx, dtopk, dw1, dw2 = ctx.op.backward(
                ctx.shmem,
                grad_y.contiguous(),
                topk_weights,
                w1,
                w2,
                counts,
                training_saved_state,
                dispatch_output,
                ctx.max_recv,
                ctx.swiglu_limit,
                ctx.save_gmm1_output,
            )
        # Diagnostic: the fused backward runs on a device stream that overlaps
        # with the NEXT micro-batch's fused forward; both use the same SHMEM
        # window, so without a barrier they race and fault with acl 507015.
        if os.environ.get("ASCEND_DEEPEP_BWD_SYNC", "0") == "1":
            torch.npu.synchronize()
        return (None, None, dx, None, dtopk, dw1, dw2, None, None, None,
                None, None)


class MegaMoeBF16Train(torch.nn.Module):
    """MegaMoe BF16 training wrapper.

    forward returns (y, counts); the fused backward computes
    grad_x / grad_topk_weights / grad_weight1 / grad_weight2 from the
    training_saved_state produced by the same forward call.

    ``save_gmm1_output`` on forward selects bf16 (default), none, auto or
    mxfp8. Auto saves MXFP8 for swiglu_limit=0 and BF16 when clipping is on.
    Explicit mxfp8 with clipping also saves three bit planes from raw gate/up.
    Each autograd context retains its own mode. Backward consumes BF16/MXFP8
    saved gate/up directly; only ``none`` falls back to GMM1 recompute.

    ``retain_dispatch_output=True`` keeps packed tokens and their packed TopK
    weights for the current backward. Setting it to False releases that
    independent Tensor after forward; backward then requires the future
    redispatching implementation and currently raises a clear error.

    The caller owns ``shmem`` and must keep it alive until all pending
    backward calls complete. ``close()`` releases only this operator's resources;
    it does not destroy the shared SHMEM instance.

    Construct after selecting this rank's NPU device. The persistent
    communication context is initialized here, before the first forward.
    """

    def __init__(self, shmem, local_mem_size: int, external_ptr: int | None = None):
        super().__init__()
        if shmem is None:
            raise ValueError("shmem must not be None")
        if local_mem_size <= 0:
            raise ValueError("local_mem_size must be positive")
        self._shmem = shmem
        if external_ptr:
            # Symmetric address already published by another heap init.
            # The operator must not aclshmem_malloc or aclshmem_free it.
            self._op = MegaMoeBF16(int(local_mem_size), int(external_ptr))
        else:
            self._op = MegaMoeBF16(local_mem_size)
        context_like = torch.empty((), device="npu", dtype=torch.bfloat16)
        self._op.prepare_context(shmem, context_like)
        self._closed = False

    def compute_train(self, *args, **kwargs):
        """Raw fused forward. The MoonEP hybrid reduces weight grads itself."""
        if self._closed:
            raise RuntimeError("MegaMoeBF16Train has been closed")
        return self._op.compute_train(*args, **kwargs)

    def backward(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("MegaMoeBF16Train has been closed")
        return self._op.backward(*args, **kwargs)

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        w1,
        w2,
        max_recv_tokens: int,
        *,
        swiglu_limit: float = 0.0,
        save_gmm1_output: str = "bf16",
        retain_dispatch_output: bool = True,
        diagnostic_holder=None,
    ):
        if self._closed:
            raise RuntimeError("MegaMoeBF16Train has been closed")
        return _MegaMoeBF16TrainFunction.apply(
            self._op,
            self._shmem,
            x,
            topk_ids,
            topk_weights,
            w1,
            w2,
            max_recv_tokens,
            float(swiglu_limit),
            save_gmm1_output,
            bool(retain_dispatch_output),
            diagnostic_holder,
        )

    def local_mem_size(self) -> int:
        return self._op.local_mem_size()

    def close(self) -> None:
        if not self._closed:
            self._op.destroy()
            self._closed = True

    def destroy(self) -> None:
        """Compatibility alias for ``close()``."""
        self.close()


# Planner policy is optional; the same executor accepts validated external plans.
from .execution_plan import ExecutionPlanSpec, PreparedExecutionPlan, prepare_execution_plan, import_device_plan
from .execution_plan_device import DevicePlanValidator
from .megaflux_planners import NativeOnlinePlanner, ReplayPlanner
__all__ += ["ExecutionPlanSpec", "PreparedExecutionPlan", "prepare_execution_plan",
            "import_device_plan", "DevicePlanValidator", "NativeOnlinePlanner", "ReplayPlanner"]

from .megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain
__all__ += ["MegaFluxBF16WorkerPipelineTrain"]

from .moonep_plan import make_moonep_plan
from .moonep_planner import MoonEPPlanner
__all__ += ["make_moonep_plan", "MoonEPPlanner"]
