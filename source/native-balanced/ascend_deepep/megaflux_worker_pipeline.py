"""Opt-in BF16 worker executor; unchanged plans, owner identities and SUM order."""
import ctypes,os,pathlib
import torch
from .megaflux_early import MegaFluxBF16EarlyTrain


class MegaFluxBF16WorkerPipelineTrain(MegaFluxBF16EarlyTrain):
    def __init__(self,*args,**kwargs):
        shmem=kwargs.get('shmem',args[0] if args else None)
        home=kwargs.get('home_experts',args[1] if len(args)>1 else None)
        slots=kwargs.get('slots',args[2] if len(args)>2 else None)
        from .runtime_limits import validate_layout,require_gm_binary,gm_planner_requested
        if shmem is not None and slots is not None and home is not None:
            ep=shmem.world_size()
            validate_layout(ep,home,slots);require_gm_binary(ep)
        hidden=kwargs.get('hidden',args[3] if len(args)>3 else None)
        ffn=kwargs.get('ffn',args[4] if len(args)>4 else None)
        if hidden is not None and ffn is not None and (int(hidden)%256 or int(ffn)%256):
            raise ValueError('worker stripes require hidden and ffn multiples of 256')
        root=pathlib.Path(os.environ['WORKER_NATIVE'])
        lib=ctypes.CDLL(str(root/'libascend_deepep_megaflux_early_bf16.so'))
        if lib.megaflux_worker_pipeline_version()!=3:raise RuntimeError('candidate native ABI mismatch')
        if shmem is not None and gm_planner_requested(shmem.world_size()):
            if not hasattr(lib,'megaflux_max_ep') or not hasattr(lib,'megaflux_max_copy_slots') or \
                    lib.megaflux_max_ep()<shmem.world_size() or lib.megaflux_max_copy_slots()<shmem.world_size()*int(slots):
                raise RuntimeError('large EP requires the matching rebuilt S9 worker/copy tables')
        self._worker_native=lib
        self.dispatch_ready_largest=int(lib.megaflux_dispatch_ready_largest()) if hasattr(lib,'megaflux_dispatch_ready_largest') else 0
        if self.dispatch_ready_largest not in (0,1):
            raise RuntimeError('unknown dispatch-ready selector ABI')
        if self.dispatch_ready_largest and (shmem is None or shmem.world_size() not in (2,4,8)):
            raise ValueError('experimental largest-expert row readiness is gated to EP2/4/8')
        self.worker_flags=int(lib.megaflux_worker_pipeline_flags())
        self.gradient_priority=int(lib.megaflux_gradient_priority()) if hasattr(lib,'megaflux_gradient_priority') else 0
        self.worker_shared_prefix=int(lib.megaflux_worker_shared_prefix()) if hasattr(lib,'megaflux_worker_shared_prefix') else 0
        self.worker_external_refresh=int(lib.megaflux_worker_external_refresh()) if hasattr(lib,'megaflux_worker_external_refresh') else 0
        self.worker_owner_rows=int(lib.megaflux_worker_owner_rows()) if hasattr(lib,'megaflux_worker_owner_rows') else 16
        self.worker_owner_compact=int(lib.megaflux_worker_owner_compact()) if hasattr(lib,'megaflux_worker_owner_compact') else 0
        self.backward_restore_mode=int(lib.megaflux_backward_restore_mode()) if hasattr(lib,'megaflux_backward_restore_mode') else 0
        self.restore_operand_ready=int(lib.megaflux_restore_operand_ready()) if hasattr(lib,'megaflux_restore_operand_ready') else 0
        self.restore_delay_cycles=int(lib.megaflux_restore_delay_cycles()) if hasattr(lib,'megaflux_restore_delay_cycles') else 0
        super().__init__(*args,**kwargs)
        h,f=self.w1.shape[2],self.w1.shape[1]//2
        self.worker_queue=torch.empty(32+max(1,self.slots*(h//256+f//256))*8,device=self.w1.device,dtype=torch.int32)
        self.worker_sidecar=None
        self._worker_trace_ptr=None

    def _descriptor(self,plan):
        descriptor=super()._descriptor(plan)
        trace=self.pipeline_trace.data_ptr() if self.pipeline_trace is not None else 0
        if self.worker_sidecar is None or self._worker_trace_ptr!=trace:
            # Queue contents/epoch are initialized in the kernel. The immutable
            # pointer envelope can live across calls and through backward.
            if self.worker_sidecar is None:
                self.worker_sidecar=torch.tensor([3,self.worker_queue.data_ptr(),self.worker_queue.numel(),
                    self.stripe_ready.data_ptr(),trace,self.worker_flags,0,0],device=self.w1.device,dtype=torch.int64)
            else:
                # Preserve the old in-flight envelope; patch the changed trace
                # pointer on-device, without a fresh CPU tensor/upload fence.
                self.worker_sidecar=self.worker_sidecar.detach().clone()
                self.worker_sidecar[4].fill_(trace)
            self._worker_trace_ptr=trace
        # Only the explicit candidate SO interprets this opaque envelope word.
        descriptor[20]=self.worker_sidecar.data_ptr()
        return descriptor

    def collect_worker_status(self):
        if self._pending or self._active:raise RuntimeError('finish before reading diagnostics')
        return self.worker_queue[:8].cpu().tolist() if self.worker_flags&1 else None
