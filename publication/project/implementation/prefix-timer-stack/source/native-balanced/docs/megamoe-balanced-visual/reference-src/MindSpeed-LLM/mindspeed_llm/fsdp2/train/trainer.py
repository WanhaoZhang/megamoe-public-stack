import os
import gc
import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
import contextlib
from typing import Optional, Dict, Any

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.optim.clip_grad_norm import clip_grad_norm
from mindspeed_llm.fsdp2.data.processor.processor_utils import IGNORE_INDEX
from mindspeed_llm.fsdp2.features.chunkloss import chunk_loss, calculate_lm_loss
from mindspeed_llm.fsdp2.checkpoint.utils import empty_cache, cleanup_old_checkpoints
from mindspeed_llm.fsdp2.utils.dist_op import all_reduce
from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.utils.train_monitor import TrainMonitor
from mindspeed_llm.fsdp2.utils.profiler import ProfilerConfig, ProfilerManager
from mindspeed_llm.tools.model_io_trace import ModelIOTraceManager
from mindspeed_llm.tools.msprobe import MsProbeManager
from mindspeed_llm.fsdp2.models.common.mtp import roll_tensor
from mindspeed_llm.fsdp2.models.common.indexer_loss import IndexerLossAutoScaler
from dataclasses import dataclass


logger = get_logger(__name__)


@dataclass
class LossOutput:
    loss: torch.Tensor
    lm_loss_output: torch.Tensor
    mtp_loss_output: torch.Tensor
    aux_loss_output: torch.Tensor
    outputs: dict | None = None


class Trainer:
    """
    Orchestrates the training loop, coordinating Model, Optimizer, Scheduler, Data, and IO.
    Strictly follows the gradient accumulation and loop logic found in HuggingFace Transformers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataloader,
        args,  # TrainingArguments
        parallel_args,
        optimization_args,
        data_args,
        ckpt_manager,
        monitor: TrainMonitor,
        tokenizer=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.args = args
        self.parallel_args = parallel_args
        self.optimization_args = optimization_args
        self.data_args = data_args
        self.ckpt_manager = ckpt_manager
        self.train_monitor = monitor
        self.tokenizer = tokenizer

        # Training state
        self.epoch = 0
        self.global_step = 0
        self._last_logged_step = 0
        self._total_loss_scalar = 0.0
        self._total_lm_loss_scalar = 0.0
        self._total_aux_loss_scalar = 0.0
        self._total_mtp_loss_scalar = 0.0
        self._logging_loss_scalar = 0.0
        self._global_step_last_logged = 0
        self._last_logged_loss_scalar = 0.0
        self._last_logged_lm_loss_scalar = 0.0
        self._last_logged_mtp_loss_scalar = 0.0
        self._last_logged_aux_loss_scalar = 0.0
        self.batch_seqlens = []

        # Timing state
        self._step_start_time = None
        self._log_fwd_bwd_time = os.environ.get("UAM_LOG_FWD_BWD_TIME", "1") == "1"
        self._fwd_bwd_time_accum_ms = 0.0

        # Profiling support
        current_rank = dist.get_rank() if dist.is_initialized() else 0
        prof_config = ProfilerConfig(
            enabled=args.profile,
            profile_step_start=args.profile_step_start,
            profile_step_end=args.profile_step_end,
            profile_ranks=args.profile_ranks,
            profile_level=args.profile_level,
            profile_export_type=args.profile_export_type,
            profile_data_simplification=args.profile_data_simplification,
            profile_with_cpu=args.profile_with_cpu,
            profile_with_stack=args.profile_with_stack,
            profile_with_memory=args.profile_with_memory,
            profile_record_shapes=args.profile_record_shapes,
            profile_save_path=args.profile_save_path,
            current_rank=current_rank,
        )
        self.profiler_manager = ProfilerManager(prof_config)
        self.msprobe_manager = MsProbeManager(
            enabled=args.msprobe,
            config_path=args.msprobe_config_path,
        )
        self.model_io_trace_manager = ModelIOTraceManager(
            enabled=args.model_io_trace,
            config_path=args.model_io_trace_config_path,
            output_path=args.model_io_trace_output_path,
        )

    @staticmethod
    def _build_chunk_loss(labels, ignore_index=-100, chunk_size=1024):
        # For supervised finetuning stages (SFT), labels must be shifted by one position, for pretraining, labels already include shift.

        shift_labels = labels

        # Create a mask to identify valid tokens
        loss_mask = shift_labels != ignore_index

        # Default: normalize loss by total number of valid tokens in the batch.
        alpha = loss_mask.sum()  # scalar
        reduction = "sum"

        # Split shifted labels into chunks along the sequence dimension for memory-efficient processing.
        chunk_labels = torch.split(shift_labels, chunk_size, dim=1)

        # Prepare keyword arguments for each chunk to be passed to the chunked loss function.
        loss_ctx_kwargs = [
            {
                "shift_labels": chunk_labels[i],
                "ignore_index": ignore_index,
                "reduction": reduction,
                "alpha": alpha,
            }
            for i in range(len(chunk_labels))
        ]

        # Return a closure that computes the chunked language modeling loss using the prepared config.
        def loss_ctx(hidden_states, head_weight, head_bias):
            return chunk_loss(
                hidden_states,
                head_weight,
                head_bias,
                loss_forward=calculate_lm_loss,
                loss_kwargs_chunks=loss_ctx_kwargs,
                chunk_size=chunk_size,
            )

        return loss_ctx, loss_mask

    def _maybe_enable_mem_snapshot(self):
        """[MemSnapshot] NPU caching-allocator history, gated by env (rank0 only).

        MS_MEM_SNAPSHOT_STEP: dump a pickle snapshot after this global step.
        MS_MEM_SNAPSHOT_PATH:  output path (default mem_snapshot.pickle).
        Both unset -> no-op, zero overhead (same pattern as MS_LAYER_MEM_LOG).
        """
        if not os.environ.get("MS_MEM_SNAPSHOT_STEP"):
            return
        if torch.distributed.get_rank() != 0:
            return
        try:
            mem = torch.npu.memory if torch.npu.is_available() else torch.cuda.memory
            try:
                mem._record_memory_history(enabled="all", context="all", stacks="all", max_entries=1000000)
            except TypeError:
                # torch.cuda signature fallback
                mem._record_memory_history(True, trace_alloc_max_entries=1000000, trace_alloc_record_context=True)
            logger.info_rank0(f"[MemSnapshot] history recording enabled (step {os.environ['MS_MEM_SNAPSHOT_STEP']})")
        except Exception as e:  # pragma: no cover
            logger.warning(f"[MemSnapshot] failed to enable history: {e}")

    def _maybe_dump_mem_snapshot(self):
        """[MemSnapshot] Dump NPU memory snapshot pickle right after the forward
        pass of the target global step (rank0, once).

        ``MS_MEM_SNAPSHOT_STEP`` is a 1-based update-step number: the dump fires
        during the forward of that step (peak activation state, before backward).
        ``global_step`` only increments after an update finishes, so while running
        the N-th update it equals ``N - 1`` (holds also after resume, since
        ``global_step`` is restored from the checkpoint).
        """
        step = os.environ.get("MS_MEM_SNAPSHOT_STEP")
        if not step:
            return
        if torch.distributed.get_rank() != 0:
            return
        if self.global_step != int(step) - 1:
            return
        if getattr(self, "_mem_snapshot_peak_dumped", False):
            return
        path = os.environ.get("MS_MEM_SNAPSHOT_PATH", "mem_snapshot.pickle")
        try:
            mem = torch.npu.memory if torch.npu.is_available() else torch.cuda.memory
            mem._dump_snapshot(path)
            self._mem_snapshot_peak_dumped = True
            logger.info_rank0(f"[MemSnapshot] peak dumped to {path} (global_step={self.global_step})")
        except Exception as e:  # pragma: no cover
            logger.warning(f"[MemSnapshot] failed to dump: {e}")

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.
        """
        args = self.args
        train_dataloader = self.train_dataloader
        ps = ParallelState()

        # Determine the reduction group
        if ps.is_fsdp_enable():
            reduce_group = ps.get_fsdp_group()
        else:
            reduce_group = None

        # 1. Calculate total steps and current step in the epoch
        steps_in_epoch = len(train_dataloader)
        # Calculate total updates per epoch considering gradient accumulation
        total_updates_per_epoch = steps_in_epoch // args.gradient_accumulation_steps + int(
            steps_in_epoch % args.gradient_accumulation_steps > 0
        )
        total_steps = args.max_steps if args.max_steps > 0 else (total_updates_per_epoch * args.num_train_epochs)
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        save_checkpoint_path = None
        # Calculate global batch size safely
        global_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * ps.get_group_size('dp_fsdp')
        )

        logger.info_rank0("***** Running training (FSDP2) *****")
        logger.info_rank0(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info_rank0(f"  Num Epochs = {args.num_train_epochs}")
        logger.info_rank0(f"  Total Batch Size = {global_batch_size}")
        logger.info_rank0(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info_rank0(f"  Total optimization steps = {total_steps}")

        # 2. Resume training from checkpoint
        if resume_from_checkpoint:
            # Prepare state dict containing model, optimizer and extra state
            state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}
            # Load checkpoint from the specified path
            self.ckpt_manager.load(resume_from_checkpoint, state)
            extra_state = state.get("extra_state", {})

            if extra_state:
                # Restore training step counters
                self.global_step = state["extra_state"]["global_step"]
                self._global_step_last_logged = state["extra_state"].get("_global_step_last_logged", self.global_step)
                # Use the saved logging cursor; older checkpoints may only have global_step.
                self._last_logged_step = state["extra_state"].get("_last_logged_step", self.global_step)
                self._last_logged_loss_scalar = state["extra_state"].get("_last_logged_loss_scalar", 0.0)
                self._last_logged_lm_loss_scalar = state["extra_state"].get("_last_logged_lm_loss_scalar", 0.0)
                self._last_logged_mtp_loss_scalar = state["extra_state"].get("_last_logged_mtp_loss_scalar", 0.0)
                self._last_logged_aux_loss_scalar = state["extra_state"].get("_last_logged_aux_loss_scalar", 0.0)
                # Keep (total - last_logged) on the same baseline across resume.
                # Prefer restored totals; if missing, seed from last_logged cursors.
                self._total_loss_scalar = state["extra_state"].get("_total_loss_scalar", self._last_logged_loss_scalar)
                self._total_lm_loss_scalar = state["extra_state"].get(
                    "_total_lm_loss_scalar", self._last_logged_lm_loss_scalar
                )
                self._total_mtp_loss_scalar = state["extra_state"].get(
                    "_total_mtp_loss_scalar", self._last_logged_mtp_loss_scalar
                )
                self._total_aux_loss_scalar = state["extra_state"].get(
                    "_total_aux_loss_scalar", self._last_logged_aux_loss_scalar
                )
                if "train_metric" in state["extra_state"]:
                    self.train_monitor.load_state_dict(state["extra_state"]["train_metric"])

                # Calculate completed epochs and steps within current epoch based on global step
                epochs_trained = int(self.global_step // total_updates_per_epoch)
                steps_trained_in_current_epoch = self.global_step % total_updates_per_epoch

                # Restore learning rate scheduler and RNG state for reproducibility
                self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])

                # Synchronize all processes before continuing
                dist.barrier()
                logger.info_rank0(f"Load distributed checkpoint from {resume_from_checkpoint} successfully!")
                logger.info_rank0(f"Resuming from epoch {epochs_trained}, step {steps_trained_in_current_epoch}")
                logger.info_rank0(f"Global step = {self.global_step}")
            else:
                # No extra state found, only model weights were loaded, start from scratch
                logger.info_rank0("Loaded model weights only, starting training from step 0")

        self.msprobe_manager.set_init_step(self.global_step)
        self._maybe_enable_mem_snapshot()
        self.model.train()
        train_start_time = time.time()
        self._step_start_time = time.time()
        from .step_record import StepRecorder
        paired_step_recorder = StepRecorder.from_environment()
        if paired_step_recorder is not None:
            paired_step_recorder.snapshot_parameters(self.model)

        # Start profiler
        if self.profiler_manager.profiler is not None:
            self.profiler_manager.start()

        # --- Python 自动 GC 控制（参考 Megatron-LM training.py manual_gc）---
        # 自动 GC 的触发时机取决于各 rank 的对象分配量/回收代数，会在随机 step 触发
        # stop-the-world 收集，导致 step 时间抖动且各 rank 不同步。训练前关闭自动 GC，
        # 并做一次完整回收清空历史垃圾。
        #   PYTHON_AUTO_GC=1      -> 恢复 Python 默认自动 GC（排查/对照用）
        #   MANUAL_GC_INTERVAL=N  -> 每 N 个 step 手动 gc.collect() 一次（默认 0 = 不手动回收）
        manual_gc_interval = int(os.environ.get("MANUAL_GC_INTERVAL", "0") or 0)
        if os.environ.get("PYTHON_AUTO_GC", "0") == "1":
            gc.enable()
            logger.info_rank0("> Python automatic GC ENABLED (PYTHON_AUTO_GC=1, stock behavior)")
        else:
            gc.disable()
            gc.collect()
            logger.info_rank0(
                "> Python automatic GC disabled (gc.disable + gc.collect before training loop)"
                + (f", manual gc every {manual_gc_interval} steps" if manual_gc_interval else "")
            )

        # --- Epoch Loop ---
        for epoch in range(epochs_trained, int(args.num_train_epochs)):
            self.epoch = epoch

            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = iter(train_dataloader)

            # --- Gradient Accumulation Loop ---
            # Handle the remainder batch at the end of an epoch
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps

            total_updates = total_updates_per_epoch
            # Calculate the starting update_step for this epoch
            # If resuming in the first epoch after checkpoint, start from steps_trained_in_current_epoch
            # Otherwise start from 0
            if epoch == epochs_trained and steps_trained_in_current_epoch > 0:
                # Calculate number of batches to skip (accounting for gradient accumulation)
                steps_to_skip = steps_trained_in_current_epoch * args.gradient_accumulation_steps
                logger.info_rank0(f"  Skipping {steps_to_skip} batches in epoch {epoch}")

                # Skip already trained batches by advancing the iterator
                for _ in range(steps_to_skip):
                    try:
                        next(epoch_iterator)
                    except StopIteration:
                        break

                start_update_step = steps_trained_in_current_epoch
            else:
                start_update_step = 0

            for update_step in range(start_update_step, total_updates):
                # 手动 GC（可选）：固定 step 间隔做一次完整回收，各 rank 同步触发（MANUAL_GC_INTERVAL）
                if manual_gc_interval and (self.global_step + 1) % manual_gc_interval == 0:
                    gc.collect()

                # Determine how many micro-batches are in this update
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder

                # [Helper] Fetch N samples from the iterator and calculate valid token count
                if paired_step_recorder is not None:
                    paired_step_recorder.start()
                batch_samples, batch_seqlens, num_items_in_batch = self.get_batch_samples_func()(
                    ps, epoch_iterator, num_batches
                )
                paired_batch = paired_step_recorder.hold_batch(batch_samples) if paired_step_recorder is not None else None
                self.current_gradient_accumulation_steps = len(batch_samples)
                # Initialize accumulated loss for the current step
                current_step_loss = 0.0
                current_step_lm_loss = 0.0
                current_step_mtp_loss = 0.0
                current_step_aux_loss = 0.0

                self.msprobe_manager.start_step(self.model)
                self.model_io_trace_manager.start_step(self.model, self.global_step)
                self._nan_diag_maybe_setup()

                # --- Micro-Batch Loop ---
                for i, inputs in enumerate(batch_samples):
                    do_sync_step = i == len(batch_samples) - 1

                    # FSDP Communication Optimization
                    # Only synchronize gradients on the last micro-batch
                    fsdp_root = self._get_fsdp_root()

                    sync_context = (
                        fsdp_root.no_sync()
                        if (not do_sync_step and hasattr(fsdp_root, "no_sync"))
                        else contextlib.nullcontext()
                    )

                    with sync_context:
                        # Forward & Backward
                        # Note: training_step already divides loss by accum_steps
                        loss, lm_loss, mtp_loss, aux_loss = self.training_step(inputs, num_items_in_batch)

                    self._nan_diag_maybe_report(i, loss, lm_loss, mtp_loss, aux_loss, inputs=inputs)

                    # Accumulate Loss for logging (restore to original scale for display)
                    # Check for NaN/Inf and finite explosions to avoid polluting metrics:
                    # a single micro-batch with lm ~1e34 (async memory corruption from
                    # the fused MoE kernels) would otherwise poison every later average.
                    for acc_name, t in (
                        ("current_step_loss", loss),
                        ("current_step_lm_loss", lm_loss),
                        ("current_step_mtp_loss", mtp_loss),
                        ("current_step_aux_loss", aux_loss),
                    ):
                        v = t.item()
                        if os.environ.get("UAM_VALIDATE_FINITE", "0") == "1" and (not math.isfinite(v) or abs(v) >= 1e4):
                            raise FloatingPointError(f"Invalid raw {acc_name} at step {self.global_step + 1}: {v}")
                        if v == v and abs(v) != float("inf") and abs(v) < 1e4:
                            if acc_name == "current_step_loss":
                                current_step_loss += v
                            elif acc_name == "current_step_lm_loss":
                                current_step_lm_loss += v
                            elif acc_name == "current_step_mtp_loss":
                                current_step_mtp_loss += v
                            else:
                                current_step_aux_loss += v
                # --- Optimizer Step (Executed only after accumulation) ---
                # At this point, the micro-batch loop is finished, gradients are accumulated

                # 1. Clip Gradients and get Norm
                diagnostic_steps = os.environ.get("UAM_GRAD_DIAG_STEPS", "").split(",")
                os.environ["UAM_GRAD_DIAG_ACTIVE"] = "1" if str(self.global_step + 1) in diagnostic_steps else "0"
                if os.environ.get("UAM_SYNC_BEFORE_CLIP", "0") == "1":
                    torch.accelerator.synchronize()
                # MoonEP's EDP sum is once per optimizer step. Doing it inside
                # each MoE backward re-sums grads when accumulation > 1.
                from fsdp_turbo.distributed.expert_parallel.moonep.runtime import (
                    reduce_moonep_replica_gradients,
                )

                reduce_moonep_replica_gradients(self.model)
                grad_norm = clip_grad_norm(self.model, args.max_grad_norm)
                # Compatibility: Ensure grad_norm is a float
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()

                # 2. Update Parameters
                # Skip the optimizer step when the grad norm is non-finite (NaN/Inf).
                # Applying NaN/Inf grads would corrupt the weights and later trigger
                # AICPU operator failures in the MoE/EP forward; the bad step is
                # discarded by zeroing the gradients instead.
                if math.isfinite(grad_norm):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                elif torch.distributed.get_rank() == 0:
                    logger.warning(
                        f"Skipping optimizer and scheduler update at step {self.global_step + 1}: "
                        f"non-finite gradient norm ({grad_norm})."
                    )
                self.optimizer.zero_grad()
                paired_step_row = None
                if paired_step_recorder is not None:
                    paired_step_row = paired_step_recorder.stop(
                        self.global_step + 1,
                        (current_step_loss, current_step_lm_loss, current_step_mtp_loss, current_step_aux_loss),
                        grad_norm)
                self.msprobe_manager.end_step()
                self.model_io_trace_manager.end_step()

                self.global_step += 1

                # PROFILING HOOK
                if self.profiler_manager.profiler is not None:
                    self.profiler_manager.step()
                # HOOK END

                # 3. Distributed Aggregation of Loss and GradNorm
                # Only perform this when global_step updates.
                # current_step_loss is sum(micro_batches), conceptually it represents the loss of the mini-batch.

                reduced_loss, reduced_lm_loss, reduced_mtp_loss, reduced_aux_loss, reduced_grad_norm = all_reduce(
                    (current_step_loss, current_step_lm_loss, current_step_mtp_loss, current_step_aux_loss, grad_norm),
                    group=reduce_group,
                )

                self._total_loss_scalar += reduced_loss
                if paired_step_recorder is not None:
                    paired_step_row['reduced_losses'] = dict(
                        loss=float(reduced_loss), lm_loss=float(reduced_lm_loss),
                        mtp_loss=float(reduced_mtp_loss), aux_loss=float(reduced_aux_loss),
                        grad_norm=float(reduced_grad_norm))
                    paired_step_row['batch_fingerprints'] = paired_step_recorder.fingerprint_batch(paired_batch)
                    del paired_batch
                    paired_step_recorder.write(paired_step_row)
                self._total_lm_loss_scalar += reduced_lm_loss
                self._total_mtp_loss_scalar += reduced_mtp_loss
                self._total_aux_loss_scalar += reduced_aux_loss
                self.batch_seqlens.extend(batch_seqlens)

                # 4. Logging
                if self.global_step % args.logging_steps == 0:
                    _, record_info = self.train_monitor.step(
                        self.epoch,
                        self.lr_scheduler,
                        global_batch_size,
                        reduced_grad_norm,
                        self.batch_seqlens,
                        self._step_start_time,
                        total_steps,
                        self.global_step,
                        self._last_logged_step,
                        self._total_loss_scalar,
                        self._total_lm_loss_scalar,
                        self._total_mtp_loss_scalar,
                        self._total_aux_loss_scalar,
                        self._last_logged_loss_scalar,
                        self._last_logged_lm_loss_scalar,
                        self._last_logged_mtp_loss_scalar,
                        self._last_logged_aux_loss_scalar,
                        sparse_attn_reduce_group=reduce_group,
                        fwd_bwd_time_ms=(
                            self._fwd_bwd_time_accum_ms
                            / max(self.global_step - self._last_logged_step, 1)
                            if self._log_fwd_bwd_time else None
                        ),
                    )
                    # update record
                    self._fwd_bwd_time_accum_ms = 0.0
                    self._step_start_time = record_info['time']
                    self._last_logged_loss_scalar = record_info['logged_loss']
                    self._last_logged_lm_loss_scalar = record_info['logged_lm_loss']
                    self._last_logged_mtp_loss_scalar = record_info['logged_mtp_loss']
                    self._last_logged_step = record_info['logged_step']
                    self._last_logged_aux_loss_scalar = record_info['logged_aux_loss']
                    self.batch_seqlens.clear()

                # 5. Saving
                if args.save_steps > 0 and self.global_step % args.save_steps == 0:
                    if not args.output_dir:
                        logger.info_rank0("output_dir is not set, skipping checkpoint saving")
                    else:
                        empty_cache()
                        save_checkpoint_path = os.path.join(
                            args.output_dir, "checkpoints", f"global_step_{self.global_step}"
                        )
                        state = {
                            "model": self.model,
                            "optimizer": self.optimizer,
                            "extra_state": {
                                "global_step": self.global_step,
                                "_global_step_last_logged": self._global_step_last_logged,
                                "_last_logged_step": self._last_logged_step,
                                "_last_logged_loss_scalar": self._last_logged_loss_scalar,
                                "_last_logged_lm_loss_scalar": self._last_logged_lm_loss_scalar,
                                "_last_logged_mtp_loss_scalar": self._last_logged_mtp_loss_scalar,
                                "_last_logged_aux_loss_scalar": self._last_logged_aux_loss_scalar,
                                "_total_loss_scalar": self._total_loss_scalar,
                                "_total_lm_loss_scalar": self._total_lm_loss_scalar,
                                "_total_mtp_loss_scalar": self._total_mtp_loss_scalar,
                                "_total_aux_loss_scalar": self._total_aux_loss_scalar,
                                "lr_scheduler": self.lr_scheduler.state_dict(),
                                "train_metric": self.train_monitor.state_dict(),
                                "torch_rng_state": torch.get_rng_state(),
                            },
                        }
                        self.ckpt_manager.save(
                            path=save_checkpoint_path,
                            state=state,
                            save_only_model=args.save_only_model,
                            global_steps=self.global_step,
                        )
                        dist.barrier()
                        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                        cleanup_old_checkpoints(args, self.data_args.data_shared_file_system)

                if self.global_step >= total_steps:
                    break
            # Reset counter after completing an epoch
            steps_trained_in_current_epoch = 0
            # Save checkpoint at specified epoch intervals
            already_saved = args.save_steps > 0 and self.global_step % args.save_steps == 0
            if args.save_epochs and (epoch + 1) % args.save_epochs == 0 and not already_saved:
                if not args.output_dir:
                    logger.info_rank0("output_dir is not set, skipping checkpoint saving")
                else:
                    empty_cache()
                    save_checkpoint_path = os.path.join(
                        args.output_dir, "checkpoints", f"global_step_{self.global_step}"
                    )
                    state = {
                        "model": self.model,
                        "optimizer": self.optimizer,
                        "extra_state": {
                            "global_step": self.global_step,
                            "_global_step_last_logged": self._global_step_last_logged,
                            "_last_logged_step": self._last_logged_step,
                            "_last_logged_loss_scalar": self._last_logged_loss_scalar,
                            "_last_logged_lm_loss_scalar": self._last_logged_lm_loss_scalar,
                            "_last_logged_mtp_loss_scalar": self._last_logged_mtp_loss_scalar,
                            "_last_logged_aux_loss_scalar": self._last_logged_aux_loss_scalar,
                            "_total_loss_scalar": self._total_loss_scalar,
                            "_total_lm_loss_scalar": self._total_lm_loss_scalar,
                            "_total_mtp_loss_scalar": self._total_mtp_loss_scalar,
                            "_total_aux_loss_scalar": self._total_aux_loss_scalar,
                            "lr_scheduler": self.lr_scheduler.state_dict(),
                            "train_metric": self.train_monitor.state_dict(),
                            "torch_rng_state": torch.get_rng_state(),
                        },
                    }
                    self.ckpt_manager.save(
                        path=save_checkpoint_path,
                        state=state,
                        save_only_model=args.save_only_model,
                        global_steps=self.global_step,
                    )
                    dist.barrier()
                    logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                    cleanup_old_checkpoints(args, self.data_args.data_shared_file_system)

            if self.global_step >= total_steps:
                break
        if paired_step_recorder is not None:
            paired_step_recorder.close()
        # Stop profiler
        if self.profiler_manager.profiler is not None:
            self.profiler_manager.stop()

        # Save model in HF format — all ranks participate in gather, rank 0 writes
        if args.save_hf_weights and save_checkpoint_path is not None:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state_dict = get_model_state_dict(self.model.model, options=options)

            if dist.get_rank() == 0:
                model_configs = [self.model.model.config, self.tokenizer]
                self.ckpt_manager.save_model_weights(args.output_dir, model_state_dict, model_configs=model_configs)
                logger.info_rank0(f"Huggingface checkpoint saved at {args.output_dir} successfully!")

            del model_state_dict
            gc.collect()
            empty_cache()
            dist.barrier()

        # Save training args — rank 0 only
        if dist.get_rank() == 0 and args.output_dir:
            self.ckpt_manager.save_args(args, args.output_dir)
            logger.info_rank0(f"Training arguments saved at {args.output_dir} successfully!")
        logger.info_rank0(f"Training completed in {time.time() - train_start_time:.2f}s")

    # ---- MegaMOE NaN diagnosis (temporary, env-gated by MEGAMOE_NAN_DIAG_STEP) ----
    def _nan_diag_maybe_setup(self):
        import os
        step = os.environ.get("MEGAMOE_NAN_DIAG_STEP")
        if not step or self.global_step < int(step) or getattr(self, "_nan_diag_handles", None):
            return
        self._nan_diag_records = []

        def _tensors_of(obj, depth=0):
            if depth > 3:
                return []
            if isinstance(obj, torch.Tensor):
                return [obj]
            if isinstance(obj, (tuple, list)):
                out = []
                for v in obj:
                    out.extend(_tensors_of(v, depth + 1))
                return out
            if isinstance(obj, dict):
                out = []
                for v in obj.values():
                    out.extend(_tensors_of(v, depth + 1))
                return out
            return []

        def _finite_all(ts):
            for t in ts:
                if not bool(torch.isfinite(t.detach()).all().item()):
                    return False
            return True

        def _max_abs(ts):
            m = 0.0
            for t in ts:
                v = t.detach().abs().max().item()
                if v > m:
                    m = v
            return m

        def make_hook(name, cls_name):
            def hook(module, inputs, output):
                ins = _tensors_of(inputs)[:8]
                outs = _tensors_of(output)[:8]
                rec = {
                    "name": name,
                    "cls": cls_name,
                    "in_finite": _finite_all(ins),
                    "in_max": _max_abs(ins) if ins else None,
                    "out_finite": _finite_all(outs),
                    "out_max": _max_abs(outs) if outs else None,
                }
                if not rec["out_finite"] and ins:
                    rec["out_nan"] = int(torch.isnan(outs[0].detach()).sum().item()) if outs else 0
                    rec["out_inf"] = int(torch.isinf(outs[0].detach()).sum().item()) if outs else 0
                    rec["out_shape"] = tuple(outs[0].shape) if outs else None
                self._nan_diag_records.append(rec)
            return hook

        handles = []
        for name, module in self.model.named_modules():
            if name == "":
                # root: pre-hook clears records so each forward starts fresh
                handles.append(module.register_forward_pre_hook(
                    lambda mod, args: self._nan_diag_records.clear()))
            else:
                handles.append(module.register_forward_hook(make_hook(name, type(module).__name__)))
        self._nan_diag_handles = handles
        print(f"[NanDiag] hooks registered at step={self.global_step} "
              f"({len(handles)} modules)", flush=True)

    def _nan_diag_maybe_report(self, micro_idx, loss, lm_loss, mtp_loss, aux_loss, inputs=None):
        if not getattr(self, "_nan_diag_handles", None):
            return

        def val(t):
            try:
                return float(t.detach().float().item())
            except Exception:
                return None

        def fin(t):
            v = val(t)
            return v is not None and v == v and abs(v) != float("inf")

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        recs = getattr(self, "_nan_diag_records", [])
        nonfinite = [r for r in recs if not r["out_finite"]]

        lm_v = val(lm_loss)
        loss_v = val(loss)
        print(f"[NanDiag][rank{rank}] step={self.global_step} micro={micro_idx} "
              f"loss={loss_v:.4e} lm={lm_v:.4e} mtp={val(mtp_loss):.4e} "
              f"aux={val(aux_loss):.4e} mods={len(recs)} nonfinite_mods={len(nonfinite)}",
              flush=True)

        # Finite explosion: all module outputs stay finite but the loss jumps to
        # astronomical values (observed 4.15e34 at step 951). Catch it before the
        # loss pollutes the running averages / backward pass.
        exploded = (lm_v is not None and lm_v > 1e4) or (loss_v is not None and loss_v > 1e4)
        if not (exploded
                or not fin(loss) or not fin(lm_loss) or not fin(mtp_loss) or not fin(aux_loss)
                or lm_v == 0.0):
            return

        reason = ("lm_exploded" if exploded else
                  "lm_zero" if lm_v == 0.0 else "non_finite_loss")
        print(f"[NanDiag][rank{rank}] TRIGGER reason={reason}", flush=True)

        # Top-20 modules by output magnitude: locates the explosion source.
        by_mag = sorted(recs, key=lambda r: -(r.get("out_max") or 0.0))[:20]
        for r in by_mag:
            print(f"[NanDiag][rank{rank}] TOP-MAG {r['name']} ({r['cls']}) "
                  f"in_finite={r['in_finite']} in_max={r['in_max']} "
                  f"out_finite={r['out_finite']} out_max={r['out_max']}", flush=True)
        for r in nonfinite[:3]:
            print(f"[NanDiag][rank{rank}] FIRST-NONFINITE module={r['name']} "
                  f"cls={r['cls']} in_finite={r['in_finite']} "
                  f"in_max={r['in_max']} out_nan={r.get('out_nan')} "
                  f"out_inf={r.get('out_inf')} out_shape={r.get('out_shape')}",
                  flush=True)
        for r in recs[-10:]:
            print(f"[NanDiag][rank{rank}] ctx {r['name']} ({r['cls']}) "
                  f"in_finite={r['in_finite']} in_max={r['in_max']} "
                  f"out_finite={r['out_finite']} out_max={r['out_max']}", flush=True)

        out_dir = "/mnt/share/z00583051/codes/10t_1k/output/nan_diag"
        os.makedirs(out_dir, exist_ok=True)
        tag = f"r{rank}_s{self.global_step}_m{micro_idx}"
        # Full per-module magnitude trace for offline analysis.
        try:
            import json
            with open(f"{out_dir}/records_{tag}.json", "w") as f:
                json.dump(recs, f, indent=1)
            print(f"[NanDiag][rank{rank}] records saved", flush=True)
        except Exception as exc:
            print(f"[NanDiag][rank{rank}] records save failed: {exc}", flush=True)
        # Save the offending batch for offline reproduction.
        try:
            dump = {k: v.detach().cpu() for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)} if inputs else {}
            dump["_meta"] = torch.tensor([self.global_step, micro_idx, rank])
            torch.save(dump, f"{out_dir}/batch_{tag}.pt")
            print(f"[NanDiag][rank{rank}] batch saved to {out_dir}/batch_{tag}.pt", flush=True)
        except Exception as exc:
            print(f"[NanDiag][rank{rank}] batch save failed: {exc}", flush=True)
        print(f"[NanDiag][rank{rank}] captured reason={reason}; removing diag hooks, training continues", flush=True)
        # Keep the 2000-step validation running: dump once, then detach the
        # per-module hooks so later steps pay no sync overhead.
        for h in getattr(self, "_nan_diag_handles", []) or []:
            try:
                h.remove()
            except Exception:
                pass
        self._nan_diag_handles = None

    def training_step(self, inputs: Dict[str, Any], num_items_in_batch: Optional[int]) -> torch.Tensor:
        """
        Performs a single forward and backward pass.
        """
        # 1. Set model to train mode
        self.model.train()
        # Some custom optimizers require explicit train() calls
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # Indexer KL loss backward scale: only the batch statistics are passed in here;
        # the mcore-parity scale derivation lives in indexer_loss
        # (IndexerLossAutoScaler.set_loss_scale_from_batch_stats).
        IndexerLossAutoScaler.set_loss_scale_from_batch_stats(
            num_items_in_batch, self.current_gradient_accumulation_steps
        )

        # 2. Forward pass
        if self._log_fwd_bwd_time:
            # Drain prior optimizer/data work before the host clock starts.
            # Device-wide synchronization also covers FSDP/EP/UAM streams.
            torch.accelerator.synchronize()
            fwd_bwd_start = time.time()
        loss_all = self._compute_loss(inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        # [MemSnapshot] peak-state dump right after forward (activations still alive)
        self._maybe_dump_mem_snapshot()
        if self.args.stage == 'pt':
            loss_all.loss = loss_all.loss / self.current_gradient_accumulation_steps
            loss_all.lm_loss_output = loss_all.lm_loss_output / self.current_gradient_accumulation_steps
            loss_all.mtp_loss_output = loss_all.mtp_loss_output / self.current_gradient_accumulation_steps
            loss_all.aux_loss_output = loss_all.aux_loss_output / self.current_gradient_accumulation_steps

        loss = loss_all.loss
        # 3. Clean up inputs to save memory
        del inputs

        # 4. Multi-device parallelism: Average loss across devices (if not using FSDP internal handling)
        if torch.cuda.device_count() > 1:
            # Note: Standard FSDP usually handles loss averaging via the reduction of gradients or explicit loss reduction.
            # If using DDP logic manually without DDP wrapper, this might be needed.
            # For FSDP2, loss is usually local until aggregation.
            loss = loss.mean()

        loss_for_log = loss.detach()
        # CP backward loss scaling: scale the backward loss by cp_size (all cp types)
        # to compensate the differentiable all-gather's backward reduction. The cp grad
        # grad-norm scaling in clip_grad_norm is removed, so this is the single CP
        # gradient scaling point. Skip the mul when cp_size == 1 to keep the cp-off
        # backward graph identical to non-CP.
        if self.parallel_args.cp_size > 1:
            loss = loss * self.parallel_args.cp_size

        # 5. Backward pass
        # NaN/Inf guard: backward through a non-finite loss runs the AICPU/HCCL
        # backward ops (e.g. alltoallv) on NaN gradients, which hang for ~120 s
        # and then abort the whole run with CANN error 507018 (AICPU exception).
        # The forward path is finite here (the NaN appears only in the final loss
        # combination), so backing through nan_to_num(loss) yields zero gradients
        # for this micro-batch: the FSDP/EP gradient sync still runs (accumulated
        # grads from other micro-batches are untouched) and the bad sample simply
        # contributes nothing.
        # Finite-explosion guard: the fused MoE kernels can asynchronously corrupt
        # the loss-related device memory AFTER the forward hooks already observed
        # normal logits (observed lm=4.15e34 at step 951 with all module outputs
        # finite). Backward through such a loss scales every gradient by ~1e34;
        # zero it instead so the micro-batch contributes nothing, and re-read the
        # values after a full device sync to distinguish a persistent wild value
        # from an async-race transient read.
        loss_finite = bool(torch.isfinite(loss).item())
        loss_mag = abs(float(loss_for_log.item())) if loss_finite else float("nan")
        lm_mag = abs(float(loss_all.lm_loss_output.detach().item())) if loss_finite else float("nan")
        exploded = loss_finite and (loss_mag > 1e4 or lm_mag > 1e4)
        if (not loss_finite) or exploded:
            rank = dist.get_rank() if dist.is_initialized() else -1
            print(
                f"[Loss-Guard][rank={rank}] step={self.global_step} "
                f"loss={loss_mag:.3e} lm={lm_mag:.3e} "
                f"mtp={float(loss_all.mtp_loss_output.detach().item()):.3e} "
                f"aux={float(loss_all.aux_loss_output.detach().item()):.3e} "
                f"finite={loss_finite} loss_ptr={loss.data_ptr()} "
                f"lm_ptr={loss_all.lm_loss_output.data_ptr()} -> zero-grad backward",
                flush=True,
            )
            if exploded:
                torch.npu.synchronize()
                print(
                    f"[Loss-Guard][rank={rank}] post-sync re-read "
                    f"loss={float(loss_for_log.item()):.3e} "
                    f"lm={float(loss_all.lm_loss_output.detach().item()):.3e}",
                    flush=True,
                )
                loss = loss * 0.0
            else:
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        loss.backward()
        if self._log_fwd_bwd_time:
            # Autograd returns after submission; wait for device completion.
            # Clip, optimizer, scheduler and zero_grad are outside this interval.
            torch.accelerator.synchronize()
            self._fwd_bwd_time_accum_ms += (time.time() - fwd_bwd_start) * 1000.0

        # 6. Return detached loss (pre-cp-scale for logging)
        return (
            loss_for_log,
            loss_all.lm_loss_output.detach(),
            loss_all.mtp_loss_output.detach(),
            loss_all.aux_loss_output.detach(),
        )

    def get_batch_samples_func(self):
        if self.parallel_args.cp_size > 1:
            if self.parallel_args.cp_type == "ulysses":
                return self._get_batch_samples_ulysses
            elif self.parallel_args.cp_type == "ring":
                return self._get_batch_samples_megatron
            elif self.parallel_args.cp_type == "kvallgather":
                return self._get_batch_samples_ulysses
            else:
                raise ValueError(f"Unsupported cp_type: '{self.parallel_args.cp_type}' when cp_size > 1.")
        else:
            return self._get_batch_samples

    def _get_batch_samples_ulysses(self, parallel_state, epoch_iterator, num_batches):
        cp_size = parallel_state.get_group_size("cp")
        cp_rank = parallel_state.get_rank("cp")
        batch, batch_seqlens, num_items_in_batch = self._get_batch_samples(parallel_state, epoch_iterator, num_batches)

        for sample in batch:
            labels = torch.nn.functional.pad(sample['labels'], (0, 1), value=IGNORE_INDEX)
            shift_labels = labels[..., 1:]
            sample['shift_labels'] = shift_labels
            if "position_ids" not in sample:
                position_ids = torch.arange(0, shift_labels.shape[1], device=shift_labels.device).unsqueeze(0)
                sample['position_ids'] = position_ids

            for key, val in sample.items():
                if key in ('attention_mask', 'actual_seq_len', 'dataset_id'):
                    continue
                if val is not None:
                    seq_dim = 1
                    # 2. Calculate total sequence length
                    seq_total_len = val.size(seq_dim)

                    # 3. Calculate the sequence length each rank is responsible for (handle indivisible cases)
                    chunk_size = seq_total_len // cp_size
                    remainder = seq_total_len % cp_size  # Remainder, the first 'remainder' ranks take 1 more token each

                    # 4. Calculate the start and end indices of the slice for current rank (core logic of Ring slicing)
                    if cp_rank < remainder:
                        # For the first 'remainder' ranks, each rank is responsible for (chunk_size + 1) tokens
                        start_idx = cp_rank * (chunk_size + 1)
                        end_idx = start_idx + (chunk_size + 1)
                    else:
                        # For the remaining ranks, each rank is responsible for 'chunk_size' tokens
                        start_idx = remainder * (chunk_size + 1) + (cp_rank - remainder) * chunk_size
                        end_idx = start_idx + chunk_size
                    # 5. Perform slicing: retain only the sequence part responsible for current rank
                    val_sliced = val.narrow(seq_dim, start_idx, end_idx - start_idx)
                    # 6. Update the value in sample with the sliced tensor
                    if key == 'shift_labels':
                        val_sliced = val_sliced.contiguous()
                    device = torch.accelerator.current_device()
                    sample[key] = val_sliced.to(device, non_blocking=True)

        return batch, batch_seqlens, num_items_in_batch

    def _get_batch_samples_megatron(self, parallel_state, epoch_iterator, num_batches):
        cp_size = parallel_state.get_group_size("cp")
        cp_rank = parallel_state.get_rank("cp")
        batch, batch_seqlens, num_items_in_batch = self._get_batch_samples(parallel_state, epoch_iterator, num_batches)

        for sample in batch:
            # ===================== Original logic: generate shift_labels =====================
            labels = torch.nn.functional.pad(sample['labels'], (0, 1), value=IGNORE_INDEX)
            shift_labels = labels[..., 1:]
            sample['shift_labels'] = shift_labels

            # Original logic: generate position_ids
            if "position_ids" not in sample:
                position_ids = torch.arange(0, shift_labels.shape[1], device=shift_labels.device).unsqueeze(0)
                sample['position_ids'] = position_ids

            # ===================== Core modification: Replace with Ring CP load-balanced splitting =====================
            for key, val in sample.items():
                # Skip attention_mask
                if key == 'attention_mask':
                    continue
                if val is not None:
                    seq_dim = 1  # Fixed sequence dimension, consistent with reference code

                    # ========== Core logic of Ring CP load-balanced splitting (fully aligned with reference code) ==========
                    # 1. Reshape: [bs, seq_len, ...] -> [bs, 2*cp_size, seq_len/(2*cp_size), ...]
                    val = val.view(
                        *val.shape[0:seq_dim],
                        2 * cp_size,
                        val.shape[seq_dim] // (2 * cp_size),
                        *val.shape[(seq_dim + 1) :],
                    )
                    # 2. Generate ring symmetric indices (core of load balancing)
                    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
                    # 3. Select tensor by indices
                    val = val.index_select(seq_dim, index)
                    # 4. Merge dimensions and restore shape
                    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                    # ====================================================================================================

                    # ===================== Original logic: Contiguous + device migration =====================
                    if key == 'shift_labels':
                        val = val.contiguous()
                    device = torch.accelerator.current_device()
                    sample[key] = val.to(device, non_blocking=True)

        return batch, batch_seqlens, num_items_in_batch

    def _get_batch_samples(self, parallel_state, epoch_iterator, num_batches):
        """
        Fetch num_batches samples from the iterator at once.

        Args:
            parallel_state: Parallel state object (unused, kept for interface consistency)
            epoch_iterator: Data iterator for the current epoch
            num_batches: Number of batches to fetch

        Returns:
            tuple: Three values with distinct purposes:

            batch_samples (List[Dict[str, torch.Tensor]]):
                Raw input data for model forward pass. Example: [{"input_ids": tensor1, "labels": tensor1_labels, ...}, ...]

            batch_seqlens (List[int]):
                Sequence length of each sample across all fetched batches. Example: [128, 256, 64, 192]

            num_items_in_batch (int):
                Total valid tokens across all fetched batches (distributed aggregation). Example: 640
        """
        batch_samples = []
        batch_seqlens = []
        for _ in range(num_batches):
            try:
                device = torch.accelerator.current_device()
                data = next(epoch_iterator)
                data["input_ids"] = data["input_ids"].to(device, non_blocking=True)
                data["labels"] = data["labels"].to(device, non_blocking=True)
                if "attention_mask" in data:
                    data["attention_mask"] = data["attention_mask"].to(device, non_blocking=True)
                if "position_ids" in data:
                    data["position_ids"] = data["position_ids"].to(device, non_blocking=True)
                if "actual_seq_len" in data:
                    data["actual_seq_len"] = data["actual_seq_len"].view(-1)
                batch_samples.append(data)

                # Calculate sequence lengths for each sample in the current batch
                sample_seqlens = []
                if "attention_mask" in data:
                    sample_seqlens = data["attention_mask"].sum(-1).tolist()
                elif "labels" in data:
                    sample_seqlens = (data["labels"].ne(-100)).sum(-1).tolist()
                batch_seqlens.extend(sample_seqlens)
            except StopIteration:
                break

        # Calculate total valid tokens across all fetched batches (distributed aggregation)
        num_items_in_batch = self._get_num_items_in_batch(batch_samples)
        return batch_samples, batch_seqlens, num_items_in_batch

    def _get_num_items_in_batch(self, batch_samples):
        """
        Calculate the number of valid tokens in a batch (i.e., labels != -100).
        Aggregates this count across all ranks.
        """
        num_items_in_batch = None
        device = torch.accelerator.current_device()
        ps = ParallelState()

        # Check if 'labels' exist in the data
        count_num_items_in_batch = len(batch_samples) > 0 and "labels" in batch_samples[0]

        if count_num_items_in_batch:
            try:
                # Local sum of valid tokens
                num_items_in_batch = sum((batch["labels"].ne(-100)).sum().item() for batch in batch_samples)
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            # Distributed Aggregation
            if dist.is_initialized():
                num_items_tensor = torch.tensor(num_items_in_batch, device=device, dtype=torch.int64)
                # Note: Using all_reduce(SUM) is often more efficient than all_gather + sum
                dist.all_reduce(num_items_tensor, op=dist.ReduceOp.SUM, group=ps.get_group("dp_fsdp"))
                num_items_in_batch = num_items_tensor.item()

            # Adjustment for non-data-parallel ranks (e.g., pipeline parallelism)
            pc = getattr(self.args, "non_data_parallel_size", None)
            if pc:
                num_items_in_batch = num_items_in_batch // pc

        return num_items_in_batch

    def _compute_loss(self, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the loss for the batch.
        """
        args = self.args
        device = torch.accelerator.current_device()
        # 1. Inject num_items_in_batch into inputs if present (for token-weighted loss)
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch

        labels = inputs['labels'].to(device, non_blocking=True)
        if args.stage == 'pt':
            inputs['labels'] = None

        if self.optimization_args.chunk_loss_size and args.stage == 'pt':
            loss_ctx, loss_mask = self._build_chunk_loss(labels, chunk_size=self.optimization_args.chunk_loss_size)
            kwargs['loss_ctx'] = loss_ctx
            kwargs['loss_mask'] = loss_mask

        if (
            args.stage == 'pt'
            and self.optimization_args.chunk_loss_size
            and os.getenv("FSDP_TURBO_MTP_CHUNK_LOSS", "0").lower() in ("1", "true", "yes", "on")
            and getattr(getattr(self.model, "hf_config", None), "model_type", None) == "deepseek_v4"
        ):
            ps = ParallelState()
            cp_group = ps.get_group("cp") if ps.get_group_size("cp") > 1 else None
            mtp_labels = labels
            mtp_loss_ctx = []
            for _ in range(self.model.hf_config.num_nextn_predict_layers):
                # Match the existing external MTP loss: one extra next-token shift per layer.
                mtp_labels = roll_tensor(mtp_labels, fill_value=-100, cp_group=cp_group)
                context, _ = self._build_chunk_loss(
                    mtp_labels, chunk_size=self.optimization_args.chunk_loss_size
                )
                mtp_loss_ctx.append(context)
            kwargs['mtp_loss_ctx'] = mtp_loss_ctx

        if args.stage == 'pt' and getattr(args, 'router_aux_loss_coef', 0.0) > 0:
            # PT keeps labels outside the model, so request router outputs explicitly
            # and add the returned auxiliary loss below with the external LM loss.
            kwargs['output_router_logits'] = True
        # Merge inputs without modifying the original dictionary in-place
        model_inputs = {**inputs, **kwargs}

        # 2. Forward pass
        outputs = self.model(**model_inputs)

        lm_loss_output = torch.tensor(0.0, device=device, dtype=torch.float32)
        mtp_loss_output = torch.tensor(0.0, device=device, dtype=torch.float32)
        aux_loss_output = torch.tensor(0.0, device=device, dtype=torch.float32)
        aux_loss = getattr(outputs, 'aux_loss', None)
        if isinstance(aux_loss, torch.Tensor):
            aux_loss_output = aux_loss.clone()

        # 3. Extract loss from outputs
        if args.stage == 'pt' and "loss" not in outputs:
            logits = outputs.logits.contiguous().float()
            loss = self._compute_language_model_pretrain_loss(logits, labels, **kwargs)
            lm_loss_output = loss.clone()
            if isinstance(aux_loss, torch.Tensor):
                router_aux_loss_coef = getattr(args, 'router_aux_loss_coef', 0.0)
                loss = loss + router_aux_loss_coef * aux_loss.to(loss.device)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(f"Model outputs have no loss key: {list(outputs.keys())}")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            lm_loss_output = loss.clone()

        if args.stage == 'pt' and getattr(outputs, 'mtp_logits', None) is not None:
            mtp_losses = []
            for one_mtp_logits in outputs.mtp_logits:
                labels = roll_tensor(labels, fill_value=-100)
                mtp_loss = self._compute_language_model_pretrain_loss(
                    one_mtp_logits.contiguous().float(), labels, **kwargs
                )
                mtp_losses.append(mtp_loss)
        else:
            mtp_losses = getattr(outputs, 'mtp_loss', None)
        if mtp_losses:
            final_mtp_loss = 0
            for i in range(len(mtp_losses)):
                final_mtp_loss += mtp_losses[i] / len(mtp_losses)
            mtp_loss_output = final_mtp_loss.clone()
            loss += final_mtp_loss * getattr(args, 'mtp_loss_scaling_factor', 0.3)

        # 4. Cross-device token averaging adjustment
        # If the loss was calculated using 'mean' locally but needs global scaling based on tokens
        # sometimes we multiply by world size here so standard all_reduce(mean) works correctly.
        # This depends heavily on the specific loss function implementation.
        ps = ParallelState()
        if dist.is_initialized() and args.stage != 'pt':
            loss *= ps.get_group_size("dp_fsdp")
            lm_loss_output *= ps.get_group_size("dp_fsdp")
            if mtp_losses:
                final_mtp_loss = 0
                for i in range(len(mtp_losses)):
                    final_mtp_loss += mtp_losses[i] * ps.get_group_size("dp_fsdp") / len(mtp_losses)
                loss += final_mtp_loss * getattr(args, 'mtp_loss_scaling_factor', 0.3)
                mtp_loss_output *= ps.get_group_size("dp_fsdp")

        # 5. Return loss (or tuple of loss + outputs)
        model_outputs = outputs if return_outputs else None
        loss_result = LossOutput(
            loss=loss,
            lm_loss_output=lm_loss_output,
            mtp_loss_output=mtp_loss_output,
            aux_loss_output=aux_loss_output,
            outputs=model_outputs,
        )
        return loss_result

    def _compute_language_model_pretrain_loss(self, logits, labels, ignore_index: int = -100, **kwargs) -> torch.Tensor:
        args = self.args

        shift_labels = labels
        shift_labels = shift_labels.reshape(-1)
        logits = logits.view(-1, logits.shape[-1])

        ps = ParallelState()

        if ps.get_group_size("cp") > 1:
            loss = F.cross_entropy(logits, shift_labels, reduction='sum', ignore_index=ignore_index)

            num_items_in_batch = (labels.ne(ignore_index)).sum()
            dist.all_reduce(num_items_in_batch, op=dist.ReduceOp.SUM, group=ps.get_group("cp"))
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=ps.get_group("cp"))

            loss = loss / num_items_in_batch.item()
        else:
            if args.calculate_per_token_loss:
                loss = F.cross_entropy(logits, shift_labels, reduction='none', ignore_index=ignore_index)
            else:
                loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)

        return loss

    def _log_metrics(self, grad_norm=None, batch_size=0):
        """
        Logs training metrics:
        1. Calculate average loss since last log.
        2. Calculate throughput (elapsed time).
        3. Log current Grad Norm.
        """
        # Calculate step difference
        step_diff = self.global_step - self._global_step_last_logged
        if step_diff == 0:
            return

        # 1. Calculate average interval loss
        # (Total Loss - Total Loss at last log) / steps elapsed
        avg_loss = (self._total_loss_scalar - self._logging_loss_scalar) / step_diff

        # 2. Update logging cursor
        self._logging_loss_scalar = self._total_loss_scalar
        self._global_step_last_logged = self.global_step

        # 3. Calculate timing and throughput
        current_time = time.time()
        if self._step_start_time is None:
            elapsed_time_seconds = 0.0
        else:
            elapsed_time_seconds = current_time - self._step_start_time

        # Reset start time for next interval
        self._step_start_time = current_time

        # Avoid division by zero
        elapsed_time_per_iteration_ms = (elapsed_time_seconds / step_diff) * 1000
        throughput = (batch_size / (elapsed_time_seconds / step_diff)) if elapsed_time_seconds > 0 else 0.0

        # 4. Assemble metrics
        metrics = {
            "loss": avg_loss,
            "lr": self.lr_scheduler.get_last_lr()[0],
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        # 5. Log (Rank 0 only)
        # Note: consumed samples is estimated
        consumed_samples = self.global_step * batch_size
        logger.info_rank0(
            f"iteration: {self.global_step} | "
            f"consumed samples: {consumed_samples} | "
            f"elapsed time per iteration (ms): {elapsed_time_per_iteration_ms:.1f} | "
            f"throughput(samples/s): {throughput:.1f} | "
            f"learning rate: {metrics['lr']:.6E} | "
            f"global batch size: {batch_size:4d} | "
            f"lm loss: {avg_loss:.4f} | "
            f"grad norm: {grad_norm:.4f}"
        )

    def _get_fsdp_root(self):
        """
        Helper to get the inner FSDP module to access context managers like `no_sync`.
        Handles different wrapping depths.
        """
        # Direct check
        if isinstance(self.model, FSDP):
            return self.model

        # Check one level deep (standard DDP/Wrapper)
        if hasattr(self.model, "module"):
            if isinstance(self.model.module, FSDP):
                return self.model.module

            # Check two levels deep (complex wrapping)
            if hasattr(self.model.module, "model") and isinstance(self.model.module.model, FSDP):
                return self.model.module.model

        # Fallback to the model itself (context manager might fail if not FSDP, hence the check in caller)
        return self.model
