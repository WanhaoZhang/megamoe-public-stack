# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Device stream and event helpers, kept inside one API family."""

from __future__ import annotations

import torch


def device_stream_module(device_type: str):
    """Return the accelerator proxy installed by the framework startup."""
    return torch.accelerator


def _record_stream(tensor: torch.Tensor | None, stream) -> None:
    """Tie a tensor produced on the comm stream to the consuming stream."""
    if tensor is not None and stream is not None:
        tensor.record_stream(stream)
