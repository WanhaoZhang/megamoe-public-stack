# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Centralized environment access used by distributed startup and feature flags."""

import os

_TRUE = frozenset({"1", "true", "yes", "on"})


def get_local_rank() -> int:
    """Return torchrun's local rank, or -1 outside a distributed launcher."""
    return int(os.environ.get("LOCAL_RANK", "-1"))


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag; ``1``/``true``/``yes``/``on`` (any case) are true.

    Unset variables keep ``default``. Any other value is false, which lets a
    caller disable a switch without deleting it from a launch script.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


__all__ = ["env_flag", "get_local_rank", "native_env", "launcher_env", "ensure_shmem_session"]


_NATIVE_DEFAULTS = {
    "MEGAMOE_SAVE_GMM1_OUTPUT": "bf16",
    "V4LAB_MEGAMOE_PEERMEM_LEGACY": "0",
    "ASCEND_DEEPEP_SHMEM_WINDOW_POOL": None,
    "ASCEND_DEEPEP_SHMEM_IP_PORT": "tcp://127.0.0.1:8899",
    "ASCEND_DEEPEP_SHMEM_TOTAL_BYTES": None,
    "ASCEND_DEEPEP_MAX_RECV_TOKENS": None,
    "ASCEND_ROUTER_IMBALANCE": "0",
    "V4LAB_MEGAMOE_DTYPE_DEBUG": "0",
    "ASCEND_DEEPEP_TOKEN_CHUNK_M": "1024",
    "ASCEND_DEEPEP_FWD_SYNC": "0",
    "ASCEND_DEEPEP_FWD_PRE_STREAM_SYNC": "0",
    "ASCEND_DEEPEP_FIXED_ROUTER_SPLIT_CACHE": "0",
    "ASCEND_DEEPEP_MEGAMOE_REDISPATCH": "0",
    "ASCEND_DEEPEP_MEGAMOE_REDISPATCH_VERIFY": "0",
    "ASCEND_DEEPEP_MEGAMOE_REDISPATCH_OVERLAP": "0",
    "ASCEND_DEEPEP_BWD_STREAM_SYNC": "1",
    "ASCEND_DEEPEP_MEGAMOE_BALANCED_DIAG": "0"
}

def native_env(name: str, *, configured_only: bool = False) -> str | None:
    """Read a retained Native compatibility setting with its centralized default."""
    if name not in _NATIVE_DEFAULTS:
        raise ValueError(f"Unknown Native environment setting: {name}")
    return os.environ.get(name, None if configured_only else _NATIVE_DEFAULTS[name])

def launcher_env(name: str) -> str | None:
    """Read an explicitly configured distributed launcher value."""
    if name not in ("MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE"):
        raise ValueError(f"Unknown launcher setting: {name}")
    return os.environ.get(name)

def ensure_shmem_session(endpoint: str) -> None:
    """Keep explicit caller sessions; otherwise derive the Native SDK session."""
    os.environ.setdefault("SHMEM_UID_SESSION_ID", endpoint.removeprefix("tcp://"))
