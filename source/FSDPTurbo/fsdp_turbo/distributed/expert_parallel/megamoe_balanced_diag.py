# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Diagnostics for megamoe_balanced (off by default in training).

Enable with ``ASCEND_DEEPEP_MEGAMOE_BALANCED_DIAG=1`` or
``MegaMoeBalancedConfig.enable_diagnostics=True`` / MindSpeed
``--optimization.megamoe_balanced_enable_diagnostics``.
When disabled, balanced dispatch skips startup logs and the native pool
skips stats/audit/replay capture even if legacy ``WUHAN_*`` env vars are set.
"""

from __future__ import annotations

from fsdp_turbo.utils.env import native_env, launcher_env, ensure_shmem_session
from typing import Any

_DIAG_ENV = "ASCEND_DEEPEP_MEGAMOE_BALANCED_DIAG"


def _env_diag_enabled() -> bool:
    return native_env("ASCEND_DEEPEP_MEGAMOE_BALANCED_DIAG").strip().lower() in ("1", "true", "yes", "on")


def megamoe_balanced_diagnostics_enabled(config: Any | None = None) -> bool:
    if config is not None and getattr(config, "enable_diagnostics", False):
        return True
    return _env_diag_enabled()
