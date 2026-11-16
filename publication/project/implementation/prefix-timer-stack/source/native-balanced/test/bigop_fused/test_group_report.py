import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import BF16_SMALL_ATOL, group_report  # noqa: E402


def test_all_small_nonzero_vectors_do_not_fail_only_on_cosine():
    ref = torch.tensor([1.0e-8, 0.0, 0.0], dtype=torch.float32)
    got = torch.tensor([0.0, 1.0e-8, 0.0], dtype=torch.float32)

    ok, status, detail = group_report("all-small", got, ref)

    assert ok
    assert status == "SUCCESS"
    assert "cos=0.00000000(not-gated:all-small)" in detail
    assert "small_bad=0/3" in detail


def test_all_small_absolute_error_regression_still_fails():
    ref = torch.zeros(128, dtype=torch.float32)
    got = torch.full_like(ref, 2.0 * BF16_SMALL_ATOL)

    ok, status, detail = group_report("all-small-bad", got, ref)

    assert not ok
    assert status == "Failed"
    assert "small_bad=128/128" in detail


def test_cosine_gate_remains_enabled_when_reference_has_normal_values():
    ref = torch.tensor([1.0, 1.0], dtype=torch.float32)
    got = torch.tensor([1.0, -1.0], dtype=torch.float32)

    ok, status, detail = group_report("normal", got, ref)

    assert not ok
    assert status == "Failed"
    assert "not-gated:all-small" not in detail
