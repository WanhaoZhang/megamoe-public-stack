# Diagnostic clone patch

This patch adds gated prefix-Timer/plan-hash diagnostics and clears each
backward invocation's coordinate trace. It does not alter timed A/B sources.

Apply to a clean clone of the final research/step-lifetime tree. Git archives
and Linux checkouts use LF. On Windows, capture_diagnostics.py can have CRLF;
use `git apply --ignore-whitespace <patch-path>` for its unchanged context.
Index-only validation uses `git apply --check --cached <patch-path>`.
The new flags allow at most three diagnostic updates. Timer qualification
requires step0 gradients. Plan-hash diagnostics intentionally avoid main-shape
multi-GiB gradient dumps, and are never performance runs.
