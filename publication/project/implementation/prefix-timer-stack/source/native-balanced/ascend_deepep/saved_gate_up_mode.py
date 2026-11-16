"""Host-only saved G/U policy, matching MakeMegaMoeHostTiling.

These helpers do not import torch or initialize a device.  The public limit is
a double; native code tests ``limit > 0`` before casting to FP32, and uses
FLT_MAX as the disabled-clipping sentinel.  Explicit modes never fall back.
"""
import math
import struct


SAVE_GMM1_OUTPUT_MODES = ('bf16', 'none', 'mxfp8', 'auto')
_FLOAT_MAX = float.fromhex('0x1.fffffep+127')


def validate_save_gmm1_output(mode):
    if mode not in SAVE_GMM1_OUTPUT_MODES:
        raise ValueError('save_gmm1_output must be bf16, none, mxfp8 or auto')
    return mode


def normalized_swiglu_limit(limit):
    value = float(limit)
    if not value > 0.0:
        return _FLOAT_MAX
    try:
        return struct.unpack('f', struct.pack('f', value))[0]
    except OverflowError:
        return math.inf


def resolve_save_gmm1_output(mode, limit):
    mode = validate_save_gmm1_output(mode)
    if mode == 'auto':
        return 'mxfp8' if normalized_swiglu_limit(limit) == _FLOAT_MAX else 'bf16'
    return mode


def saved_gate_up_metadata(mode, limit):
    """Describe the actual native format for an explicit capture/report."""
    resolved = resolve_save_gmm1_output(mode, limit)
    normalized = normalized_swiglu_limit(limit)
    masks = resolved == 'mxfp8' and normalized != _FLOAT_MAX
    return dict(requested=mode, resolved=resolved, normalized_limit=normalized,
                layout_version=2 if masks else 1,
                clip_mask_kind='three_bitplanes_v1' if masks else 'none')
