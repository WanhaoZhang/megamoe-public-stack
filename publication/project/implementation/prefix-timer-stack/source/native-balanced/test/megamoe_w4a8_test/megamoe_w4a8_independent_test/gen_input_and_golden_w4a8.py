"""
W4A8 MegaMoE精度测试脚本 - 输入/Golden数据生成

W4A8配置：
- W (Weight): float4_e2m1 (4bit权重)
- A (Activation): float8_e4m3fn (8bit激活)

所有输入、输出及中间结果均通过.bin文件写出，以卡(card)为单位组织数据。
元信息(shape/dtype)保存到配套的json文件中。
"""

import argparse
import torch
import torch.nn.functional as F
import math
import numpy as np
import copy
import re
import importlib.metadata as importlib_metadata
from packaging import version
import subprocess
import sys
import os
import csv

# 导入工具模块
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import toolss_pro_w4a8 as toolss_pro

# ============================================================
# W4A8 量化配置
# ============================================================
WEIGHT_DTYPE = "float4_e2m1"       # W4: 权重使用 float4_e2m1
ACTIVATION_DTYPE = "float8_e4m3fn" # A8: 激活使用 float8_e4m3fn
SCALE_DTYPE = "float8_e8m0"        # 缩放因子使用 E8M0
QUANT_MODE = 4                     # 量化模式4 = MX量化(W4A8)
BLOCK_SIZE = 32                    # MX量化block大小

# ============================================================
# 辅助函数
# ============================================================

def swiglu(gmm_out):
    """SwiGLU激活函数"""
    c_temp4, gate = gmm_out.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU激活
    c_temp6 = c_temp5 * gate
    return c_temp6


def numpy_float4_e2m1():
    try:
        from en_dtypes import float4_e2m1
        return float4_e2m1
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed for float4_e2m1 dtype! pip3 install en-dtypes")


def numpy_float8_e4m3fn():
    try:
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed for float8_e4m3fn dtype! pip3 install ml-dtypes")


def numpy_float8_e8m0():
    try:
        from en_dtypes import float8_e8m0
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError("en_dtypes is needed for float8_e8m0 dtype! pip3 install en-dtypes")


def golden_output_process(output, param_split_item, groupType):
    golden = []
    if param_split_item in [2, 3]:
        if groupType == 0:
            ret = torch.cat(output, dim=-2)
            golden.append(ret)
        elif groupType == 2:
            ret = torch.stack(output, dim=0)
            golden.append(ret)
        else:
            print("unsupported group type, please check")
        num_inf = torch.isinf(golden[0]).sum().item()
        num_nan = torch.isnan(golden[0]).sum().item()
        ratio = 100 * ((num_inf + num_nan) / (ret.shape[0] * ret.shape[1])) if ret.shape[0] * ret.shape[1] != 0 else 0
        print(f"---INF_num = {num_inf}, NAN_num = {num_nan}, Ratio = {ratio} %")
    else:
        golden = output
    return golden


def transpose_last_two_dim(input, transpose_flag):
    if isinstance(input, torch.Tensor):
        if transpose_flag == 1:
            ndim = input.dim()
            if ndim < 2:
                raise ValueError("Tensor must have at least 2 dimensions.")
            new_dims = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
            return input.permute(*new_dims)
        else:
            return input
    elif isinstance(input, np.ndarray):
        if transpose_flag == 1:
            ndim = input.ndim
            axes = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
            return np.transpose(input, axes=axes)
        else:
            return input
    else:
        raise TypeError(f"unsupported data type: {type(input)}")


def regenerate_quant_data(w_shape, params_out_dtype):
    mu_W = -0.05
    Sigma_W = 4
    W_tmp = (np.random.normal(mu_W, Sigma_W, w_shape) - np.random.uniform(0.1 * mu_W, -0.1 * mu_W, w_shape)).astype(np.float32)
    W_tmp = np.clip(W_tmp, 0, 0.1)
    if params_out_dtype == 'fp32':
        W_tmp = np.abs(W_tmp)
    return W_tmp


FP4_E2M1_LUT = np.array(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=np.float32,
)


def _make_expert_weight_rng(random_seed, global_expert_id, weight_index):
    """Create a deterministic RNG independent of generation order."""
    random_seed = int(random_seed)
    global_expert_id = int(global_expert_id)
    return np.random.default_rng(np.random.SeedSequence([
        random_seed & 0xFFFFFFFF,
        (random_seed >> 32) & 0xFFFFFFFF,
        global_expert_id & 0xFFFFFFFF,
        (global_expert_id >> 32) & 0xFFFFFFFF,
        int(weight_index),
    ]))


def _unpack_mxfp4_e2m1(packed, logical_last_dim):
    """Decode packed E2M1 nibbles to FP32 for the CPU golden path."""
    unpacked = np.empty(
        packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float32
    )
    unpacked[..., 0::2] = FP4_E2M1_LUT[packed & 0x0F]
    unpacked[..., 1::2] = FP4_E2M1_LUT[packed >> 4]
    return unpacked[..., :logical_last_dim]


def fast_mxfp4_random_data_prepare(
        num_experts, input_dim, output_dim, weight_transpose_flag,
        random_seed, weight_index, generate_dequant=True):
    """Generate quantized W4 storage directly, matching the HaiSi fast path.

    This deliberately skips FP32 random-weight construction and MXFP4 LUT
    quantization. Each packed nibble is already a valid E2M1 storage code and
    every MX block uses scale 1. The optional FP32 matrix is decoded from those
    exact storage codes, so the CPU golden and NPU input remain bit-consistent.
    """
    if weight_transpose_flag not in (0, 1):
        raise ValueError(
            f"weight_transpose_flag must be 0 or 1, got {weight_transpose_flag}"
        )

    if weight_transpose_flag == 1:
        packed_shape = (num_experts, output_dim, (input_dim + 1) // 2)
    else:
        packed_shape = (num_experts, input_dim, (output_dim + 1) // 2)

    w_npu = np.empty(packed_shape, dtype=np.uint8)
    dequant_matrix = (
        np.empty((num_experts, input_dim, output_dim), dtype=np.float32)
        if generate_dequant else None
    )

    for global_expert_id in range(num_experts):
        rng = _make_expert_weight_rng(
            random_seed, global_expert_id, weight_index
        )
        # HaiSi masks random E2M1 storage codes with 0x0B. Apply the same mask
        # independently to both packed nibbles.
        low = rng.integers(
            0, 16, size=packed_shape[1:], dtype=np.uint8
        ) & np.uint8(0x0B)
        high = rng.integers(
            0, 16, size=packed_shape[1:], dtype=np.uint8
        ) & np.uint8(0x0B)
        packed = low | (high << np.uint8(4))
        w_npu[global_expert_id] = packed

        if generate_dequant:
            logical_last_dim = (
                input_dim if weight_transpose_flag == 1 else output_dim
            )
            decoded = _unpack_mxfp4_e2m1(packed, logical_last_dim)
            if weight_transpose_flag == 1:
                decoded = decoded.T
            dequant_matrix[global_expert_id] = decoded

    scale_block_num = (input_dim + BLOCK_SIZE - 1) // BLOCK_SIZE
    if scale_block_num % 2 != 0:
        scale_block_num += 1
    scale_npu = np.ones(
        (num_experts, scale_block_num, output_dim), dtype=np.float32
    ).astype(numpy_float8_e8m0(), copy=False)

    return w_npu, scale_npu, dequant_matrix


def qbmm_mx_golden_calculate(x, weight, pertoken_scale, has_bias=0, bias=None, bias_dtype=None, offset=None, act_type=0):
    """MX量化golden计算：weight已预反量化，直接matmul"""
    output = [None] * len(x)
    for i in range(len(x)):
        if isinstance(x[i], torch.Tensor):
            x1 = x[i].numpy().astype(np.float32)
        else:
            x1 = x[i].astype(np.float32)

        if pertoken_scale is not None:
            if isinstance(pertoken_scale[i], torch.Tensor):
                pertoken_scale_i = pertoken_scale[i].numpy()
            else:
                pertoken_scale_i = pertoken_scale[i]
            pertoken_scale_mx_broadcast = np.repeat(pertoken_scale_i, 32, axis=-1)
            x1_dims = len(x1.shape)
            x1_pad_len = pertoken_scale_mx_broadcast.shape[-1] - x1.shape[-1]
            x1 = np.pad(x1, [(0, 0)] * (x1_dims - 1) + [(0, x1_pad_len)], mode='constant', constant_values=0)
            x1 = x1 * pertoken_scale_mx_broadcast

        if isinstance(weight[i], torch.Tensor):
            x2 = weight[i].numpy().astype(np.float32)
        else:
            x2 = weight[i].astype(np.float32)

        x1 = torch.from_numpy(x1).to(torch.float)
        x2 = torch.from_numpy(x2).to(torch.float)

        output[i] = torch.matmul(x1, x2)
        if has_bias == 1:
            output[i] = torch.add(output[i], bias[i])
    return output


# ============================================================
# MX量化核心函数 (dispatch_mx_quantize)
# ============================================================

def dispatch_mx_quantize(fp_array, mx_ele_dtype="float4_e2m1",
                          axis=-1, block_size=32, round_mode="rint"):
    """
    量化FP16/BF16/FP32到MX dtype
    支持: float4_e2m1, float4_e1m2, float8_e4m3fn, float8_e5m2
    返回: (量化后数据, scale数据)
    """
    def get_dtype_range(dt):
        if "float4_e2m1" in str(dt):
            return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
        if "float4_e1m2" in str(dt):
            return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
        if "float8_e8m0" in str(dt):
            return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
        if "float8_e5m2" in str(dt):
            return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
        if "float8_e4m3fn" in str(dt):
            return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
        numpy_dtype = np.dtype(dt)
        if numpy_dtype.kind in "iu":
            numpy_info = np.iinfo(numpy_dtype)
        else:
            numpy_info = np.finfo(numpy_dtype)
        return numpy_info.min, numpy_info.max

    def _mx_reshape_to_blocks(fp_array, axis, block_size):
        fp_array = np.expand_dims(fp_array, axis=axis + 1)
        orig_shape = fp_array.shape
        pad = [[0, 0] for _ in range(len(orig_shape))]
        pad_size = orig_shape[axis] % block_size
        pad[axis][1] = block_size - pad_size
        if pad_size > 0:
            fp_array = np.pad(fp_array, pad, 'constant')
        padded_shape = fp_array.shape
        reshape = list(padded_shape)
        reshape[axis + 1] = block_size
        reshape[axis] = reshape[axis] // block_size
        fp_array = fp_array.reshape(reshape)
        return fp_array, orig_shape, padded_shape, pad_size

    def _mx_undo_reshape_to_blocks(fp_array, axis, orig_shape, padded_shape):
        fp_array = fp_array.reshape(padded_shape)
        if tuple(padded_shape) != tuple(orig_shape):
            slices = [slice(0, x) for x in orig_shape]
            fp_array = fp_array[tuple(slices)]
        fp_array = np.squeeze(fp_array, axis=axis + 1)
        return fp_array

    def _mx_calculate_share_exp(fp_array, scale_axis, mx_ele_dtype):
        FP32_EXPONENT_BIAS = 127
        FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
        ele_emax = 0
        mx_dtype = str(mx_ele_dtype)
        if "float4_e2m1" in mx_dtype:
            ele_emax = 2
        elif "float8_e4m3fn" in mx_dtype:
            ele_emax = 8
        elif "float8_e5m2" in mx_dtype:
            ele_emax = 15
        fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
        res = np.floor(
            np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
        ) - ele_emax
        res[fp_abs_max == 0] = -float("inf")
        return res

    def _mx_round_mantissa(fp_array, round_mode):
        if round_mode in ("rint", "even"):
            fp_array = np.rint(fp_array)
        elif round_mode in ("round", "nearest"):
            fp_array = np.sign(fp_array) * np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        elif round_mode == "floor":
            fp_array = np.floor(fp_array)
        elif round_mode == "ceil":
            fp_array = np.ceil(fp_array)
        elif round_mode == "trunc":
            fp_array = np.trunc(fp_array)
        else:
            raise Exception(f"Unrecognized round method {round_mode}")
        return fp_array

    def _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode):
        mx_dtype = str(mx_ele_dtype)
        match = re.search(r'e(\d+)m(\d+)', mx_dtype)
        if match:
            exp_bits = int(match.group(1))
            mantissa_bits = int(match.group(2))
        else:
            raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

        ret = fp_array / (2 ** share_exp)
        private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))).astype(fp_array.dtype, copy=False)
        min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
        ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
        ret = _mx_round_mantissa(ret, round_mode)
        ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
        max_norm = get_dtype_range(mx_dtype)[1]
        np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        return ret

    def pad_to_even(tensor, axis):
        shape = tensor.shape
        length = shape[axis]
        if length % 2 == 0:
            return tensor
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[axis] = (0, 1)
        padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
        return padded_tensor

    if not isinstance(fp_array, np.ndarray):
        raise RuntimeError(f"Input should be numpy array. Got {type(fp_array)}")
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(f"Input dtype not supported: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape, pad_size = _mx_reshape_to_blocks(fp_array, axis, block_size)
    share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype)
    scale_emax = 2 ** (8 - 1) - 1
    share_exp[share_exp > scale_emax] = float("NaN")

    # 量化到element format
    quant_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)

    # 恢复原始形状
    quant_array = _mx_undo_reshape_to_blocks(quant_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)

    # 转换为目标dtype
    ele_dtype_np = numpy_float4_e2m1() if mx_ele_dtype == "float4_e2m1" else (
        numpy_float4_e1m2() if mx_ele_dtype == "float4_e1m2" else (
            numpy_float8_e4m3fn() if mx_ele_dtype == "float8_e4m3fn" else numpy_float8_e5m2()
        )
    )
    scale_array = 2 ** share_exp
    if quant_array.dtype.name == "bfloat16":
        quant_array = quant_array.astype("float32", copy=False)
    # NPU会将NaN转为正零
    quant_array = np.nan_to_num(quant_array, nan=0.0, copy=False)
    quant_array = quant_array.astype(ele_dtype_np, copy=False)
    # Scale需要pad到偶数长度
    scale_array_pad = pad_to_even(scale_array, axis=axis)
    scale_e8m0 = scale_array_pad.astype(numpy_float8_e8m0(), copy=False)

    return quant_array, scale_e8m0


def mx_quantize(fp_array, mx_ele_dtype="float8_e4m3fn", axis=-1, block_size=32, round_mode="rint"):
    """MX量化封装，返回(量化数据, scale)"""
    return dispatch_mx_quantize(fp_array, mx_ele_dtype, axis, block_size, round_mode)

def convert_nd_to_zN_int8(input):

    # numpy -> torch
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    b, n, k = input.shape

    n_tile = 16
    k_tile = 16

    n_loop = (n + n_tile - 1) // n_tile
    k_loop = (k + k_tile - 1) // k_tile

    n_align = n_loop * n_tile
    k_align = k_loop * k_tile

    src_pad = F.pad(
        input,
        (0, k_align-k,
         0, n_align-n)
    )

    zN_w = src_pad.reshape(
        b,
        n_align,
        k_loop,
        k_tile
    ).permute(
        0,
        2,
        1,
        3
    )

    return zN_w.numpy()


def transform_mxfp4_scale_for_npu(scale, transpose_flag):
    """参考gen_data.py的scale写回格式: reshape + permute

    将scale的block维度配对为2一组, 并根据转置标志排列维度。
    gen_data.py中的变换:
      b_scale.reshape(num_blocks//2, 2, n) → permute based on trans

    Args:
        scale: 3D numpy array (E, num_blocks, N), float8_e8m0
            始终为行方向量化(axis=0): (E, input/32, output)
        transpose_flag: 1=转置, 0=不转置

    Returns:
        变换后的scale: 4D numpy array, 最后一维为2(配对维度)
            transpose_flag=1: (E, N, num_blocks//2, 2) = (E, output, input/32//2, 2)
            transpose_flag=0: (E, num_blocks//2, N, 2) = (E, input/32//2, output, 2)
    """
    # scale始终为(E, num_blocks, N) = (E, input/32, output)格式 (axis=0量化)
    E, num_blocks, N = scale.shape
    scale = scale.reshape(E, num_blocks // 2, 2, N)
    if transpose_flag == 1:
        # → (E, N, num_blocks//2, 2) = (E, output, input/32//2, 2)
        return scale.transpose(0, 3, 1, 2)
    else:
        # → (E, num_blocks//2, N, 2) = (E, input/32//2, output, 2)
        return scale.transpose(0, 1, 3, 2)


# def pack_mxfp4_to_uint8(mxfp4_data):
#     """将2个float4_e2m1值打包为1个uint8: 偶数索引放低4位，奇数索引放高4位
#     输入: mxfp4_data, dtype=float4_e2m1, itemsize=1 (每元素1字节，仅低4位有效)
#     输出: packed, dtype=uint8, 最后一维长度减半
#     """
#     raw = mxfp4_data.view(np.uint8)
#     # float4_e2m1 每元素1字节中只有低4位是有效数据
#     low_nibble = raw[..., 0::2] & 0x0F   # 偶数索引，取低4位
#     high_nibble = raw[..., 1::2] & 0x0F   # 奇数索引，取低4位
#     packed = low_nibble | (high_nibble << 4)
#     convert_nd_to_zN_int8(packed)
#     return packed


def qbmm_mxfp4_quant_data_prepare(
        W_tmp, w_dtype="float4_e2m1", weight_transpose_flag=1,
        generate_dequant=True):
    """W4权重量化数据准备 - 使用MXFP4MatrixQuantizer (参考gen_data.py的gen_data_fp4_e2m1)

    与dispatch_mx_quantize不同，MXFP4MatrixQuantizer使用查找表进行最近邻量化，
    且仅支持2D矩阵，因此对3D权重(experts, M, N)按expert循环处理。

    参考gen_data.py的gen_data_fp4_e2m1，只返回3个值:
        w_npu:          uint8 packed权重 (2个fp4打包为1个uint8, 转置(如需)后打包, 用于NPU bin写出)
        scale_npu:      float8_e8m0 scale (用于NPU bin写出)
        dequant_matrix: 反量化矩阵 (转置(如需)以匹配golden matmul维度)
    """
    from gen_data import MXFP4MatrixQuantizer

    # axis映射: transpose_flag=0 → axis=0 (行方向), transpose_flag=1 → axis=1 (列方向)
    quant_axis = 0  # 始终沿input方向(rows)量化, 参考gen_data.py的gen_data_fp4_e2m1(k, n, 0, trans)

    # W_tmp是3D (num_experts, M, N), MXFP4MatrixQuantizer只处理2D, 需要循环每个expert
    num_experts = W_tmp.shape[0]

    w_npu_list = []        # uint8 packed (用于NPU bin)
    scale_npu_list = []    # e8m0 scale (用于NPU bin)
    dequant_list = []      # 反量化矩阵 (用于golden)

    for i in range(num_experts):
        # numpy → torch tensor
        matrix = torch.from_numpy(W_tmp[i]).float()

        # 创建量化器 (参考gen_data_fp4_e2m1)
        quantizer = MXFP4MatrixQuantizer(data_format='E2M1', axis=quant_axis, block_size=BLOCK_SIZE)

        # 量化: 返回float32量化矩阵和float32 scale矩阵
        quantized_matrix, scale_matrix = quantizer.quantize_matrix(matrix)

        if generate_dequant:
            # 反量化仅供CPU golden矩阵乘使用。
            dequantized_matrix = quantizer.dequantize_matrix(
                quantized_matrix, scale_matrix)

        # NPU数据: 参考gen_data_fp4_e2m1, 转置(如需)后再打包
        if weight_transpose_flag == 1:
            npu_matrix = quantized_matrix.t()
        else:
            npu_matrix = quantized_matrix
        w_npu = quantizer.matrix_fp32_to_fp4(npu_matrix).numpy()
        scale_npu = scale_matrix.numpy().astype(numpy_float8_e8m0())

        w_npu_list.append(w_npu)
        scale_npu_list.append(scale_npu)
        if generate_dequant:
            # golden用反量化矩阵: 已是(input, output)格式, 无需转置。
            dequant_list.append(dequantized_matrix.numpy().astype(np.float32))

    # 堆叠回3D
    w_npu = np.stack(w_npu_list, axis=0)
    scale_npu = np.stack(scale_npu_list, axis=0)
    dequant_matrix = np.stack(dequant_list, axis=0) if generate_dequant else None

    return w_npu, scale_npu, dequant_matrix


# ============================================================
# GenInputAndGolden_W4A8 - W4A8精度数据生成类
# ============================================================

class GenInputAndGolden_W4A8():
    def __init__(self, **kwargs):
        self.bs = kwargs['bs']
        self.h = kwargs['h']
        self.n = kwargs['n']
        self.quant_dtype = kwargs['quant_dtype']        # dispatch量化dtype = float8_e4m3fn (A8)
        self.is_x_random = kwargs['is_x_random']
        self.global_bs = kwargs['global_bs']
        self.out_dtype = kwargs['out_dtype']
        self.moe_expert_num = kwargs['moe_expert_num']
        self.is_topk_balance = kwargs['is_topk_balance']
        self.k = kwargs['k']
        self.is_mask = kwargs['is_mask']
        self.is_unfixed_real_token = kwargs['is_unfixed_real_token']
        self.ep_world_size = kwargs['ep_world_size']
        self.is_variable_bs = kwargs['is_variable_bs']
        self.expert_scales_dtype = kwargs['expert_scales_dtype']
        self.local_moe_expert_num = kwargs['local_moe_expert_num']
        self.quant_mode = kwargs['quant_mode']
        self.transpose_weight = kwargs['transpose_weight']
        self.comm_quant_mode = kwargs['comm_quant_mode']
        self.weight_dtype = kwargs['weight_dtype']      # W4 = float4_e2m1
        self.swiglu_quant_dtype = kwargs['swiglu_quant_dtype']  # A8 = float8_e4m3fn
        self.skip_golden = kwargs.get('skip_golden', False)
        self.fast_random_weight = kwargs.get('fast_random_weight', True)
        self.random_seed = int(kwargs.get('random_seed', 831))
        # dispatch_map txt 仅供调试，三层嵌套循环开销大，默认关闭。
        # 通过环境变量 MEGA_MOE_W4A8_GEN_DISPATCH_MAP=1 开启。
        self.gen_dispatch_map = kwargs.get(
            'gen_dispatch_map',
            os.environ.get('MEGA_MOE_W4A8_GEN_DISPATCH_MAP', '0') == '1',
        )

        self.tag = f"w4a8_{self.bs}_{self.h}_{self.n}_{self.ep_world_size}"

        # 初始化数据目录
        toolss_pro.init_data_path()

        # 生成所有输入数据和golden结果
        print("=" * 60)
        print("Step 1: Generate Input Data (x, expert_ids, expert_scales)")
        print("=" * 60)
        self.x = self.gen_x()
        self.expert_ids = self.gen_expert_ids()
        self.x_active_mask = self.gen_x_active_mask()
        self.expert_scales = self.gen_expert_scales()

        if self.skip_golden:
            print("=" * 60)
            print("Step 2: Generate W4A8 Weight Inputs (CPU golden skipped)")
            print("=" * 60)
            self.gen_ffn_golden()
        else:
            print("=" * 60)
            print("Step 2: Generate Dispatch Golden (reorder + quantize)")
            print("=" * 60)
            self.a_reordered_list, self.token_indices_list, self.prob_list, self.a_scale_list, self.expert_token_nums = \
                self.gen_dispatch_golden(torch.cat(self.expert_ids, dim=0), torch.cat(self.x, dim=0))

            print("=" * 60)
            print("Step 3: Generate FFN Golden (GMM1 + SwiGLU + Quant + GMM2)")
            print("=" * 60)
            self.c2_fp32_list, self.weight1, self.weight2, self.weight_scales1, self.weight_scales2 = self.gen_ffn_golden()

            print("=" * 60)
            print("Step 4: Generate Combine Golden (final output)")
            print("=" * 60)
            self.x_final = self.gen_combine_golden(torch.cat(self.expert_scales, dim=0), torch.cat(self.expert_ids, dim=0))

        print("=" * 60)
        if self.gen_dispatch_map:
            print("Step 5: Generate Dispatch Map (token dispatch across cards)")
            print("=" * 60)
            self.gen_dispatch_map_txt()
        else:
            print("Step 5: Generate Dispatch Map skipped (set "
                  "MEGA_MOE_W4A8_GEN_DISPATCH_MAP=1 to enable)")
            print("=" * 60)

        print("=" * 60)
        if self.skip_golden:
            print("All input data generated; CPU golden calculation skipped!")
        else:
            print("All data generated and saved to bin files!")
        print(f"Data directory: {toolss_pro.DATA_PATH}")
        toolss_pro.list_all_bin_files()
        print("=" * 60)

    def chunk_tensor(self, tensor, chunks):
        return list(tensor.chunk(chunks))

    def expert_to_card(self, expert_id):
        """expert_id -> card_rank"""
        return expert_id // self.local_moe_expert_num

    def card_experts(self, card_rank):
        """card_rank上的expert_id列表"""
        start = card_rank * self.local_moe_expert_num
        return list(range(start, start + self.local_moe_expert_num))

    # ========================================================
    # Step 1: 输入数据生成（按卡写出）
    # ========================================================

    def gen_x(self):
        """生成输入x数据，按卡写出"""
        if self.is_x_random:
            cur_x = torch.empty(size=[self.global_bs, self.h], dtype=self.out_dtype).uniform_(0, 0.1)
        else:
            cur_x = torch.arange(1, self.global_bs + 1, dtype=self.out_dtype).view(-1, 1).repeat(1, self.h)

        # 按卡写出输入x
        x_per_card = self.chunk_tensor(cur_x, self.ep_world_size)
        for c in range(self.ep_world_size):
            toolss_pro.write_tensor_to_bin(x_per_card[c], f"rank{c}_input_x_{self.tag}")

        return x_per_card

    def gen_expert_ids(self):
        """生成expert_ids，按卡写出"""
        def gen_balanced_expert_ids_element(round_index):
            return torch.arange(round_index, round_index + self.moe_expert_num, dtype=torch.int32) % self.moe_expert_num

        def gen_unbalanced_expert_ids():
            hot_expert_count = min(8, self.moe_expert_num)
            warm_expert_count = min(
                16, max(self.moe_expert_num - hot_expert_count, 0)
            )
            global_cold_expert_ratio = 0.005
            rank_cold_expert_ratio = 0.015
            expert_weights = torch.ones(
                self.moe_expert_num, dtype=torch.float32
            )
            routing_scale = self.moe_expert_num / max(self.k, 1)
            expert_weights[0] = 20.0 * routing_scale
            if hot_expert_count > 1:
                expert_weights[1:hot_expert_count] = torch.logspace(
                    math.log10(max(1.25 * routing_scale, 1.0)),
                    math.log10(max(0.42 * routing_scale, 1.0)),
                    hot_expert_count - 1,
                )
            warm_expert_start = hot_expert_count
            warm_expert_end = warm_expert_start + warm_expert_count
            expert_weights[warm_expert_start:warm_expert_end] = max(
                0.21 * routing_scale, 1.0
            )
            global_cold_expert_count = min(
                round(self.moe_expert_num * global_cold_expert_ratio),
                max(self.moe_expert_num - warm_expert_end, 0),
                max(self.moe_expert_num - self.k, 0),
            )
            if global_cold_expert_count > 0:
                expert_weights[-global_cold_expert_count:] = 0

            ordinary_expert_end = self.moe_expert_num - global_cold_expert_count
            ordinary_expert_indices = torch.arange(
                warm_expert_end, ordinary_expert_end
            )
            rank_cold_expert_count = min(
                round(self.moe_expert_num * rank_cold_expert_ratio),
                ordinary_expert_indices.numel(),
                max(
                    self.moe_expert_num
                    - self.k
                    - global_cold_expert_count,
                    0,
                ),
            )
            expert_id_generator = torch.Generator().manual_seed(831)
            expert_permutation = torch.randperm(
                self.moe_expert_num, generator=expert_id_generator
            )
            rank_expert_ids = []
            sample_chunk_size = 4096
            for _ in range(self.ep_world_size):
                rank_expert_weights = expert_weights.clone()
                if rank_cold_expert_count > 0:
                    rank_cold_indices = ordinary_expert_indices[
                        torch.randperm(
                            ordinary_expert_indices.numel(),
                            generator=expert_id_generator,
                        )[
                            :rank_cold_expert_count
                        ]
                    ]
                    rank_expert_weights[rank_cold_indices] = 0
                rank_expert_weights = rank_expert_weights[expert_permutation]
                sampled_chunks = []
                for start in range(0, self.bs, sample_chunk_size):
                    current_bs = min(sample_chunk_size, self.bs - start)
                    sampled_chunks.append(torch.multinomial(
                        rank_expert_weights.expand(current_bs, -1),
                        self.k,
                        replacement=False,
                        generator=expert_id_generator,
                    ).to(torch.int32))
                rank_expert_ids.append(torch.cat(sampled_chunks, dim=0))
            return torch.cat(rank_expert_ids, dim=0)

        if self.is_topk_balance:
            ep_expert_ids = [gen_balanced_expert_ids_element(round_index)
                             for round_index in range((self.global_bs * self.k) // self.moe_expert_num + 1)]
            expert_ids = torch.cat(ep_expert_ids)[:self.global_bs * self.k].view([self.global_bs, self.k])
        else:
            expert_ids = gen_unbalanced_expert_ids()
            expert_counts = torch.bincount(
                expert_ids.reshape(-1).to(torch.long),
                minlength=self.moe_expert_num,
            )
            max_expert_hits = int(expert_counts.max().item())
            average_expert_hits = float(
                expert_counts.to(torch.float64).mean().item()
            )
            print(
                f"[维测] 热点专家命中统计: 最大值={max_expert_hits}, "
                f"平均值={average_expert_hits:.2f}, "
                f"最大值/平均值={max_expert_hits / average_expert_hits:.2f}"
            )

        # 按卡写出expert_ids
        expert_ids_per_card = self.chunk_tensor(expert_ids, self.ep_world_size)
        for c in range(self.ep_world_size):
            toolss_pro.write_tensor_to_bin(expert_ids_per_card[c], f"rank{c}_input_expert_ids_{self.tag}")

        return expert_ids_per_card

    def gen_x_active_mask(self):
        """生成x_active_mask，按卡写出"""
        min_real_token_num = 0 if self.is_mask else 1
        real_token_num = self.bs if not self.is_unfixed_real_token else \
            torch.randint(min_real_token_num, self.bs + 1, (1,)).item()

        x_active_mask = [torch.cat((torch.ones(real_token_num), torch.zeros(self.bs - real_token_num))).to(torch.bool)
                         for _ in range(self.ep_world_size)]

        # 按卡写出x_active_mask
        for c in range(self.ep_world_size):
            toolss_pro.write_tensor_to_bin(x_active_mask[c], f"rank{c}_input_x_active_mask_{self.tag}")

        return x_active_mask

    def gen_expert_scales(self):
        """生成expert_scales，按卡写出（bf16格式）"""
        expert_scales = torch.empty(size=[self.global_bs, self.k], dtype=torch.bfloat16).uniform_(-0.1, 0.1)

        # 按卡写出expert_scales (bf16)
        expert_scales_per_card = self.chunk_tensor(expert_scales, self.ep_world_size)
        for c in range(self.ep_world_size):
            toolss_pro.write_tensor_to_bin(expert_scales_per_card[c], f"rank{c}_input_expert_scales_{self.tag}")

        return expert_scales_per_card

    # ========================================================
    # Step 2: Dispatch Golden生成（按卡写出）
    # ========================================================

    def gen_dispatch_golden(self, expert_ids, x):
        """生成dispatch golden结果，按卡写出"""
        w_dtype = self.quant_dtype  # A8 = float8_e4m3fn
        a_reordered_list = []
        token_indices_list = []
        prob_list = []

        input_tokens_fp32 = x.to(torch.float32)

        expert_token_nums = []
        current_row = 0

        for g in range(self.moe_expert_num):
            token_indices = []
            group_probs = []

            for idx in range(self.global_bs):
                g_positions = torch.where(expert_ids[idx] == g)[0]
                for pos in g_positions:
                    token_indices.append(idx)
                    group_probs.append(expert_ids[idx, pos])

            token_indices = torch.tensor(token_indices, dtype=torch.long)

            if len(token_indices) > 0:
                a_reordered = input_tokens_fp32[token_indices]
                a_reordered_list.append(a_reordered)
                token_indices_list.append(token_indices)
                prob_list.append(torch.tensor(group_probs, dtype=torch.float32))
            else:
                a_reordered_list.append(torch.empty(0, self.h, dtype=torch.float32))
                token_indices_list.append(torch.empty(0, dtype=torch.long))
                prob_list.append(torch.empty(0, dtype=torch.float32))

        expert_token_nums = [(expert_ids == i).sum().item() for i in range(self.moe_expert_num)]
        expert_token_nums_matrix = torch.tensor(
            expert_token_nums, dtype=torch.int64
        ).view(self.ep_world_size, self.local_moe_expert_num)
        print("\n=== Expert Token Count Matrix ===")
        print("rows: destination EP rank")
        print("columns: local expert index")
        print("global expert id = rank * local_moe_expert_num + local expert index")
        print(expert_token_nums_matrix.numpy())
        print(f"total routed tokens: {expert_token_nums_matrix.sum().item()}\n")

        # 按卡写出expert_token_nums
        for c in range(self.ep_world_size):
            experts_on_card = self.card_experts(c)
            card_token_nums = torch.tensor([expert_token_nums[g] for g in experts_on_card], dtype=torch.int64)
            toolss_pro.write_tensor_to_bin(card_token_nums, f"rank{c}_dispatch_expert_token_nums_{self.tag}")

        if self.quant_mode == 4:
            a_quant_list = []
            a_scale_list = []
            for g in range(self.moe_expert_num):
                if a_reordered_list[g].shape[0] > 0:
                    # A8量化: float8_e4m3fn
                    a_quant_g, a_scale_g = dispatch_mx_quantize(
                        a_reordered_list[g].numpy(), mx_ele_dtype=w_dtype,
                        axis=-1, block_size=BLOCK_SIZE, round_mode="rint")
                    a_quant_list.append(a_quant_g)
                    a_scale_list.append(a_scale_g)
                else:
                    a_quant_list.append(np.empty((0, self.h), dtype=np.float32))
                    a_scale_list.append(np.empty((0, (self.h + 31) // 32), dtype=np.float32))

            # 按卡写出quant数据：token和scale合并写成一个bin文件
            # 每个token的layout: [token_a8(h), token_scale(ceil(h/32)*2)]
            for c in range(self.ep_world_size):
                experts_on_card = self.card_experts(c)
                card_quant = [a_quant_list[g] for g in experts_on_card if a_quant_list[g].shape[0] > 0]
                card_scale = [a_scale_list[g] for g in experts_on_card if a_scale_list[g].shape[0] > 0]
                if card_quant:
                    card_quant_all = np.concatenate(card_quant, axis=0)
                    card_scale_all = np.concatenate(card_scale, axis=0)
                    # 合并: 每个token的前面是quant数据，后面是scale数据
                    # card_quant_all dtype: float8_e4m3fn (itemsize=1)
                    # card_scale_all dtype: float8_e8m0 (itemsize=1)
                    # 先转为uint8再合并，避免dtype不匹配
                    quant_uint8 = card_quant_all.view(np.uint8)
                    scale_uint8 = card_scale_all.view(np.uint8)
                    combined = np.concatenate([quant_uint8, scale_uint8], axis=-1)
                    toolss_pro.write_tensor_to_bin(combined, f"rank{c}_dispatch_quant_combined_{self.tag}")
                else:
                    combined = np.empty((0, self.h + (self.h + 31) // 32 * 2), dtype=np.float32)
                    toolss_pro.write_tensor_to_bin(combined, f"rank{c}_dispatch_quant_combined_{self.tag}")

            return a_quant_list, token_indices_list, prob_list, a_scale_list, \
                self.chunk_tensor(torch.tensor(expert_token_nums), self.ep_world_size)

        return a_reordered_list, token_indices_list, prob_list, None, \
            self.chunk_tensor(torch.tensor(expert_token_nums), self.ep_world_size)

    # ========================================================
    # Step 3: FFN Golden生成 (GMM1 + SwiGLU + GMM2)（按卡写出）
    # ========================================================

    def gen_ffn_golden(self):
        """生成FFN golden结果，按卡写出"""
        params_out_dtype = torch.bfloat16
        tag = self.tag

        # W4量化: float4_e2m1
        w_dtype = self.weight_dtype  # "float4_e2m1"
        if self.fast_random_weight:
            if w_dtype != "float4_e2m1":
                raise ValueError(
                    "fast_random_weight currently supports only float4_e2m1, "
                    f"got {w_dtype}"
                )
            print(
                "[INFO] fast W4 weight generation enabled: direct E2M1 "
                f"storage codes, unit scale, seed={self.random_seed}"
            )
            w1_npu, weight_scale1_npu, weight1_dequant = \
                fast_mxfp4_random_data_prepare(
                    self.moe_expert_num,
                    self.h,
                    self.n,
                    self.transpose_weight,
                    self.random_seed,
                    weight_index=1,
                    generate_dequant=not self.skip_golden,
                )
            w2_npu, weight_scale2_npu, weight2_dequant = \
                fast_mxfp4_random_data_prepare(
                    self.moe_expert_num,
                    self.n // 2,
                    self.h,
                    self.transpose_weight,
                    self.random_seed,
                    weight_index=2,
                    generate_dequant=not self.skip_golden,
                )
        else:
            print("[INFO] legacy FP32 -> MXFP4 weight quantization enabled")
            # Legacy path: generate full FP32 weights, then quantize per expert.
            weight1_fp32 = regenerate_quant_data(
                (self.moe_expert_num, self.h, self.n), params_out_dtype
            )
            weight2_fp32 = regenerate_quant_data(
                (self.moe_expert_num, self.n // 2, self.h), params_out_dtype
            )
            w1_npu, weight_scale1_npu, weight1_dequant = \
                qbmm_mxfp4_quant_data_prepare(
                    weight1_fp32, w_dtype, self.transpose_weight,
                    generate_dequant=not self.skip_golden)
            w2_npu, weight_scale2_npu, weight2_dequant = \
                qbmm_mxfp4_quant_data_prepare(
                    weight2_fp32, w_dtype, self.transpose_weight,
                    generate_dequant=not self.skip_golden)

        # 按卡写出W4量化后的weight数据 (参考gen_data.py的写回方式)
        # npu weight (uint8 packed) → zN格式转换(16x16 tiling)后写回
        # npu scale (float8_e8m0) → reshape+permute(block配对)后写回
        for c in range(self.ep_world_size):
            experts_on_card = self.card_experts(c)
            # weight: zN格式转换 (参考gen_data.py convert_nd_to_zN_int8)
            w1_npu_card = convert_nd_to_zN_int8(w1_npu[experts_on_card])
            w2_npu_card = convert_nd_to_zN_int8(w2_npu[experts_on_card])
            # scale: reshape + permute (参考gen_data.py block配对写回)
            ws1_npu_card = transform_mxfp4_scale_for_npu(weight_scale1_npu[experts_on_card], self.transpose_weight)
            ws2_npu_card = transform_mxfp4_scale_for_npu(weight_scale2_npu[experts_on_card], self.transpose_weight)

            toolss_pro.write_tensor_to_bin(w1_npu_card, f"rank{c}_weight1_w4_npu_{tag}")
            toolss_pro.write_tensor_to_bin(w2_npu_card, f"rank{c}_weight2_w4_npu_{tag}")
            toolss_pro.write_tensor_to_bin(ws1_npu_card, f"rank{c}_weight1_w4_npu_scale_{tag}")
            toolss_pro.write_tensor_to_bin(ws2_npu_card, f"rank{c}_weight2_w4_npu_scale_{tag}")

        if self.skip_golden:
            return None

        weight1_list = list(weight1_dequant)
        weight2_list = list(weight2_dequant)

        x_list = self.a_reordered_list  # dispatch量化后的激活数据
        x_scale_list = self.a_scale_list

        # ---- GMM1: Matmul(x_quant, weight1_w4) ----
        print("--------------- Start: GMM1 (Matmul_1) -------------")
        # GMM1输入 = dispatch量化输出，已通过dispatch_quant_a8写出，不再重复写出

        output_gmm1 = qbmm_mx_golden_calculate(x_list, weight1_list, x_scale_list)
        # Kernel的GMM1 FixPipe以BF16写出，SwiGLU从BF16 GM读取。
        output_gmm1 = [value.to(torch.bfloat16).to(torch.float32) for value in output_gmm1]
        output_gmm1 = golden_output_process(output_gmm1, 2, 0)

        # ---- SwiGLU ----
        print("--------------- Start: SwiGLU -------------")
        swiglu_out = swiglu(output_gmm1[0])

        # 转回fp32进行量化
        swiglu_out_bf16 = swiglu_out.to(torch.bfloat16)
        swiglu_out_fp32 = swiglu_out_bf16.float()

        # ---- SwiGLU后A8量化 ----
        print("--------------- Start: SwiGLU A8 Quantize -------------")
        quant_output, quant_scale_output = mx_quantize(
            swiglu_out_fp32.numpy(), mx_ele_dtype=self.swiglu_quant_dtype, axis=-1)

        # 按卡写出swiglu量化数据
        for c in range(self.ep_world_size):
            experts_on_card = self.card_experts(c)
            split_sizes = [x_list[g].shape[0] for g in experts_on_card]
            split_indices_swiglu = np.cumsum([0] + [t.shape[0] for t in self.a_reordered_list])
            # 按卡拆分
            global_split_sizes = [t.shape[0] for t in self.a_reordered_list]
            card_swiglu_quant = [quant_output[split_indices_swiglu[g]:split_indices_swiglu[g + 1]]
                                  for g in experts_on_card if global_split_sizes[g] > 0]
            if card_swiglu_quant:
                card_swiglu_quant_all = np.concatenate(card_swiglu_quant, axis=0)
                toolss_pro.write_tensor_to_bin(card_swiglu_quant_all, f"rank{c}_swiglu_quant_a8_{tag}")

        # 处理perTokenScale格式
        x_transpose_flag = 0
        perTokenScale_input = np.concatenate([quant_scale_output], axis=-2 if x_transpose_flag == 1 else -1)
        # perTokenScale_input = perTokenScale_input.reshape(perTokenScale_input.shape[0], perTokenScale_input.shape[1] * 2)
        perTokenScale_input = transpose_last_two_dim(perTokenScale_input, x_transpose_flag)

        # 按卡写出perTokenScale
        for c in range(self.ep_world_size):
            experts_on_card = self.card_experts(c)
            split_sizes_all = [t.shape[0] for t in self.a_reordered_list]
            split_indices_all = np.cumsum([0] + split_sizes_all)
            card_pts = [perTokenScale_input[split_indices_all[g]:split_indices_all[g + 1]]
                         for g in experts_on_card if split_sizes_all[g] > 0]
            if card_pts:
                card_pts_all = np.concatenate(card_pts, axis=0)
                toolss_pro.write_tensor_to_bin(card_pts_all, f"rank{c}_swiglu_per_token_scale_{tag}")

        # 拆分给各expert
        split_sizes = [t.shape[0] for t in self.a_reordered_list]
        split_indices = np.cumsum([0] + split_sizes)
        quant_output_split = [quant_output[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]
        quant_scale_split = [perTokenScale_input[split_indices[i]:split_indices[i + 1]] for i in range(len(split_indices) - 1)]

        # GMM2输入 = SwiGLU量化输出，已通过swiglu_quant_a8写出，不再重复写出

        # ---- GMM2: Matmul(swiglu_quant_a8, weight2_w4) ----
        print("--------------- Start: GMM2 (Matmul_2) -------------")
        output_gmm2 = qbmm_mx_golden_calculate(quant_output_split, weight2_list, quant_scale_split)
        # Kernel的GMM2 FixPipe以BF16写出，Combine从BF16 GM读取。
        output_gmm2 = [value.to(torch.bfloat16).to(torch.float32) for value in output_gmm2]

        # 按卡写出GMM2输出
        for c in range(self.ep_world_size):
            experts_on_card = self.card_experts(c)
            card_gmm2_out = [output_gmm2[g] for g in experts_on_card if len(self.token_indices_list[g]) > 0]
            if card_gmm2_out:
                card_gmm2_all = torch.cat(card_gmm2_out, dim=0)
                toolss_pro.write_tensor_to_bin(card_gmm2_all.to(torch.bfloat16), f"rank{c}_gmm2_output_bf16_{tag}")

        # 拆分weight给各ep_rank
        weight1_list = np.array_split(np.stack(weight1_list, axis=0), self.ep_world_size)
        weight2_list = np.array_split(np.stack(weight2_list, axis=0), self.ep_world_size)
        weight1_scale_list = np.array_split(weight_scale1_npu, self.ep_world_size)
        weight2_scale_list = np.array_split(weight_scale2_npu, self.ep_world_size)

        return output_gmm2, weight1_list, weight2_list, weight1_scale_list, weight2_scale_list

    # ========================================================
    # Step 4: Combine Golden生成（按卡写出）
    # ========================================================

    def gen_combine_golden(self, expert_scales, expert_ids):
        """生成combine golden结果（包含unpermute），按卡写出"""
        tag = self.tag
        c2_final = torch.zeros(self.global_bs, self.h, dtype=torch.float32)

        # Combine: 将每个expert的GMM2结果乘以对应的expert权重，写回到原始token位置
        # token_indices_list[g] 存储的是原始token在全局batch中的位置，这就是unpermute
        for g in range(self.moe_expert_num):
            if len(self.token_indices_list[g]) > 0:
                indices = self.token_indices_list[g]
                for i, idx in enumerate(indices):
                    expert_scale_cur = expert_scales[idx][torch.where(expert_ids[idx] == g)[0].item()]
                    c2_final[idx] += self.c2_fp32_list[g][i] * expert_scale_cur

        # 按卡写出combine最终结果
        c2_final_per_card = self.chunk_tensor(c2_final, self.ep_world_size)
        for c in range(self.ep_world_size):
            toolss_pro.write_tensor_to_bin(c2_final_per_card[c].to(torch.bfloat16), f"rank{c}_combine_output_bf16_{tag}")

        # 按卡写出combine通信后结果（unpermute回原始token顺序, m*topk行, 不含加权）
        topk = expert_ids.shape[1]
        combine_gmm2 = torch.zeros(self.global_bs * topk, self.h, dtype=torch.float32)
        for g in range(self.moe_expert_num):
            if len(self.token_indices_list[g]) > 0:
                indices = self.token_indices_list[g]
                for i, idx in enumerate(indices):
                    topk_idx = torch.where(expert_ids[idx] == g)[0].item()
                    combine_gmm2[idx * topk + topk_idx] = self.c2_fp32_list[g][i]
        combine_gmm2_per_card = self.chunk_tensor(combine_gmm2, self.ep_world_size)
        for c in range(self.ep_world_size):
            toolss_pro.write_tensor_to_bin(combine_gmm2_per_card[c].to(torch.bfloat16), f"rank{c}_combine_gmm2_output_bf16_{tag}")

        # 每卡原始token按全局专家顺序排列的GMM2输出（dispatch前视角）
        # 卡c的token(c*bs ~ (c+1)*bs-1)按expert 0~moe_expert_num-1顺序排列
        for c in range(self.ep_world_size):
            card_start = c * self.bs
            card_end = (c + 1) * self.bs
            card_gmm2_local = []
            for g in range(self.moe_expert_num):
                if len(self.token_indices_list[g]) > 0:
                    indices = self.token_indices_list[g]
                    mask = (indices >= card_start) & (indices < card_end)
                    local_positions = torch.where(mask)[0]
                    if len(local_positions) > 0:
                        card_gmm2_local.append(self.c2_fp32_list[g][local_positions])
            if card_gmm2_local:
                card_gmm2_local_all = torch.cat(card_gmm2_local, dim=0)
            else:
                card_gmm2_local_all = torch.empty(0, self.h, dtype=torch.float32)
            toolss_pro.write_tensor_to_bin(card_gmm2_local_all.to(torch.bfloat16), f"rank{c}_gmm2_output_local_by_expert_bf16_{tag}")

        return c2_final_per_card

    # ========================================================
    # Step 5: Dispatch Map生成 (每张卡的token分发信息)
    # ========================================================

    def gen_dispatch_map_txt(self):
        """生成每张卡的token分发映射信息，写入txt文件"""
        tag = self.tag
        txt_file_path = f"{toolss_pro.DATA_PATH}/dispatch_map_{tag}.txt"

        expert_ids_per_card = self.expert_ids  # list of [bs, k], len = ep_world_size

        with open(txt_file_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"W4A8 MegaMoE Dispatch Map\n")
            f.write(f"bs={self.bs}, h={self.h}, n={self.n}, k={self.k}\n")
            f.write(f"local_moe_expert_num={self.local_moe_expert_num}, ep_world_size={self.ep_world_size}\n")
            f.write(f"moe_expert_num={self.moe_expert_num}\n")
            f.write(f"expert_id -> card mapping: expert_id // local_moe_expert_num = dest_card\n")
            f.write("=" * 80 + "\n\n")

            # ---- 1. 每张卡发送到各目标卡的token总数 ----
            f.write("=== 每张卡发送到各目标卡的token总数 (send_counts) ===\n\n")

            send_counts = np.zeros((self.ep_world_size, self.ep_world_size), dtype=np.int64)

            for src_card in range(self.ep_world_size):
                card_expert_ids = expert_ids_per_card[src_card]
                for token_idx in range(card_expert_ids.shape[0]):
                    for topk_idx in range(card_expert_ids.shape[1]):
                        expert_id = card_expert_ids[token_idx, topk_idx].item()
                        dest_card = self.expert_to_card(expert_id)
                        send_counts[src_card, dest_card] += 1

            header = "src_card | " + " | ".join([f"->card{d}" for d in range(self.ep_world_size)]) + " | total"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for src_card in range(self.ep_world_size):
                row = f"  card{src_card}  | "
                for dest_card in range(self.ep_world_size):
                    row += f"  {send_counts[src_card, dest_card]:6d} | "
                row += f"  {send_counts[src_card].sum():6d}"
                f.write(row + "\n")
            f.write("\n")

            # ---- 2. 每张卡发送到各专家的token数 ----
            f.write("=== 每张卡发送到各专家的token数 ===\n\n")

            per_card_per_expert_counts = np.zeros((self.ep_world_size, self.moe_expert_num), dtype=np.int64)

            for src_card in range(self.ep_world_size):
                card_expert_ids = expert_ids_per_card[src_card]
                for token_idx in range(card_expert_ids.shape[0]):
                    for topk_idx in range(card_expert_ids.shape[1]):
                        expert_id = card_expert_ids[token_idx, topk_idx].item()
                        per_card_per_expert_counts[src_card, expert_id] += 1

            header = "src_card | "
            for g in range(self.moe_expert_num):
                header += f" E{g}(c{self.expert_to_card(g)}) | "
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for src_card in range(self.ep_world_size):
                row = f"  card{src_card}  | "
                for g in range(self.moe_expert_num):
                    row += f" {per_card_per_expert_counts[src_card, g]:4d}    | "
                f.write(row + "\n")
            f.write("\n")

            # ---- 3. 每个专家收到的来自各卡的token数 ----
            f.write("=== 每个专家收到的来自各卡的token数 ===\n\n")

            recv_counts_per_expert = np.zeros((self.moe_expert_num, self.ep_world_size), dtype=np.int64)
            for src_card in range(self.ep_world_size):
                recv_counts_per_expert[:, src_card] = per_card_per_expert_counts[src_card, :]

            header = "expert   | card归属 | " + " | ".join([f"from_card{s}" for s in range(self.ep_world_size)]) + " | total"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for g in range(self.moe_expert_num):
                row = f"  E{g:2d}    |  card{self.expert_to_card(g):2d}  | "
                for src_card in range(self.ep_world_size):
                    row += f"  {recv_counts_per_expert[g, src_card]:6d} | "
                row += f"  {recv_counts_per_expert[g].sum():6d}"
                f.write(row + "\n")
            f.write("\n")

            # ---- 4. 每张卡上本地专家收到的总token数 ----
            f.write("=== 每张卡上本地专家收到的总token数 ===\n\n")
            for c in range(self.ep_world_size):
                experts_on_card = self.card_experts(c)
                total_recv = sum((torch.cat(self.expert_ids, dim=0) == g).sum().item() for g in experts_on_card)
                f.write(f"  rank{c}: experts={experts_on_card}, total_tokens_received={total_recv}\n")
            f.write("\n")

            # ---- 5. 逐token分发详情 ----
            f.write("=== 逐token分发详情 ===\n\n")
            for c in range(self.ep_world_size):
                f.write(f"--- rank{c} ---\n")
                card_expert_ids = expert_ids_per_card[c]
                for token_idx in range(card_expert_ids.shape[0]):
                    topk_experts = card_expert_ids[token_idx].tolist()
                    dest_cards = [self.expert_to_card(e) for e in topk_experts]
                    f.write(f"  token{token_idx}: top-k={topk_experts} -> cards={dest_cards}\n")
                f.write("\n")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="generate runtime inputs and quantized weights without CPU golden",
    )
    args = parser.parse_args()

    csv_path = os.environ.get(
        "MEGA_MOE_W4A8_CSV_PATH",
        os.path.join(script_dir, "test_shapes.csv"),
    )
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        config = next(reader)   # 读取第一行数据

    bs = int(config["M"])
    h = int(config["K"])
    n = int(config["N"])
    k = int(config["TOPK"])
    local_moe_expert_num = int(config["expertPerRank"])
    ep_world_size = int(config["EP"])
    moe_expert_num = local_moe_expert_num * ep_world_size  # 总expert数
    global_bs = bs * ep_world_size  # 全局batch size

    gen = GenInputAndGolden_W4A8(
        bs=bs,
        h=h,
        n=n,
        k=k,
        global_bs=global_bs,
        is_x_random=True,
        out_dtype=torch.bfloat16,
        moe_expert_num=moe_expert_num,
        local_moe_expert_num=local_moe_expert_num,
        is_topk_balance=False,
        is_mask=False,
        is_unfixed_real_token=False,
        ep_world_size=ep_world_size,
        is_variable_bs=False,
        expert_scales_dtype=torch.bfloat16,
        quant_mode=QUANT_MODE,
        transpose_weight=1,
        comm_quant_mode=4,
        weight_dtype=WEIGHT_DTYPE,
        quant_dtype=ACTIVATION_DTYPE,
        swiglu_quant_dtype=ACTIVATION_DTYPE,
        skip_golden=args.skip_golden,
        fast_random_weight=(
            os.environ.get("MEGA_MOE_W4A8_FAST_RANDOM_WEIGHT", "1") == "1"
        ),
        random_seed=int(os.environ.get("MEGA_MOE_W4A8_RANDOM_SEED", "831")),
    )
