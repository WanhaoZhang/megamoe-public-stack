import numpy as np
import functools
import random
import time
import multiprocessing as mp
import gc
import psutil
import datetime
import copy
import os
import shutil
import torch
import struct
import json

# 尝试导入MX自定义dtype
try:
    from ml_dtypes import float8_e4m3fn, float8_e5m2, bfloat16
except ImportError:
    float8_e4m3fn = None
    float8_e5m2 = None
    bfloat16 = None
    print("WARNING: ml_dtypes not installed, FP8/BF16 bin read may fail")

try:
    from en_dtypes import float4_e2m1, float4_e1m2, float8_e8m0
except ImportError:
    float4_e2m1 = None
    float4_e1m2 = None
    float8_e8m0 = None
    print("WARNING: en_dtypes not installed, FP4/E8M0 bin read may fail")


DATA_PATH = os.environ.get(
    "MEGA_MOE_W4A8_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)
META_PATH = os.path.join(DATA_PATH, "meta")

print(f'Use DATA_PATH = {DATA_PATH}')
print(f'Use META_PATH = {META_PATH}')


def init_data_path(force_clean=None):
    """初始化数据目录。

    force_clean:
        True  -> 清空旧数据（原行为）
        False -> 保留旧数据，仅确保目录存在（用于复用已生成产物）
        None  -> 由环境变量 MEGA_MOE_W4A8_FORCE_CLEAN 决定（默认 0=保留）
    """
    if force_clean is None:
        force_clean = os.environ.get("MEGA_MOE_W4A8_FORCE_CLEAN", "0") == "1"
    if force_clean:
        shutil.rmtree(DATA_PATH, ignore_errors=True)
        print(f"[init_data_path] Data path cleaned: {DATA_PATH}")
    else:
        print(f"[init_data_path] Data path reused (not cleaned): {DATA_PATH}")
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(META_PATH, exist_ok=True)


def getDataType(type_str):
    type_dict = {
        'fp32': np.float32,
        'fp16': np.float16,
        'int8': np.int8,
        'uint8': np.uint8,
        'int16': np.int16,
        'uint16': np.uint16,
        'int32': np.int32,
        'uint32': np.uint32,
        'int64': np.int64,
        'uint64': np.uint64,
        'bool': np.bool_,
        'fp64': np.float64,
        'complex64': np.complex64,
        'complex128': np.complex128,
        'float8_e4m3fn': float8_e4m3fn,
        'float8_e5m2': float8_e5m2,
        'bfloat16': bfloat16,
        'float4_e2m1': float4_e2m1,
        'float4_e1m2': float4_e1m2,
        'float8_e8m0': float8_e8m0,
    }
    return type_dict.get(type_str)


def getStrDataType(type_id):
    type_dict = {
        0: "fp32", 1: "fp16", 2: "int8", 3: "int32", 4: "uint8",
        6: "int16", 7: "uint16", 8: "uint32", 9: "int64", 10: "uint64",
        11: "double", 12: "bool", 26: "undefined"
    }
    return type_dict.get(int(float(type_id)), "fp32")


def _dtype_to_str(dtype):
    """将dtype转换为可读字符串，用于文件名和元信息"""
    if isinstance(dtype, torch.dtype):
        return str(dtype)
    elif isinstance(dtype, np.dtype):
        return str(dtype)
    elif dtype is None:
        return "None"
    else:
        return str(dtype)


def write_tensor_to_bin(tensor, prefix):
    """
    将tensor写入bin文件，并保存shape和dtype元信息到json文件。
    支持torch.Tensor和numpy.ndarray，包括MX自定义dtype。
    所有中间数据都会被完整保存。
    """
    if tensor is None:
        print(f"[write_tensor_to_bin] {prefix} is None, skip writing.")
        return

    dtype_str = _dtype_to_str(tensor.dtype)
    bin_file_path = f"{DATA_PATH}/{prefix}.bin"
    meta_file_path = f"{META_PATH}/{prefix}.json"

    # 写入bin数据
    if isinstance(tensor, torch.Tensor):
        type_dict = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.int8: torch.int8,
            torch.float32: torch.int32,
            torch.int32: torch.int32,
            torch.int64: torch.int64,
            torch.float8_e4m3fn: torch.uint8,
            torch.float8_e5m2: torch.uint8,
            torch.bool: torch.bool,
        }
        if tensor.dtype in type_dict:
            raw_data = tensor.view(type_dict[tensor.dtype]).numpy().tobytes()
        else:
            # 不支持的torch dtype，转为numpy再写
            raw_data = tensor.numpy().tobytes()
    elif isinstance(tensor, np.ndarray):
        raw_data = tensor.tobytes()
    else:
        raise TypeError(f"write_tensor_to_bin: unsupported type {type(tensor)}")

    with open(bin_file_path, 'wb') as f:
        f.write(raw_data)

    # 写入元信息 (shape, dtype, itemsize)
    meta_info = {
        "prefix": prefix,
        "dtype": dtype_str,
        "shape": list(tensor.shape) if hasattr(tensor, 'shape') else [],
        "itemsize": tensor.dtype.itemsize if hasattr(tensor.dtype, 'itemsize') else 0,
        "numel": tensor.numel() if isinstance(tensor, torch.Tensor) else tensor.size,
        "is_torch": isinstance(tensor, torch.Tensor),
    }
    # 记录torch dtype的view类型用于读回
    if isinstance(tensor, torch.Tensor) and tensor.dtype in {
        torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64,
        torch.int8, torch.float8_e4m3fn, torch.float8_e5m2, torch.bool
    }:
        if tensor.dtype in type_dict:
            meta_info["view_dtype"] = str(type_dict[tensor.dtype])

    with open(meta_file_path, 'w') as f:
        json.dump(meta_info, f, indent=2)

    print(f"[write_tensor_to_bin] {bin_file_path} | dtype={dtype_str} | shape={list(tensor.shape)} | size={len(raw_data)} bytes")


def read_tensor_from_bin(prefix, target_dtype=None):
    """
    从bin文件读回tensor，自动读取元信息以恢复shape和dtype。
    如果指定target_dtype，则按该dtype读回并reshape。
    """
    bin_file_path = f"{DATA_PATH}/{prefix}.bin"
    meta_file_path = f"{META_PATH}/{prefix}.json"

    if not os.path.exists(bin_file_path):
        raise FileNotFoundError(f"Bin file not found: {bin_file_path}")

    # 读取元信息
    if os.path.exists(meta_file_path):
        with open(meta_file_path, 'r') as f:
            meta_info = json.load(f)
        shape = tuple(meta_info["shape"])
        original_dtype_str = meta_info["dtype"]
        is_torch = meta_info.get("is_torch", False)
    else:
        # 没有元信息文件，需要用户提供dtype和shape
        if target_dtype is None:
            raise ValueError(f"No meta file found for {prefix}, target_dtype must be provided")
        shape = None
        original_dtype_str = None
        is_torch = False

    # 确定读取的dtype
    read_dtype = target_dtype if target_dtype is not None else original_dtype_str

    if is_torch and read_dtype is not None:
        # torch tensor读取
        torch_dtype_map = {
            'torch.float32': np.float32,
            'torch.float16': np.float16,
            'torch.bfloat16': np.uint16,  # bf16用uint16存储
            'torch.int32': np.int32,
            'torch.int64': np.int64,
            'torch.int8': np.int8,
            'torch.float8_e4m3fn': np.uint8,
            'torch.float8_e5m2': np.uint8,
            'torch.bool': np.bool_,
        }
        # 检查view_dtype
        view_dtype_str = meta_info.get("view_dtype", None) if meta_info else None
        if view_dtype_str:
            np_view_dtype = eval(f"np.{view_dtype_str.replace('torch.', '')}")
        elif read_dtype in torch_dtype_map:
            np_view_dtype = torch_dtype_map[read_dtype]
        else:
            np_view_dtype = np.uint8  # fallback

        raw = np.fromfile(bin_file_path, dtype=np_view_dtype)
        if shape is not None and len(shape) > 0:
            raw = raw.reshape(shape)

        # 转换回原始torch dtype
        torch_dtype_str_map = {
            'torch.float32': torch.float32,
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.int32': torch.int32,
            'torch.int64': torch.int64,
            'torch.int8': torch.int8,
            'torch.float8_e4m3fn': torch.float8_e4m3fn,
            'torch.float8_e5m2': torch.float8_e5m2,
            'torch.bool': torch.bool,
        }
        target_torch_dtype = torch_dtype_str_map.get(original_dtype_str, None) if original_dtype_str else None
        if target_torch_dtype:
            data = torch.from_numpy(raw).view(target_torch_dtype)
        else:
            data = torch.from_numpy(raw)
    else:
        # numpy ndarray读取
        np_dtype_map = {
            'float32': np.float32, 'float16': np.float16, 'int8': np.int8,
            'uint8': np.uint8, 'int32': np.int32, 'uint32': np.uint32,
            'int64': np.int64, 'uint64': np.uint64, 'bool': np.bool_,
            'float8_e4m3fn': float8_e4m3fn, 'float8_e5m2': float8_e5m2,
            'bfloat16': bfloat16, 'float4_e2m1': float4_e2m1,
            'float4_e1m2': float4_e1m2, 'float8_e8m0': float8_e8m0,
        }

        np_read_dtype = np_dtype_map.get(read_dtype, np.uint8) if isinstance(read_dtype, str) else read_dtype
        data = np.fromfile(bin_file_path, dtype=np_read_dtype)
        if shape is not None and len(shape) > 0:
            data = data.reshape(shape)

    print(f"[read_tensor_from_bin] {bin_file_path} | dtype={_dtype_to_str(data.dtype)} | shape={list(data.shape)}")
    return data


def write_shape_info(prefix, shape, dtype_str):
    """单独写入shape信息（用于某些需要额外记录的场景）"""
    meta_file_path = f"{META_PATH}/{prefix}_shape.json"
    info = {"shape": list(shape), "dtype": dtype_str}
    with open(meta_file_path, 'w') as f:
        json.dump(info, f, indent=2)


def gen_data(data_size, min_val, max_val, dtype):
    return np.random.uniform(min_val, max_val, size=data_size).astype(dtype)


def getData(data_path, shape, dtype):
    data_pool = np.fromfile(data_path, dtype)
    data_len = functools.reduce(lambda x, y: x * y, shape)
    return data_pool[0:data_len].reshape(shape)


def calRelativediff(x, y, diffThd):
    if float(max(abs(x), abs(y))) < float((1.0 / (1 << 14)) / diffThd):
        result = abs(float(x) - float(y)) / (float((1.0 / (1 << 14)) / diffThd) + 10e-10)
    else:
        result = abs(float(x) - float(y)) / (float(max(abs(x), abs(y))) + 10e-10)
    return result


def calRelativediffNumpy(dataCheck, dataExpect, diffThd):
    a = np.abs(np.subtract(dataCheck, dataExpect))
    b1 = np.maximum(np.abs(dataCheck), np.abs(dataExpect))
    b2 = float((1.0 / (1 << 14)) / diffThd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diffThd, a, a / b)
    return result


def dataCompare(dataCheck, dataExpect, diffThd=0.01, pctThd=0.05):
    npu_shape = dataCheck.shape
    expect_shape = dataExpect.shape
    dataCheck = dataCheck.flatten()
    dataExpect = dataExpect.flatten()
    start = 0
    end = dataCheck.size - 1

    # 分段计算相对误差
    segments = 8
    segment_size = dataCheck.size // segments
    rdiff_parts = []
    for s in range(segments):
        s_start = s * segment_size
        s_end = (s + 1) * segment_size if s < segments - 1 else dataCheck.size
        rdiff_parts.append(calRelativediffNumpy(
            dataCheck[s_start:s_end].astype(np.float32),
            dataExpect[s_start:s_end].astype(np.float32),
            diffThd
        ))
    rdiff = np.concatenate(rdiff_parts)

    split_rdiff = rdiff[start:end + 1]
    splitCount = int(end - start + 1) if end != start else 1
    ltNum = sum(part[part < diffThd].size for part in rdiff_parts)
    ltNum = ltNum + dataExpect[np.isinf(dataExpect)].size + dataExpect[np.isnan(dataExpect)].size
    ltPct = float(ltNum) / float(splitCount) * 100.0

    pctThd = (1 - pctThd) * 100.0
    result = "PASS" if (ltPct >= pctThd) else "FAIL"
    if npu_shape != expect_shape:
        result = "FAIL"

    print(f'\n[Compare Result] DiffThd={diffThd:.3f} | PctThd={pctThd:.2f}% | PctRlt={ltPct:.6f}% | Result={result}')

    if result == "FAIL":
        # 打印部分误差较大的数据
        err_indices = np.where(rdiff > diffThd)[0]
        count = 0
        for idx in err_indices[:100]:
            j = idx + start
            print(f'  Err[{j}]: Expect={dataExpect[j]:.7f} Real={dataCheck[j]:.7f} Diff={abs(float(dataExpect[j]) - float(dataCheck[j])):.7f}')

    return result, ltPct


def list_all_bin_files():
    """列出所有保存的bin文件及其元信息"""
    if not os.path.exists(META_PATH):
        print("[list_all_bin_files] No meta directory found.")
        return

    for filename in sorted(os.listdir(META_PATH)):
        if filename.endswith('.json'):
            meta_file_path = os.path.join(META_PATH, filename)
            with open(meta_file_path, 'r') as f:
                meta_info = json.load(f)
            prefix = meta_info.get("prefix", filename.replace('.json', ''))
            dtype = meta_info.get("dtype", "unknown")
            shape = meta_info.get("shape", [])
            numel = meta_info.get("numel", 0)
            bin_path = f"{DATA_PATH}/{prefix}.bin"
            bin_size = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
            print(f"  {prefix}: dtype={dtype}, shape={shape}, numel={numel}, filesize={bin_size} bytes")


def compare_bin_files(prefix1, prefix2, diffThd=0.01, pctThd=0.05):
    """比较两个bin文件的数据精度"""
    data1 = read_tensor_from_bin(prefix1)
    data2 = read_tensor_from_bin(prefix2)

    # 转为float32进行比较
    if isinstance(data1, torch.Tensor):
        data1 = data1.float().numpy()
    else:
        data1 = data1.astype(np.float32)
    if isinstance(data2, torch.Tensor):
        data2 = data2.float().numpy()
    else:
        data2 = data2.astype(np.float32)

    return dataCompare(data1, data2, diffThd, pctThd)
