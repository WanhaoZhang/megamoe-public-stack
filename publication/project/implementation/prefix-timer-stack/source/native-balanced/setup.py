import os
import platform
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import torch_npu

ROOT = Path(__file__).resolve().parent
ASCEND_HOME_PATH = Path(os.environ["ASCEND_HOME_PATH"])
SHMEM_HOME_PATH = Path(
    os.environ.get("SHMEM_HOME_PATH", ROOT / "3rdparty" / "shmem" / "install")
)
KERNEL_LIB_DIR = Path(
    os.environ.get("ASCEND_DEEPEP_KERNEL_LIB_DIR", ROOT / "ascend_deepep" / "lib")
)
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent

if not (KERNEL_LIB_DIR / "libascend_deepep_kernels.so").exists():
    raise RuntimeError(
        f"Kernel library not found in {KERNEL_LIB_DIR}. Run scripts/build.sh first."
    )

def host_arch_dir():
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return "aarch64-linux"
    if machine in {"x86_64", "amd64"}:
        return "x86_64-linux"
    return f"{machine}-linux"


def existing_paths(paths):
    return [path for path in paths if path.exists()]


ASCEND_ARCH_DIR = ASCEND_HOME_PATH / host_arch_dir()

include_dirs = [
    ROOT,
    ROOT / "include",
    ROOT / "csrc",
    ROOT / "kernels" / "mega_moe_bf16" / "include",
    ASCEND_HOME_PATH / "include",
    SHMEM_HOME_PATH / "shmem" / "include",
    SHMEM_HOME_PATH / "shmem" / "src",
    SHMEM_HOME_PATH / "memfabric_hybrid" / "include" / "smem" / "host",
    TORCH_NPU_ROOT / "include",
    TORCH_NPU_ROOT.parent,
] + existing_paths(
    [
        ASCEND_ARCH_DIR / "include",
        ASCEND_ARCH_DIR / "pkg_inc" / "runtime",
        ASCEND_ARCH_DIR / "runtime" / "include",
        ASCEND_ARCH_DIR / "include" / "runtime",
    ]
)

library_dirs = [
    KERNEL_LIB_DIR,
    ASCEND_HOME_PATH / "lib64",
    SHMEM_HOME_PATH / "shmem" / "lib",
    SHMEM_HOME_PATH / "memfabric_hybrid" / "lib",
    TORCH_NPU_ROOT / "lib",
] + existing_paths(
    [
        ASCEND_ARCH_DIR / "lib64",
    ]
)

rpath_dirs = [
    KERNEL_LIB_DIR,
    ASCEND_HOME_PATH / "lib64",
    SHMEM_HOME_PATH / "shmem" / "lib",
    SHMEM_HOME_PATH / "memfabric_hybrid" / "lib",
] + existing_paths(
    [
        ASCEND_ARCH_DIR / "lib64",
    ]
)

setup(
    name="ascend-deepep",
    version="0.1.0",
    description="AscendC expert-parallel communication runtime based on CANN SHMEM",
    packages=find_packages(),
    package_data={"ascend_deepep": ["lib/*.so", "lib/*.so.*"]},
    include_package_data=True,
    ext_modules=[
        CppExtension(
            name="ascend_deepep._C",
            sources=[
                "csrc/pybind.cpp",
                "csrc/torch_classes.cpp",
                "csrc/launcher.cpp",
                "csrc/ops/allgather_matmul.cpp",
                "csrc/ops/combine.cpp",
                "csrc/ops/dispatch.cpp",
                "csrc/ops/mega_moe.cpp",
                "csrc/ops/mega_moe_bf16.cpp",
                "csrc/ops/mega_moe_bf16_replica.cpp",
                "csrc/ops/mega_moe_bf16_mgroup.cpp",
                "csrc/ops/mega_moe_bf16_udma.cpp",
                "csrc/ops/mega_moe_w4a8.cpp",
                "csrc/shmem_instance.cpp",
            ],
            include_dirs=[str(path) for path in include_dirs],
            library_dirs=[str(path) for path in library_dirs],
            libraries=[
                "ascend_deepep_kernels",
                "torch_npu",
                "ascendcl",
                "runtime",
                "tiling_api",
                "platform",
                "shmem",
                "dl",
            ],
            extra_compile_args=["-O2", "-std=c++17", "-DMEGAMOE_MAX_LOCAL_EXPERTS=" + os.environ.get("ASCEND_DEEPEP_MAX_LOCAL_EXPERTS", "32")],
            extra_link_args=["-Wl,-rpath,$ORIGIN/lib"]
            + [f"-Wl,-rpath,{path}" for path in rpath_dirs],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
