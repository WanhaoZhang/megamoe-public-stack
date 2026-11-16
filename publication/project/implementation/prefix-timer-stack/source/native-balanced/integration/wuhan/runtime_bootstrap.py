"""Canonical SHMEM import order, then execute a training/test Python entry.

This wrapper changes no backend math and does not initialize SHMEM or an NPU.
Use in every control/candidate process so author Hybrid and S9 share the same
runtime library image. A second image is a hard deployment error.
"""
import importlib
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: runtime_bootstrap.py ENTRY.py [args...]')
    native_only = sys.argv[1] == '--native-only'
    if native_only:
        del sys.argv[1]
        if len(sys.argv) < 2:
            raise SystemExit('usage: runtime_bootstrap.py --native-only ENTRY.py [args...]')
    import torch
    import torch_npu
    import shmem
    # Resolve the lazy pybind loader before either native extension can open
    # its own absolute-path copy. No UID call/runtime init is performed here.
    _ = shmem.aclshmem_get_unique_id
    if native_only:
        # The standalone native operator needs no external MoonEP Python code.
        # Promote the already selected Python SHMEM image, never a package copy.
        paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if line.split()[-1].endswith('/libshmem.so')})
        if len(paths) != 1:
            raise RuntimeError(f'expected one selected Python SHMEM image: {paths}')
        directory = Path(paths[0]).parent
        mode = getattr(os,'RTLD_NOW',0) | getattr(ctypes,'RTLD_GLOBAL',0)
        for name in ('libshmem_utils.so','aclshmem_bootstrap_config_store.so','libshmem.so','aclshmem_bootstrap_uid.so'):
            path = directory/name
            if path.exists():ctypes.CDLL(str(path),mode=mode)
    else:
        importlib.import_module('npu_patch')
    importlib.import_module('ascend_deepep')
    mappings = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if line.split()[-1].endswith('/libshmem.so')})
    if len(mappings) != 1:
        raise RuntimeError(f'expected one canonical SHMEM image: {mappings}')
    print('WUHAN_RUNTIME', json.dumps(dict(shmem=mappings, torch=torch.__version__,
          torch_npu=torch_npu.__version__)), flush=True)
    entry = Path(sys.argv[1]).resolve()
    sys.argv = [str(entry), *sys.argv[2:]]
    sys.path.insert(0, str(entry.parent))
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
