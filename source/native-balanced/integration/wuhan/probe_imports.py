"""Host-only library/import isolation probe. Never initializes a device/runtime."""
import ctypes
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def mappings():
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if 'shmem' in line and line.split()[-1].startswith('/')})


if len(sys.argv) > 1 and sys.argv[1] == '--child':
    for item in sys.argv[2:]:
        if item.startswith('/'):
            ctypes.CDLL(item, mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL)
        else:
            module = importlib.import_module(item)
            print('IMPORTED', item, getattr(module, '__file__', None), flush=True)
        print('MAPPINGS', json.dumps(mappings()), flush=True)
    print('NORMAL_TEARDOWN', flush=True)
else:
    cases = [('torch', 'torch_npu', 'ascend_deepep'),
             ('torch', 'torch_npu', 'npu_patch'),
             ('torch', 'torch_npu', 'ascend_deepep', 'npu_patch'),
             ('torch', 'torch_npu', 'npu_patch', 'ascend_deepep'),
             ('torch', 'torch_npu', 'transformers', 'npu_patch', 'ascend_deepep', 'fsdp_turbo')]
    results = []
    for case in cases:
        p = subprocess.run([sys.executable, __file__, '--child', *case], capture_output=True, text=True)
        record = dict(imports=case, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
        results.append(record)
        print(json.dumps(record), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(results, indent=2))
