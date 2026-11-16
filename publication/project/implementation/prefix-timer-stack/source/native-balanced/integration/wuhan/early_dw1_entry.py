"""Validate the selected experimental SO before executing a device entry."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def verify(native, expected, loader=ctypes.CDLL):
    native = Path(native).resolve()
    record = json.loads((native/'build.json').read_text())
    if expected not in (0, 1) or record.get('backward_replicated_dw1_early') != expected:
        raise ValueError('replicated DW1 requested/manifest mismatch')
    path = native/'libascend_deepep_megaflux_early_bf16.so'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != record['native_sha256']:
        raise ValueError('worker SO SHA mismatch')
    lib = loader(str(path))
    if int(lib.megaflux_backward_replicated_dw1_early()) != expected:
        raise ValueError('replicated DW1 loaded getter mismatch')
    for name, value in (('megaflux_worker_pipeline_flags',0),
                        ('megaflux_backward_restore_mode',2),
                        ('megaflux_restore_operand_ready',1),
                        ('megaflux_restore_delay_cycles',0),
                        ('megaflux_worker_owner_rows',16),
                        ('megaflux_worker_owner_compact',0),
                        ('megaflux_dispatch_ready_largest',0)):
        if int(getattr(lib,name)()) != value:
            raise ValueError('incompatible loaded worker: '+name)
    return dict(early_dw1=expected,sha256=sha,path=str(path))


if __name__ == '__main__':
    if len(sys.argv)<2:
        raise SystemExit('early_dw1_entry.py ENTRY [ARGS...]')
    print('EARLY_DW1_IDENTITY', json.dumps(verify(os.environ['WORKER_NATIVE'],
          int(os.environ['EXPECTED_EARLY_DW1']))), flush=True)
    entry=Path(sys.argv[1]).resolve();sys.argv=[str(entry),*sys.argv[2:]]
    runpy.run_path(str(entry),run_name='__main__')
