"""Cancel only this study's not-yet-admitted W1/W2 diagnostic jobs."""
from pathlib import Path
import os
import signal

root = Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack')
tags = ('ga4-w1-aligned','ga4-w2-aligned')
for process in Path('/proc').iterdir():
    if not process.name.isdigit():
        continue
    try:
        command = (process/'cmdline').read_bytes().split(b'\0')
        arguments = [part.decode() for part in command if part]
        if 'tools/controller.py' not in arguments or (process/'cwd').resolve() != root:
            continue
        for tag in tags:
            if tag in arguments and not (root/'evidence'/tag).exists():
                os.kill(int(process.name),signal.SIGTERM)
                print('CANCELLED_BEFORE_ADMISSION',tag,process.name,flush=True)
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        continue
