"""Recognize a manager-only monitor for bounded observation, never acceptance."""
import re


def transient_monitor_only(rc, stdout, stderr, idle):
    if rc != 0 or idle != list(range(8)):
        return False
    if stdout.strip() and not all(p.isdigit() for p in stdout.split()):
        return False
    paths = set(re.findall(r'/dev/[a-zA-Z0-9_]+', stderr))
    if paths != {'/dev/davinci_manager'}:
        return False
    rows = []
    for line in stderr.splitlines():
        line = re.sub(r'/dev/[a-zA-Z0-9_]+:', '', line).strip()
        if not line or line.split() == ['USER', 'PID', 'ACCESS', 'COMMAND']:
            continue
        # fuser sends PIDs to stdout on some hosts; on others they appear here.
        if not re.fullmatch(r'\S+\s+(?:\d+\s+)?[.A-Za-z]+\s+npu-smi', line):
            return False
        rows.append(line)
    return bool(rows)
