"""Finite CPU-only build of the standalone ready-frontier AscendC plugin."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
with (root/'build-temporal-scheduler.log').open('wb') as output:
    result=subprocess.run(['ssh','pod6-1-root','docker','exec','-i',
        'temporal-replica-frontier-20261115','bash'],
        input=(root/'build_temporal_scheduler.sh').read_bytes(),stdout=output,stderr=subprocess.STDOUT)
print(result.returncode,flush=True)
raise SystemExit(result.returncode)
