"""Refresh only local analysis/report products as guarded screen receipts arrive."""
from pathlib import Path
import json
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
last = None
while True:
    manifest = root / 'screening-results.json'
    content = manifest.read_bytes()
    jobs = json.loads(content)['jobs']
    if content != last:
        for script in ('analyze_screening.py', 'report_current.py'):
            result = subprocess.run([sys.executable, '-X', 'utf8', str(root / script)],
                                    capture_output=True, text=True, encoding='utf-8')
            if result.returncode:
                print(result.stderr, flush=True)
                raise RuntimeError('Local report validation failed; no training process was signalled')
        print(json.dumps(dict(completed=sum(job['status'] == 'completed' for job in jobs),
                              running=[job['tag'] for job in jobs if job['status'] == 'running'],
                              failed=[job['tag'] for job in jobs if job['status'] == 'failed'])), flush=True)
        last = content
    if any(job['status'] == 'failed' for job in jobs) or all(job['status'] == 'completed' for job in jobs):
        break
    time.sleep(30)
