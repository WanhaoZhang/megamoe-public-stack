"""This foreground controller continues the authorized task after the matrix."""
import json
import subprocess
import time
from controller import TASK

while True:
    path=TASK/'evidence/campaign-primary-existing.json'
    try:rows=json.loads(path.read_text()) if path.exists() else []
    except json.JSONDecodeError:
        time.sleep(1);continue  # original writer may be closing its snapshot
    if len(rows)==90:break
    time.sleep(15)
print('BASELINE_MATRIX_READY_FOR_AUDIT',flush=True)
subprocess.run(['python3',str(TASK/'tools/recover_initialization.py')],check=True)
subprocess.run(['python3',str(TASK/'tools/analyze_matrix.py'),'--campaign','primary-existing','--campaign','primary-existing-recovery'],check=True)
subprocess.run(['python3',str(TASK/'tools/after_baselines.py')],check=True)
