"""Restore two verified EOL-only checkout changes in this newly created clone."""
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent/'stack'
names=('tools/write_planner_report.py','tools/write_report.py')
subprocess.run(['git','-C',str(root),'diff','--ignore-space-at-eol','--exit-code','--',*names],check=True)
for name in names:
    (root/name).write_bytes(subprocess.check_output(['git','-C',str(root),'show','HEAD:'+name]))
