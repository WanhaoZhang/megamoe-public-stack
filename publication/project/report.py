"""Current report entrypoint; early qualification reducer is preserved separately."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name('report_current.py')),run_name='__main__')
