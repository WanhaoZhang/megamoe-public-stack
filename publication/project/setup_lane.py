"""Prepare isolated deployment helpers without touching either predecessor."""
from pathlib import Path
root=Path(__file__).resolve().parent
for name in ('implementation','diagnostics','evidence'):(root/name).mkdir(exist_ok=True)
source=(root.parent/'remote.py').read_text()
(root/'remote.py').write_text(source.replace('step-lifetime-study-20261115','step-lifetime-raw-control-20261116'),newline='\n')
source=(root.parent/'causal-diagnostics-20261116/implementation/build-causal-timer.sh').read_text()
for before,after in (
 ('step-lifetime-causal-20261116','step-lifetime-raw-control-20261116'),
 ('ep8-cap64-lifetime-timer-causal-v1','ep8-cap64-lifetime-raw-control-v2'),
 ('MEGAFLUX_CALIBRATION_TIMER=ON','MEGAFLUX_CALIBRATION_TIMER=OFF'),
 ('v2 0 0 1','v2 0 0 0'),
 ("for name in ('temporal-worker-timer-freeze.json','lifetime-timer-runtime-freeze.json'):","for name in ('lifetime-runtime-freeze.json',):"),
 ('causal-diagnostic-only ABI1 original-Timer schema9','private ABI2 TimerOFF raw-output schedule control'),
 ('CAUSAL_WORKER','RAW_CONTROL_WORKER')):source=source.replace(before,after)
(root/'implementation/build-raw-worker.sh').write_text(source,newline='\n')
