"""Build a separate TimerV2 worker from the admitted compute configuration."""
from pathlib import Path
import hashlib
import json
import subprocess

root=Path(__file__).resolve().parent
native=root/'source/native-balanced'
baseline=native/'build-worker-ep8-cap64-timeroff/CMakeCache.txt'
cache={}
for line in baseline.read_text().splitlines():
    if not line or line.startswith(('#','//')) or '=' not in line or ':' not in line:continue
    key,rest=line.split(':',1);kind,value=rest.split('=',1)
    if kind not in ('INTERNAL','STATIC'):cache[key]=(kind,value)
selected={key:value for key,value in cache.items() if key.startswith(
    ('ASCEND_DEEPEP_','MEGAFLUX_','MEGAMOE_')) or key in
    ('SHMEM_HOME_PATH','ASCEND_HOME_PATH','CMAKE_BUILD_TYPE','CMAKE_CXX_FLAGS','CMAKE_CXX_FLAGS_RELEASE')}
selected['ASCEND_DEEPEP_ENABLE_TIMER']=('BOOL','ON')
selected['MEGAFLUX_CALIBRATION_TIMER']=('BOOL','ON')
selected['MEGAFLUX_SUMMARY_TIMER']=('BOOL','ON')
selected['ASCEND_DEEPEP_SHMEM_SOURCE_DIR']=('PATH',str(native/'3rdparty/shmem'))
build=native/'build-temporal-cap64-timer'
command=['cmake','-S',str(native),'-B',str(build)]+[f'-D{key}:{kind}={value}' for key,(kind,value) in sorted(selected.items())]
out=root/'evidence/timer-worker-build';out.mkdir(exist_ok=True,parents=True)
(out/'configure.json').write_text(json.dumps(dict(command=command,baseline=str(baseline),
    timing_only_overrides=['ASCEND_DEEPEP_ENABLE_TIMER','MEGAFLUX_CALIBRATION_TIMER'],
    compute_flags_copied=True),indent=2))
subprocess.run(command,check=True)
subprocess.run(['cmake','--build',str(build),'--target','ascend_deepep_megaflux_worker_bf16','-j2'],check=True)
source=build/'lib/libascend_deepep_megaflux_worker_bf16.so'
variant=native/'worker-variants/temporal-ep8-cap64-timer';variant.mkdir(exist_ok=True)
binary=source.read_bytes()
entries={}
for name in ('libascend_deepep_megaflux_worker_bf16.so','libascend_deepep_megaflux_early_bf16.so'):
    target=variant/name
    if target.exists() and target.read_bytes()!=binary:raise RuntimeError('Refusing to overwrite another diagnostic build')
    target.write_bytes(binary)
    entries[str(target.relative_to(root))]=hashlib.sha256(binary).hexdigest()
manifest=root/'configs/temporal-worker-timer-freeze.json'
manifest.write_text(json.dumps(dict(scope='diagnostic_only; not performance binary',binaries=entries),indent=2))
(out/'binary.json').write_text(manifest.read_text())
print(json.dumps(dict(diagnostic_binary_sha256=hashlib.sha256(binary).hexdigest(),manifest=str(manifest))))
