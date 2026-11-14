#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?absolute isolated source root}
OUT=${2:?absolute evidence directory}
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/build.exit"' EXIT
# Explicit official build; do not inherit the image's CANN 9.1 environment.
set +u
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -u
# SHMEM's upstream helper sources the *parent* toolkit environment. In this
# image that parent selects 9.1. Use a private symlink view with no parent
# set_env.sh, leaving the image and host toolkit links untouched.
mkdir -p "$ROOT/.toolchain"
if [[ ! -L "$ROOT/.toolchain/cann" ]]; then
    ln -s /usr/local/Ascend/cann-9.2.0 "$ROOT/.toolchain/cann"
fi
export ASCEND_HOME_PATH="$ROOT/.toolchain/cann"
export PATH="$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"
export CPLUS_INCLUDE_PATH="/usr/local/Ascend/cann-9.2.0/python/site-packages/pypto/lib/framework/3rd/include:${CPLUS_INCLUDE_PATH:-}"
test -f "${CPLUS_INCLUDE_PATH%%:*}/nlohmann/json.hpp"
test "$(readlink -f "$ASCEND_HOME_PATH")" = /usr/local/Ascend/cann-9.2.0
command -v bisheng > "$OUT/compiler-path.txt"
export MAX_JOBS=8
export ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"
export ASCEND_MOONEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"
export SHMEM_HOME_PATH="$ROOT/ascend_deepep_hkq/3rdparty/shmem/install"
export ASCEND_PROCESS_LOG_PATH="$OUT/plog"
python -m pip freeze > "$OUT/pip-freeze.txt"
date -u > "$OUT/start.utc"
cd "$ROOT/ascend_deepep_hkq"
if [[ "${CONTROL_RESUME_MOONEP:-0}" == 1 ]]; then
    test -f ascend_deepep/lib/libascend_deepep_kernels.so
    test -f ascend_deepep/lib/libascend_deepep_mega_moe_bf16.so
    test -f "$SHMEM_HOME_PATH/set_env.sh"
    sha256sum ascend_deepep/lib/*.so* ascend_deepep/_C*.so > "$OUT/resumed-native.sha256"
else
    MEGA_MOE_FORCE_SHMEM_REBUILD=1 bash scripts/build.sh -soc_type Ascend950
    bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
fi
bash "$(dirname "$(realpath "$0")")/install_python_shmem.sh" "$ROOT" "$OUT"
cd "$ROOT/MoonEP"
bash npu_patch/scripts/build_a5.sh /usr/local/Ascend/cann-9.2.0/set_env.sh
python -m pip install -e "$ROOT/FSDPTurbo" --no-deps --no-build-isolation
export PYTHONPATH="$ROOT/MindSpeed-LLM:$ROOT/FSDPTurbo:$ROOT/MoonEP:${PYTHONPATH:-}"
python - "$ROOT" "$OUT" <<'PY'
import hashlib, importlib, json, os, pathlib, sys
r, out = map(pathlib.Path, sys.argv[1:])
modules = {}
for name in ('torch', 'torch_npu', 'transformers', 'npu_patch', 'ascend_deepep', 'fsdp_turbo'):
    mod = importlib.import_module(name)
    modules[name] = {'file': mod.__file__, 'version': str(getattr(mod, '__version__', ''))}
files = sorted(set(r.glob('ascend_deepep_hkq/ascend_deepep/lib/*.so*')) |
               set(r.glob('ascend_deepep_hkq/ascend_deepep/_C*.so')) |
               set(r.glob('MoonEP/npu_patch/lib/*.so*')) |
               set(r.glob('MoonEP/npu_patch/_C*.so')))
mapped = sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
                 if line.split()[-1].endswith('/libshmem.so')})
assert len(mapped) == 1, mapped
record = dict(modules=modules, shmem_mappings=mapped, native={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
              cann=os.environ['ASCEND_HOME_PATH'])
(out/'environment.json').write_text(json.dumps(record, indent=2))
print('CONTROL_NATIVE_BUILT', json.dumps(record, indent=2), flush=True)
PY
date -u > "$OUT/end.utc"
