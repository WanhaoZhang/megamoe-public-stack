#!/usr/bin/env bash
# Build only the Python wrapper against the already frozen common SHMEM tree.
set -euo pipefail
ROOT=${1:?control composite source root}
OUT=${2:?evidence directory}
SRC="$ROOT/ascend_deepep_hkq/3rdparty/shmem"
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/python-shmem.exit"' EXIT
sha256sum "$SRC"/install/shmem/lib/*.so* > "$OUT/shmem-before-python.sha256"
cmake -S "$SRC" -B "$SRC/build" -DBUILD_PYTHON=ON
cmake --build "$SRC/build" --target _pyshmem --parallel 8
python - "$SRC" <<'PY'
from pathlib import Path
import shutil
import sys
root = Path(sys.argv[1])
extensions = list((root/'build/lib').glob('_pyshmem*.so'))
assert len(extensions) == 1, extensions
shutil.copy2(extensions[0], root/'src/python/shmem'/extensions[0].name)
backend = root/'install/shmem/backends/950'
backend.mkdir(parents=True, exist_ok=True)
for lib in (root/'install/shmem/lib').glob('*.so*'):
    shutil.copy2(lib, backend/lib.name)
PY
_SHMEM_PREBUILT=1 python -m pip install "$SRC" --no-deps --no-build-isolation
# The Python wrapper build must not replace the frozen installed runtime.
sha256sum -c "$OUT/shmem-before-python.sha256"
python -m pip show cann-shmem > "$OUT/python-shmem-package.txt"
