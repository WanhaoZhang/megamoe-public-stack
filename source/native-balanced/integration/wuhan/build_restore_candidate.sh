#!/usr/bin/env bash
# Build an isolated worker SO. Preserve the frozen core/planner binary lineage.
set -euo pipefail
ROOT=${1:?new committed composite snapshot}
DONOR=${2:?frozen composite runtime donor}
OUT=${3:?new build evidence directory}
NAME=${4:?unique variant name}
MODE=${5:?0 staged control, 1 expert candidate or 2 panel candidate}
[[ "$MODE" =~ ^[012]$ && "$NAME" =~ ^[a-z0-9-]+$ ]]
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/build.exit"' EXIT
set +u
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -u
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
export CPLUS_INCLUDE_PATH="$ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include:${CPLUS_INCLUDE_PATH:-}"
export SHMEM_HOME_PATH="$ROOT/ascend_deepep_hkq/3rdparty"
export MEGAFLUX_BACKWARD_RESTORE="$MODE"
export MEGAFLUX_WORKER_SHARED_PREFIX=0 MEGAFLUX_WORKER_EXTERNAL_REFRESH=0
export MEGAFLUX_OWNER_ROWS=16 MEGAFLUX_OWNER_COMPACT=0 MAX_JOBS=4
SHMEM_BUILD_ROOT=/tmp/megaflux-shmem-build
rm -rf "$SHMEM_BUILD_ROOT"
mkdir -p "$SHMEM_BUILD_ROOT"
cp -a --reflink=auto "$ROOT/ascend_deepep_hkq/3rdparty/shmem" "$SHMEM_BUILD_ROOT/shmem"
mkdir -p "$SHMEM_BUILD_ROOT/shmem/lib"
cp "$DONOR/ascend_deepep_hkq/ascend_deepep/lib/libshmem.so" "$SHMEM_BUILD_ROOT/shmem/lib/"
export SHMEM_HOME_PATH="$SHMEM_BUILD_ROOT"
python - "$ROOT" "$DONOR" "$OUT" <<'PY'
import hashlib,json,pathlib,shutil,sys
root,donor,out=map(pathlib.Path,sys.argv[1:])
n,d=[p/'ascend_deepep_hkq' for p in (root,donor)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checked={}
for prefix in ('csrc','include','3rdparty'):
    for p in (n/prefix).rglob('*'):
        if p.is_file():
            rel=p.relative_to(n)
            assert (d/rel).is_file() and sha(p)==sha(d/rel),str(rel)
            checked[str(rel)]=sha(p)
records={}
for p in [*(d/'ascend_deepep/lib').glob('*.so'),*(d/'ascend_deepep').glob('_C*.so')]:
    rel=p.relative_to(d);target=n/rel
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    records[str(rel)]=dict(sha256=sha(target),donor=str(p))
(out/'borrowed-control-runtime.json').write_text(json.dumps(dict(checked_binding_dependencies=checked,
    artifacts=records,note='Core/planner SOs retain donor source/build identity; worker is rebuilt below.'),indent=2))
PY
bash "$ROOT/ascend_deepep_hkq/scripts/build_megaflux_worker_pipeline.sh" "$NAME" 0 0 0
cp "$ROOT/ascend_deepep_hkq/worker-variants/$NAME/build.json" "$OUT/worker-build.json"
