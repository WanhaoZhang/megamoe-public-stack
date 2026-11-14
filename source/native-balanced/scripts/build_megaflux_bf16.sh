#!/usr/bin/env bash
# Build from the checked-out gitlinks; source the desired CANN environment first.
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
: "${ASCEND_HOME_PATH:?Source the selected CANN set_env.sh before building}"
: "${MAX_JOBS:=8}"
export MAX_JOBS
for dependency in catccos catlass shmem; do
    expected_revision="$(git -c "safe.directory=$project_dir" rev-parse "HEAD:3rdparty/$dependency")"
    actual_revision="$(git -c "safe.directory=$project_dir/3rdparty/$dependency" -C "3rdparty/$dependency" rev-parse HEAD)"
    test "$expected_revision" = "$actual_revision" || {
        echo "Dependency revision differs from gitlink: $dependency" >&2; exit 1;
    }
done
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
cmake -S . -B build_replica \
    -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 \
    -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=OFF \
    -DSHMEM_HOME_PATH="$project_dir/3rdparty/shmem/install"
cmake --build build_replica --target ascend_deepep_replica --parallel "$MAX_JOBS"
cp build_replica/lib/libascend_deepep_replica.so ascend_deepep/lib/
python setup.py build_ext --inplace
sha256sum ascend_deepep/lib/*.so ascend_deepep/_C*.so
