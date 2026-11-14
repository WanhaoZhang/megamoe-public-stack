#!/usr/bin/env bash
# A fresh container on a compatible Ascend host. Never replaces another one.
set -euo pipefail
task_root=/home/z00956592/megaflux/planner-executor-ep8-lab
container_name=${1:-planner-executor-ep8-replay}
image_id=sha256:1fc454e2360ef25b7a529d26ca792a98200f12327d6c2ecf8a62eea384d8ca9e
test "$(uname -m)" = aarch64
test -f "$task_root/dependencies/RUNTIME-CLOSURE.json"
test -f "$task_root/dependencies/runtime/libstdc++.so.6.0.35"
test -d "$task_root/dependencies/sdk/shmem/lib"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container already exists; no replacement or stop performed: $container_name" >&2
  exit 1
fi
docker image inspect "$image_id" >/dev/null
# CANN9.2/toolkit and the driver remain host requirements at the original
# recorded paths. Nested binds replace old-project SDK/C++ dependencies with
# the byte-verified copies preserved inside this experiment.
docker run -d --name "$container_name" --label "codex.task=$container_name" \
  --privileged --network host --ipc host \
  --mount type=bind,src=/home,dst=/home \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver \
  --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi \
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info \
  --mount "type=bind,src=$task_root/dependencies/sdk,dst=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk,readonly" \
  --mount "type=bind,src=$task_root/dependencies/runtime/libstdc++.so.6.0.35,dst=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35,readonly" \
  --entrypoint /bin/bash "$image_id" -lc 'sleep infinity'
docker inspect --format '{{.Id}} {{.Image}} {{.State.Status}}' "$container_name"
echo 'Container created; no NPU training process started.'
