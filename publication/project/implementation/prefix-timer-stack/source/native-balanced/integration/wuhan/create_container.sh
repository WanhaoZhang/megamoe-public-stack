#!/usr/bin/env bash
set -euo pipefail
PROJECT=${1:?absolute host project directory}
NAME=${2:-moonflux-balanced-8p-z00956592}
IMAGE=2754601232edf86a4b7dc6e2aca87cf86648e14491e173458ebbf3d5cbd6d2f8
test "$(id -u)" = 0
test -d /usr/local/Ascend/cann-9.2.0
docker image inspect "$IMAGE" > /dev/null
if docker container inspect "$NAME" > /dev/null 2>&1; then
    echo "Container already exists; inspect it explicitly rather than replacing it" >&2
    exit 1
fi
mkdir -p "$PROJECT/evidence/container"
docker run -d --name "$NAME" --privileged --network host --ipc host --shm-size 128g \
    -v "$PROJECT:/project" \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/cann-9.2.0:/usr/local/Ascend/cann-9.2.0:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    --entrypoint /bin/bash "$IMAGE" -lc 'exec tail -f /dev/null' \
    > "$PROJECT/evidence/container/id.txt"
docker inspect "$NAME" > "$PROJECT/evidence/container/inspect.json"
