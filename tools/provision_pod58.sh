#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
base=/home/z00956592/megaflux/planner-executor-ep8-lab
name=single-mb-fbw-planner-study
test ! -e "$task"
mkdir -p "$task"/{source,evidence,artifacts,reports,analysis}
for component in source tools configs routes dependencies; do
  cp -a --reflink=auto "$base/$component" "$task/$component.stage"
  if [ "$component" = source ]; then
    # Destination is this new task's empty directory only.
    cp -a "$task/source.stage/." "$task/source/"
  else
    mv "$task/$component.stage" "$task/$component"
  fi
done
python3 - "$task" "$base" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);old=sys.argv[2]
for path in (p/'tools').glob('*'):
    if path.suffix not in ('.py','.sh'):continue
    text=path.read_text()
    text=text.replace(old,str(p)).replace("CONTAINER = 'planner-executor-ep8-lab'","CONTAINER = 'single-mb-fbw-planner-study'")
    path.write_text(text)
PY
docker run -d --name "$name" --label codex.task="$name" --privileged --network host --ipc host \
  -v /home:/home \
  -v "$task/dependencies/sdk:/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk:ro" \
  -v "$task/dependencies/runtime/libstdc++.so.6.0.35:/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35:ro" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --entrypoint /bin/bash planner-executor-ep8-runtime:20261114 -lc 'sleep infinity'
docker inspect "$name" --format '{{.Id}} {{.Image}} {{.State.Status}}'
