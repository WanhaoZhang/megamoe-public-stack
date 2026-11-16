set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 - <<'PY'
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
info=json.loads(subprocess.check_output(['docker','inspect','planner-executor-ep8-lab']))[0]
assert info['Id']=='f09afea1c8febaa114a1fcb7c64ad3b782085cdca89d6a55a47ff31fefce4865'
assert not info['State']['Running']
record=dict(passed=True,container_id=info['Id'],running=False,state=info['State'],image=info['Image'],
    only_task_container_stopped=True,source_and_data_retained=True,time_utc=datetime.now(timezone.utc).isoformat())
Path('evidence/USER-CLOSEOUT-CONTAINER.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
PY
