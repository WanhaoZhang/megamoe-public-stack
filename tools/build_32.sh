set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
OLD=/home/z00956592/megaflux/moonep-ep8-3layer-20261113/source/native-balanced32
TARGET="$TASK/source/native-balanced32"
if [ ! -d "$TARGET" ]; then cp -a --reflink=auto "$OLD" "$TARGET"; fi
python3 - "$TASK" <<'PY'
import pathlib,shutil,sys
r=pathlib.Path(sys.argv[1]);source=r/'source/native-balanced';target=r/'source/native-balanced32'
files=['CMakeLists.txt','include/mega_moe_unified_policy.h','kernels/mega_moe_bf16/mega_moe_unified_planner.cpp',
       'kernels/mega_moe_bf16/mega_moe_link_calibration.cpp','csrc/ops/mega_moe_bf16_replica.cpp']
files += ['kernels/mega_moe_bf16/mega_moe_step_time_planner.cpp']
files += ['ascend_deepep/'+x+'.py' for x in ['unified_planner','plan_replay','fsdp2_bridge','moonep_planner','megaflux_planners','planner_contract']]
for f in files:
 p=target/f;p.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source/f,p)
PY
export EP8_BUILD_CAP=32 EP8_BUILD_SUFFIX=32
bash "$TASK/tools/build_unified.sh"
