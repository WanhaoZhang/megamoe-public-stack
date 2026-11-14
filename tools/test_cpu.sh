set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
g++ -std=c++17 -O2 -fno-fast-math -shared -fPIC -I "$TASK/source/native-balanced/include" "$TASK/tools/policy_oracle.cpp" -o "$TASK/evidence/policy-oracle.so"
docker exec planner-executor-ep8-lab /usr/local/python3.12.13/bin/python3 "$TASK/tools/test_policy_oracle.py" --library "$TASK/evidence/policy-oracle.so" --out "$TASK/evidence/cpu-policy-oracle.json"
