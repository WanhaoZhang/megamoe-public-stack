#!/usr/bin/env bash
# One-controller multi-node launcher for the ascend_deepep_hkq test entries.
#
# Adapted from the ops repository's st/batch_launch_ep128.sh orchestration:
# SSH staging over the first N hosts of HOST_ORDER, where the first IP is
# always node rank 0 and provides MASTER_IP. Build once on a shared CODE_DIR
# or in parallel on per-node checkouts, verify the installation on every
# node, then launch one command per node with per-node stage logs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'HELP'
Usage: bash scripts/batch_launch.sh [--plan|--prepare-only|--dry-run] [--skip-build] [N ['command template']]
N: 1-<host count> machines, NPROC cards each; defaults to all hosts in HOST_ORDER.

Required environment:
  HOST_ORDER  comma-separated node IPv4 list; first IP is rank 0 and MASTER_IP.
  CODE_DIR    absolute repository path on every node. Either one shared mount
              (build once on rank 0; every node imports it via PYTHONPATH,
              which also shadows stale site-packages installs) or one
              checkout per node (parallel builds); auto-detected.

Optional environment:
  REMOTE_USER=root SSH_AUTH=key|password SSH_PORT=22
  CONDA_BASE CONDA_ENV        activate this Conda env on every node (optional)
  ASCEND_ENV                  CANN set_env.sh sourced on every node
  NPROC=8                     cards per node
  MASTER_PORT=29518 SHMEM_PORT=8899  torchrun / SHMEM symmetric window ports
  SEQ=4096 WARMUP=3 ITERS=10  default run_speed.sh positional arguments
  BUILD_LAYOUT=auto|all|rank0 auto: shared CODE_DIR -> rank 0, else every node
  SYNC_GIT=1                  git fetch origin + checkout the controller
                              commit on the build targets before building
  ALLOW_SHA_MISMATCH=1        warn instead of failing on a commit mismatch
  SOC_TYPE=Ascend950          passed to scripts/build.sh
  CTRL_SHA                    expected/target commit (default: controller HEAD;
                              used by SYNC_GIT checkout and verify)
  BUILD_EXTRA_ARGS            extra args for scripts/build_mega_moe_bf16.sh
                              (e.g. --enable-timer)
  GLOO_IFNAME_DEFAULT         GLOO_SOCKET_IFNAME exported on every node; the
                              SHMEM bootstrap picks its network interface with
                              it (machines with differently-named NICs fail
                              aclshmemx_init_attr otherwise)
  GLOO_IFNAME_MAP             per-node overrides, comma-separated ip=iface,
                              e.g. 141.62.17.70=eth2; wins over the default
  FORWARD_ENV_PATTERN         ERE of environment variable names forwarded to
                              every node (default also covers BENCH_*/BIGOP_*/
                              MEGAMOE_*/MEGA_MOE_*/SKIP_REF/LOCAL_DEVICES)
  PRELOAD_LIBSTDCXX=1         LD_PRELOAD the Conda libstdc++ (needs CONDA_ENV;
                              fixes CANN/Conda libstdc++ clashes)
  LOG_DIR                     controller log directory (default ./launch_logs_<RUN_ID>)

Modes:
  --plan           print per-node commands and stages; no SSH, prompts or writes.
  --prepare-only   run preflight/sync/build/verify, then stop before the tests.
  --dry-run        alias for --prepare-only.
  --skip-build     reuse existing builds; no compiler is invoked.

Stages: preflight(rank0) -> preflight(nodes, shared-FS probe) ->
        [sync(SYNC_GIT=1)] -> build(rank0 on shared, else all) or reuse ->
        verify(all) -> run(all).

Templates support {NODE_RANK} {MASTER_ADDR} {NUM_NODES} {NPROC} {MASTER_PORT}
                  {SHMEM_PORT} {SEQ} {WARMUP} {ITERS} {CODE_DIR} {RUN_ID}.
NODE_RANK/NNODES/NPROC/MASTER_IP/MASTER_PORT/SHMEM_PORT/HCCL_IF_IP and
PYTHONPATH=$CODE_DIR are also exported on every node before the command runs.

Examples:
  bash scripts/batch_launch.sh --plan 4
  SYNC_GIT=1 bash scripts/batch_launch.sh 8
  bash scripts/batch_launch.sh 2 'NODE_RANK={NODE_RANK} bash test/megamoe_bf16_test/run_backward_dual_node.sh {MASTER_ADDR}'
HELP
}

MODE=run
BUILD_STAGE=build
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
while (( $# > 0 )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --plan)
            [[ $MODE != prepare ]] || { echo '[ERROR] Choose only one of --plan and preparation mode.' >&2; exit 1; }
            MODE=plan; shift ;;
        --prepare-only|--dry-run)
            [[ $MODE != plan ]] || { echo '[ERROR] Choose only one of --plan and preparation mode.' >&2; exit 1; }
            MODE=prepare; shift ;;
        --skip-build) BUILD_STAGE=reuse; shift ;;
        *) break ;;
    esac
done
NUM_NODES=''
if (( $# > 0 )); then
    [[ $1 =~ ^[1-9][0-9]*$ ]] || { echo '[ERROR] N must be a positive integer.' >&2; exit 1; }
    NUM_NODES="$1"
    shift
fi
(( $# <= 1 )) || { usage >&2; exit 1; }

HOST_ORDER="${HOST_ORDER:-}"
CODE_DIR="${CODE_DIR:-${REMOTE_CODE:-}}"
[[ -n $HOST_ORDER ]] || { echo '[ERROR] Set HOST_ORDER to a comma-separated node IPv4 list; the first IP is rank 0 / MASTER_IP.' >&2; exit 1; }
[[ -n $CODE_DIR ]] || { echo '[ERROR] Set CODE_DIR to the repository path present on every node.' >&2; exit 1; }
[[ $CODE_DIR == /* ]] || { echo '[ERROR] CODE_DIR must be an absolute path.' >&2; exit 1; }

REMOTE_USER="${REMOTE_USER:-root}"
SSH_AUTH="${SSH_AUTH:-key}"
SSH_PORT="${SSH_PORT:-22}"
CONDA_BASE="${CONDA_BASE:-}"
CONDA_ENV="${CONDA_ENV:-}"
ASCEND_ENV="${ASCEND_ENV:-}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29518}"
SHMEM_PORT="${SHMEM_PORT:-8899}"
SEQ="${SEQ:-4096}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
BUILD_LAYOUT="${BUILD_LAYOUT:-auto}"
SYNC_GIT="${SYNC_GIT:-0}"
ALLOW_SHA_MISMATCH="${ALLOW_SHA_MISMATCH:-0}"
SOC_TYPE="${SOC_TYPE:-Ascend950}"
BUILD_EXTRA_ARGS="${BUILD_EXTRA_ARGS:-}"
GLOO_IFNAME_DEFAULT="${GLOO_IFNAME_DEFAULT:-}"
GLOO_IFNAME_MAP="${GLOO_IFNAME_MAP:-}"
FORWARD_ENV_PATTERN="${FORWARD_ENV_PATTERN:-^(BENCH_|BIGOP_|MEGAMOE_|MEGA_MOE_|SKIP_REF|LOCAL_DEVICES|TORCH_DEVICE_BACKEND|ASCEND_RT_)}"
PRELOAD_LIBSTDCXX="${PRELOAD_LIBSTDCXX:-0}"

for port in "$MASTER_PORT" "$SHMEM_PORT" "$SSH_PORT"; do
    [[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo '[ERROR] Invalid port.' >&2; exit 1; }
done
[[ $MASTER_PORT != "$SHMEM_PORT" ]] || { echo '[ERROR] torchrun and SHMEM must use different ports.' >&2; exit 1; }
[[ $SSH_AUTH == key || $SSH_AUTH == password ]] || { echo '[ERROR] SSH_AUTH must be key or password.' >&2; exit 1; }
[[ $NPROC =~ ^[1-9][0-9]*$ ]] || { echo '[ERROR] NPROC must be a positive integer.' >&2; exit 1; }
[[ $BUILD_LAYOUT == auto || $BUILD_LAYOUT == all || $BUILD_LAYOUT == rank0 ]] || { echo '[ERROR] BUILD_LAYOUT must be auto, all or rank0.' >&2; exit 1; }
[[ $SYNC_GIT == 0 || $SYNC_GIT == 1 ]] || { echo '[ERROR] SYNC_GIT must be 0 or 1.' >&2; exit 1; }
[[ $ALLOW_SHA_MISMATCH == 0 || $ALLOW_SHA_MISMATCH == 1 ]] || { echo '[ERROR] ALLOW_SHA_MISMATCH must be 0 or 1.' >&2; exit 1; }
[[ $PRELOAD_LIBSTDCXX == 0 || $PRELOAD_LIBSTDCXX == 1 ]] || { echo '[ERROR] PRELOAD_LIBSTDCXX must be 0 or 1.' >&2; exit 1; }
[[ $WARMUP =~ ^[0-9]+$ && $ITERS =~ ^[1-9][0-9]*$ && $SEQ =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || {
    echo '[ERROR] Invalid SEQ, WARMUP or ITERS.' >&2; exit 1;
}
if [[ -n $CONDA_ENV ]]; then
    [[ -n $CONDA_BASE && $CONDA_BASE == /* ]] || { echo '[ERROR] CONDA_ENV needs an absolute CONDA_BASE.' >&2; exit 1; }
fi
if [[ -n $ASCEND_ENV ]]; then
    [[ $ASCEND_ENV == /* ]] || { echo '[ERROR] ASCEND_ENV must be an absolute path.' >&2; exit 1; }
fi

HOSTS=()
for host in ${HOST_ORDER//,/ }; do
    [[ $host =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || {
        echo "[ERROR] HOST_ORDER entry '$host' is not an IPv4 address." >&2; exit 1;
    }
    for octet in "${BASH_REMATCH[@]:1:4}"; do
        (( octet <= 255 )) || { echo "[ERROR] HOST_ORDER entry '$host' is not a valid IPv4 address." >&2; exit 1; }
    done
    for seen in ${HOSTS[@]+"${HOSTS[@]}"}; do
        [[ $seen != "$host" ]] || { echo "[ERROR] HOST_ORDER contains '$host' twice." >&2; exit 1; }
    done
    HOSTS+=("$host")
done
TOTAL_HOSTS=${#HOSTS[@]}
if [[ -z $NUM_NODES ]]; then NUM_NODES=$TOTAL_HOSTS; fi
(( NUM_NODES >= 1 && NUM_NODES <= TOTAL_HOSTS )) || {
    echo "[ERROR] N must be between 1 and $TOTAL_HOSTS (hosts in HOST_ORDER)." >&2; exit 1;
}
MASTER_ADDR="${HOSTS[0]}"

if [[ -n ${BENCH_EXPERTS:-} && $BENCH_EXPERTS =~ ^[1-9][0-9]*$ ]]; then
    (( BENCH_EXPERTS % (NPROC * NUM_NODES) == 0 )) || {
        echo "[ERROR] BENCH_EXPERTS=$BENCH_EXPERTS is not divisible by $((NPROC * NUM_NODES)) ranks (NPROC x N)." >&2; exit 1;
    }
elif [[ -z ${BENCH_EXPERTS:-} && $NUM_NODES -gt 2 ]]; then
    echo "[WARN] BENCH_EXPERTS is unset; run_speed.sh's default 16 experts will not divide $((NPROC * NUM_NODES)) ranks." >&2
fi

if [[ -n ${CTRL_SHA:-} ]]; then
    [[ $CTRL_SHA =~ ^[0-9a-f]{7,40}$ ]] || { echo "[ERROR] CTRL_SHA '$CTRL_SHA' is not a commit hash." >&2; exit 1; }
else
    CTRL_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)" || {
        echo '[ERROR] The controller checkout must be a git repository (or set CTRL_SHA).' >&2; exit 1;
    }
fi
if ! git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null 2>&1; then
    echo "[WARN] Controller checkout is in detached HEAD at $CTRL_SHA; run 'git switch <branch> && git pull' to follow branch updates." >&2
fi
if [[ $SYNC_GIT == 1 ]]; then
    if ! git -C "$REPO_ROOT" branch -r --contains "$CTRL_SHA" 2>/dev/null | grep -q .; then
        echo "[WARN] Commit $CTRL_SHA is not on any remote branch; with SYNC_GIT=1 the nodes cannot fetch it. Push first." >&2
    fi
fi

FORWARD_VARS=()
while IFS= read -r name; do FORWARD_VARS+=("$name"); done < <(compgen -A variable | grep -E "$FORWARD_ENV_PATTERN" || true)
PROXY_VARS=()
for variable in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; do
    if [[ -n ${!variable:-} ]]; then PROXY_VARS+=("$variable"); fi
done

RUN_ID="$(date '+%Y%m%d_%H%M%S')_$$_${RANDOM}"
LOG_DIR="${LOG_DIR:-./launch_logs_$RUN_ID}"
DEFAULT_COMMAND='NNODES={NUM_NODES} NODE_RANK={NODE_RANK} MASTER_IP={MASTER_ADDR} NPROC={NPROC} bash test/bigop_fused/run_speed.sh {SEQ} {WARMUP} {ITERS}'
COMMAND="${1:-$DEFAULT_COMMAND}"

quote() { printf '%q' "$1"; }
render_command() {
    local rank="$1" text="$COMMAND"
    text="${text//\{NODE_RANK\}/$rank}"
    text="${text//\{MASTER_ADDR\}/$MASTER_ADDR}"
    text="${text//\{NUM_NODES\}/$NUM_NODES}"
    text="${text//\{NPROC\}/$NPROC}"
    text="${text//\{MASTER_PORT\}/$MASTER_PORT}"
    text="${text//\{SHMEM_PORT\}/$SHMEM_PORT}"
    text="${text//\{SEQ\}/$SEQ}"
    text="${text//\{WARMUP\}/$WARMUP}"
    text="${text//\{ITERS\}/$ITERS}"
    text="${text//\{CODE_DIR\}/$(quote "$CODE_DIR")}"
    text="${text//\{RUN_ID\}/$RUN_ID}"
    printf '%s\n' "$text"
}

remote_script() {
    local rank="$1" stage="$2" variable job py_check ifname pair
    printf 'set -eo pipefail\n'
    if [[ -n $CONDA_ENV ]]; then
        printf 'source %s\nconda activate %s\n' \
            "$(quote "$CONDA_BASE/etc/profile.d/conda.sh")" "$(quote "$CONDA_ENV")"
        printf 'export LAUNCH_CONDA_PREFIX="${CONDA_PREFIX:?conda activate did not set CONDA_PREFIX}"\n'
        printf 'test -x "$LAUNCH_CONDA_PREFIX/bin/python"\n'
        printf 'export PYTHON_BIN="$LAUNCH_CONDA_PREFIX/bin/python"\n'
    else
        printf 'export PYTHON_BIN="$(command -v python3)"\n'
        printf 'test -n "$PYTHON_BIN"\n'
    fi
    if [[ -n $ASCEND_ENV ]]; then
        printf 'source %s\n' "$(quote "$ASCEND_ENV")"
    fi
    # CANN or shell startup scripts may prepend another Python to PATH. Pin
    # the interpreter chosen above and restore its priority for every stage.
    printf 'export PATH="$(dirname "$PYTHON_BIN"):$PATH"\nunset PYTHONHOME\nhash -r\n'
    if [[ $PRELOAD_LIBSTDCXX == 1 && -n $CONDA_ENV ]]; then
        printf 'export LD_PRELOAD="${LAUNCH_CONDA_PREFIX}/lib/libstdc++.so.6"\n'
    fi
    printf '[[ -n "${ASCEND_HOME_PATH:-}" ]] || { echo "[ERROR] ASCEND_HOME_PATH is empty; set ASCEND_ENV to a CANN set_env.sh" >&2; exit 1; }\n'
    if [[ -n $CONDA_ENV ]]; then
        printf '"$PYTHON_BIN" -c %s\n' "$(quote 'import os,sys; from pathlib import Path; print("[PYTHON_ENV] executable="+sys.executable+" prefix="+sys.prefix+" version="+sys.version.split()[0], flush=True); assert Path(sys.prefix).resolve()==Path(os.environ["LAUNCH_CONDA_PREFIX"]).resolve(), "Python prefix does not match activated Conda: "+os.environ["LAUNCH_CONDA_PREFIX"]')"
    else
        printf '"$PYTHON_BIN" -c %s\n' "$(quote 'import sys; print("[PYTHON_ENV] executable="+sys.executable+" prefix="+sys.prefix+" version="+sys.version.split()[0], flush=True)')"
    fi
    for variable in ${PROXY_VARS[@]+"${PROXY_VARS[@]}"}; do
        printf 'export %s=%s\n' "$variable" "$(quote "${!variable}")"
    done
    for variable in ${FORWARD_VARS[@]+"${FORWARD_VARS[@]}"}; do
        printf 'export %s=%s\n' "$variable" "$(quote "${!variable}")"
    done
    printf 'cd %s\n' "$(quote "$CODE_DIR")"
    # Prepend CODE_DIR so ascend_deepep resolves to this checkout even when a
    # node's env carries a stale editable install of another checkout.
    printf 'export PYTHONPATH=%s"${PYTHONPATH:+:$PYTHONPATH}"\n' "$(quote "$CODE_DIR")"
    # The SHMEM runtime resolves fabric/bootstrap dependencies by soname; the
    # ops launcher has always exported these paths on the same cluster.
    printf 'export LD_LIBRARY_PATH=%s"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n' \
        "$(quote "$CODE_DIR/ascend_deepep/lib:$CODE_DIR/3rdparty/shmem/install/shmem/lib:$CODE_DIR/3rdparty/shmem/install/memfabric_hybrid/lib")"
    printf 'export NODE_RANK=%s NNODES=%s NPROC=%s\n' "$rank" "$NUM_NODES" "$NPROC"
    printf 'export MASTER_IP=%s MASTER_PORT=%s SHMEM_PORT=%s HCCL_IF_IP=%s\n' \
        "$MASTER_ADDR" "$MASTER_PORT" "$SHMEM_PORT" "${HOSTS[$rank]}"
    ifname=''
    if [[ -n $GLOO_IFNAME_MAP ]]; then
        for pair in ${GLOO_IFNAME_MAP//,/ }; do
            if [[ $pair == "${HOSTS[$rank]}"=* ]]; then ifname="${pair#*=}"; fi
        done
    fi
    if [[ -z $ifname && -n $GLOO_IFNAME_DEFAULT ]]; then ifname=$GLOO_IFNAME_DEFAULT; fi
    if [[ -n $ifname ]]; then
        printf 'export GLOO_SOCKET_IFNAME=%s\n' "$(quote "$ifname")"
    fi
    printf 'test -d .git\ntest -f scripts/build.sh\ntest -f scripts/build_mega_moe_bf16.sh\ntest -f test/bigop_fused/run_speed.sh\n'
    printf 'printf "[ENV] node=%s host=%%s conda=%%s cann=%%s code=%%s\\n" "$HCCL_IF_IP" "${CONDA_DEFAULT_ENV:-none}" "${ASCEND_HOME_PATH:-none}" "$PWD"\n' "$rank"
    case "$stage" in
        preflight0)
            printf '"$PYTHON_BIN" -c %s\n' "$(quote 'import torch, torch_npu; print("[TORCH] torch="+torch.__version__+" torch_npu="+torch_npu.__version__)')"
            printf '"$PYTHON_BIN" -c %s %s\n' "$(quote 'import sys, torch, torch_npu; n=torch.npu.device_count(); print("[NPU] count="+str(n)); assert n >= int(sys.argv[1]), "not enough NPU devices"')" "$NPROC"
            printf 'echo %s > .batch_launch_probe\n' "$(quote "$RUN_ID")"
            printf 'printf "[PREFLIGHT] node=%s OK\\n"\n' "$rank"
            return
            ;;
        preflight)
            printf '"$PYTHON_BIN" -c %s\n' "$(quote 'import torch, torch_npu; print("[TORCH] torch="+torch.__version__+" torch_npu="+torch_npu.__version__)')"
            printf '"$PYTHON_BIN" -c %s %s\n' "$(quote 'import sys, torch, torch_npu; n=torch.npu.device_count(); print("[NPU] count="+str(n)); assert n >= int(sys.argv[1]), "not enough NPU devices"')" "$NPROC"
            printf 'if [[ -f .batch_launch_probe ]]; then printf "[FS] shared=1\\n"; else printf "[FS] shared=0\\n"; fi\n'
            printf 'printf "[PREFLIGHT] node=%s OK\\n"\n' "$rank"
            return
            ;;
        sync)
            job='git fetch origin'$'\n'
            job+='if [[ "$(git rev-parse HEAD)" != '"$(quote "$CTRL_SHA")"' ]]; then git checkout --detach '"$(quote "$CTRL_SHA")"'; fi'$'\n'
            job+='git submodule update --init --recursive'$'\n'
            job+='[[ "$(git rev-parse HEAD)" == '"$(quote "$CTRL_SHA")"' ]] || { echo "[ERROR] checkout of controller commit failed" >&2; exit 1; }'
            ;;
        build)
            # Checked here rather than in preflight: the sync stage (which
            # runs 'git submodule update --init') only happens after
            # preflight, so a fresh clone must not be rejected before sync
            # gets a chance to initialize 3rdparty/shmem.
            job='test -f 3rdparty/shmem/scripts/build.sh || { echo "[ERROR] 3rdparty/shmem submodule missing; run git submodule update --init --recursive or use SYNC_GIT=1" >&2; exit 1; }'$'\n'
            job+='bash scripts/build.sh -soc_type '"$(quote "$SOC_TYPE")"$'\n'
            job+='bash scripts/build_mega_moe_bf16.sh --python "$PYTHON_BIN" '"$BUILD_EXTRA_ARGS"
            ;;
        verify)
            # torch/torch_npu must load first: the _C extension links against
            # libtorch_npu.so, which only resolves once torch_npu imported it.
            py_check='import torch, torch_npu, os, sys; import ascend_deepep; from ascend_deepep import MegaMoeBF16Train, ShmemInstance; p=os.path.realpath(os.path.dirname(os.path.abspath(ascend_deepep.__file__))); root=os.path.realpath(sys.argv[1]); print("[INSTALL_PATH] "+p); sys.exit(0 if p==root or p.startswith(root+os.sep) else 1)'
            job=''
            if [[ $rank == 0 ]]; then
                job+='rm -f .batch_launch_probe'$'\n'
            fi
            job+='actual_sha="$(git rev-parse HEAD)"'$'\n'
            job+='if [[ "$actual_sha" != '"$(quote "$CTRL_SHA")"' ]]; then printf "[SHA] node=%s actual=%s expected=%s\n" "$NODE_RANK" "$actual_sha" '"$(quote "$CTRL_SHA")"';'$'\n'
            job+='  if [[ '"$(quote "$ALLOW_SHA_MISMATCH")"' != 1 ]]; then echo "[ERROR] node is on a different commit; set SYNC_GIT=1 or ALLOW_SHA_MISMATCH=1" >&2; exit 1; fi; fi'$'\n'
            job+='check_install() { "$PYTHON_BIN" -c '"$(quote "$py_check")"' '"$(quote "$CODE_DIR")"'; }'$'\n'
            job+='if ! check_install; then'$'\n'
            job+='  echo "[ERROR] ascend_deepep did not import from CODE_DIR with torch/torch_npu preloaded; see the traceback above." >&2'$'\n'
            job+='  echo "[ERROR] If the traceback shows another checkout, run on this node:" "$PYTHON_BIN -m pip uninstall -y ascend_deepep" >&2'$'\n'
            job+='  echo "[ERROR] Otherwise rebuild without --skip-build." >&2'$'\n'
            job+='  exit 1'$'\n'
            job+='fi'$'\n'
            job+='printf "[VERIFY] node=%s OK\n" "$NODE_RANK"'
            ;;
        run)
            job="$(render_command "$rank")"
            ;;
        *) return 1 ;;
    esac
    job='export PATH="$(dirname "$PYTHON_BIN"):$PATH"; hash -r'$'\n'"$job"
    # A separate job process group lets SSH HUP/TERM clean up this job's workers.
    cat <<'REMOTE'
set -m
child=''
cleanup() {
    trap - EXIT HUP INT TERM
    if [[ -n $child ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
REMOTE
    printf 'bash -eo pipefail -c %s &\nchild=$!\nlaunch_rc=0\nwait "$child" || launch_rc=$?\nexit "$launch_rc"\n' "$(quote "$job")"
}

printf '[CLUSTER] nodes=%s cards=%s master=%s commit=%s\n' "$NUM_NODES" "$((NPROC * NUM_NODES))" "$MASTER_ADDR" "$CTRL_SHA"
for rank in "${!HOSTS[@]}"; do
    (( rank < NUM_NODES )) || break
    printf '[PLAN] node=%s ranks=%s-%s host=%s\n' "$rank" "$((rank * NPROC))" "$((rank * NPROC + NPROC - 1))" "${HOSTS[$rank]}"
done
printf '[ENV] conda=%s cann=%s code=%s forward_vars=%s\n' \
    "${CONDA_ENV:-none}" "${ASCEND_ENV:-login-shell}" "$CODE_DIR" "${FORWARD_VARS[*]:-none}"
if [[ $BUILD_STAGE == reuse ]]; then
    printf '[BUILD] reuse existing artifacts; no compiler is invoked\n'
else
    printf '[BUILD] layout=%s (auto: shared CODE_DIR -> rank 0, per-node -> all)\n' "$BUILD_LAYOUT"
fi
printf '[STAGES] preflight(rank0) -> preflight(nodes, shared probe)'
if (( NUM_NODES > 1 )); then :; else printf ' (skipped: single node)'; fi
if [[ $SYNC_GIT == 1 ]]; then printf ' -> sync'; fi
printf ' -> %s -> verify(all)' "$BUILD_STAGE"
if [[ $MODE == prepare ]]; then
    printf ' -> stop (prepare only; no tests)\n'
else
    printf ' -> run(all)\n'
fi
for rank in "${!HOSTS[@]}"; do
    (( rank < NUM_NODES )) || break
    if [[ $MODE != prepare ]]; then render_command "$rank"; fi
done
if [[ $MODE == plan ]]; then exit 0; fi

command -v ssh >/dev/null
SSH=(ssh -n -p "$SSH_PORT" -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new)
if [[ $SSH_AUTH == password ]]; then
    command -v sshpass >/dev/null || { echo '[ERROR] Install sshpass or use SSH_AUTH=key.' >&2; exit 1; }
    if [[ -z ${SSHPASS:-} ]]; then read -rsp "SSH password for $REMOTE_USER: " SSHPASS; echo; fi
    export SSHPASS
    SSH=(sshpass -e "${SSH[@]}")
else
    SSH+=( -o BatchMode=yes )
fi
mkdir -p "$LOG_DIR"
PIDS=()
cleanup_controller() {
    local pid
    trap - INT TERM
    for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM "$pid" 2>/dev/null || true; done
    wait || true
    unset SSHPASS
}
trap 'cleanup_controller; exit 130' INT
trap 'cleanup_controller; exit 143' TERM

run_stage() {
    local stage="$1" rank index launch_rc line failed=0
    shift
    local ranks=("$@")
    if (( ${#ranks[@]} == 0 )); then
        ranks=()
        for ((rank = 0; rank < NUM_NODES; rank++)); do ranks+=("$rank"); done
    fi
    PIDS=()
    for rank in "${ranks[@]}"; do
        printf '[START] stage=%s node=%s host=%s log=%s\n' "$stage" "$rank" "${HOSTS[$rank]}" "$LOG_DIR/node_${rank}_${HOSTS[$rank]}_${stage}.log"
        "${SSH[@]}" "$REMOTE_USER@${HOSTS[$rank]}" \
            "bash -lc $(quote "$(remote_script "$rank" "$stage")")" \
            >"$LOG_DIR/node_${rank}_${HOSTS[$rank]}_${stage}.log" 2>&1 &
        PIDS+=("$!")
    done
    for index in "${!PIDS[@]}"; do
        rank="${ranks[$index]}"
        launch_rc=0
        wait "${PIDS[$index]}" || launch_rc=$?
        printf '[%s] node=%s host=%s exit=%s log=%s\n' "$stage" "$rank" "${HOSTS[$rank]}" "$launch_rc" "$LOG_DIR/node_${rank}_${HOSTS[$rank]}_${stage}.log"
        if (( launch_rc != 0 )); then
            failed=1
            tail -n 30 "$LOG_DIR/node_${rank}_${HOSTS[$rank]}_${stage}.log"
        elif [[ $stage == preflight || $stage == preflight0 || $stage == verify ]]; then
            while IFS= read -r line; do
                case "$line" in
                    '[PYTHON_ENV]'*|'[ENV] node='*|'[TORCH]'*|'[NPU]'*|'[FS]'*|'[SHA]'*|\
                    '[INSTALL_PATH]'*|'[VERIFY]'*|'[PREFLIGHT]'*)
                        printf '[node=%s host=%s] %s\n' "$rank" "${HOSTS[$rank]}" "$line" ;;
                esac
            done < "$LOG_DIR/node_${rank}_${HOSTS[$rank]}_${stage}.log"
        fi
    done
    PIDS=()
    return "$failed"
}

die_stage() {
    unset SSHPASS
    echo "[ERROR] $1 failed; no tests launched." >&2
    exit 1
}

run_stage preflight0 0 || die_stage 'Preflight (rank 0)'
SHARED=0
if (( NUM_NODES > 1 )); then
    RANKS_REST=()
    for ((rank = 1; rank < NUM_NODES; rank++)); do RANKS_REST+=("$rank"); done
    run_stage preflight "${RANKS_REST[@]}" || die_stage 'Preflight'
    for rank in "${RANKS_REST[@]}"; do
        if grep -q '\[FS\] shared=1' "$LOG_DIR/node_${rank}_${HOSTS[$rank]}_preflight.log"; then
            SHARED=1
        fi
    done
fi

if [[ $BUILD_LAYOUT == auto ]]; then
    BUILD_TARGETS=()
    if (( SHARED == 1 )); then
        BUILD_TARGETS=(0)
    else
        for ((rank = 0; rank < NUM_NODES; rank++)); do BUILD_TARGETS+=("$rank"); done
    fi
elif [[ $BUILD_LAYOUT == rank0 ]]; then
    if (( SHARED != 1 && NUM_NODES != 1 )); then
        unset SSHPASS
        echo '[ERROR] BUILD_LAYOUT=rank0 needs a shared CODE_DIR; other nodes would have no build.' >&2
        exit 1
    fi
    BUILD_TARGETS=(0)
else
    if (( SHARED == 1 )); then
        unset SSHPASS
        echo '[ERROR] CODE_DIR is on a shared filesystem; BUILD_LAYOUT=all would corrupt concurrent in-place builds. Use auto or rank0.' >&2
        exit 1
    fi
    BUILD_TARGETS=()
    for ((rank = 0; rank < NUM_NODES; rank++)); do BUILD_TARGETS+=("$rank"); done
fi
if [[ $SYNC_GIT == 1 ]]; then
    if (( SHARED == 1 )); then SYNC_TARGETS=(0); else SYNC_TARGETS=("${BUILD_TARGETS[@]}"); fi
fi
printf '[LAYOUT] shared=%s build=%s\n' "$SHARED" "$BUILD_STAGE"

if [[ $SYNC_GIT == 1 ]]; then
    run_stage sync "${SYNC_TARGETS[@]}" || die_stage 'Git sync'
fi
if [[ $BUILD_STAGE == build ]]; then
    run_stage build "${BUILD_TARGETS[@]}" || die_stage 'Build'
fi
run_stage verify || die_stage 'Verify'
if [[ $MODE == prepare ]]; then
    unset SSHPASS
    printf '[DONE] preparation complete; nodes=%s cards=%s controller_logs=%s; no tests launched\n' "$NUM_NODES" "$((NPROC * NUM_NODES))" "$LOG_DIR"
    exit 0
fi
launch_rc=0
run_stage run || launch_rc=$?
unset SSHPASS
printf '[DONE] exit=%s controller_logs=%s\n' "$launch_rc" "$LOG_DIR"
printf '[DONE] per-node results stay under %s on each node (e.g. test/bigop_fused/run_speed*.log)\n' "$CODE_DIR"
exit "$launch_rc"
