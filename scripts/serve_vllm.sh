#!/usr/bin/env bash
# vLLM replicas of the generator G (Step 2 candidates, MC rollouts), one per GPU group.
#
#   scripts/serve_vllm.sh start  --model PATH --name NAME --gpus 0,1 --tp 1 --port-base 8401 \
#                                [--max-len 24576] [--gpu-util 0.90] [--max-images 16] \
#                                [--log-requests] [--force] [-- <extra vllm args>]
#   scripts/serve_vllm.sh wait   --name NAME [--timeout 150]   # 0 ready | 1 a replica died | 2 still loading
#   scripts/serve_vllm.sh status --name NAME
#   scripts/serve_vllm.sh urls   --name NAME                   # http://127.0.0.1:<port>,... of healthy replicas
#   scripts/serve_vllm.sh stop   --name NAME
#
# replicas = len(gpus) / tp; replica r serves on port-base + r from GPUs
# gpus[r*tp : (r+1)*tp]. NAME is both the served model name (the client's
# `served_model`) and the state dir ${MEDTRACE_SERVE_ROOT:-serve}/NAME/ (a relative
# root is taken from the working directory) holding replica_<port>.{pid,gpus,log}
# and launch.txt.
#
# Environment of vLLM (nothing about the host is assumed):
#   MEDTRACE_VLLM_PYTHON     interpreter with vllm installed (default: python, after the conda env below)
#   MEDTRACE_VLLM_CONDA_ENV  optional conda env to activate first; conda.sh from MEDTRACE_CONDA_SH,
#                            else from `conda info --base`
#   MEDTRACE_VLLM_LIB        optional directory prepended to LD_LIBRARY_PATH (e.g. the env's lib/ when
#                            its libstdc++ is newer than the system's)
#   MEDTRACE_BUSY_MIB        a GPU with more MiB in use is refused unless --force (default 2000)
#
# Choices that keep eq:temperature honest and a shared host safe:
#   --generation-config vllm   no checkpoint sampling defaults (some checkpoints ship
#                              top_k/top_p) that an unsent field could fall back to;
#                              the client sends every field anyway.
#   --limit-mm-per-prompt      vLLM's default admits too few images for M' slices.
#   --enable-prefix-caching    N candidates of a patient share the image prefill, and MC
#                              rollouts share their trajectory prefixes.
#   --host 127.0.0.1           not reachable from outside the host.
#   A GPU that is already in use is refused unless --force (someone else's job).
#   Each replica runs in its own session (setsid): `stop` signals exactly the process
#   groups recorded in its pid files (API server + engine + TP workers), never a pattern.
set -euo pipefail

SERVE_ROOT=${MEDTRACE_SERVE_ROOT:-serve}
case "$SERVE_ROOT" in /*) ;; *) SERVE_ROOT="$PWD/$SERVE_ROOT" ;; esac
BUSY_MIB=${MEDTRACE_BUSY_MIB:-2000}

die() { echo "serve_vllm: $*" >&2; exit 1; }

cmd=${1:-}; shift || true
MODEL=""; NAME=""; GPUS=""; TP=1; PORT_BASE=8401; MAX_LEN=24576; GPU_UTIL=0.90; MAX_IMAGES=16
LOG_REQ=0; FORCE=0; TIMEOUT=150; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    --port-base) PORT_BASE=$2; shift 2 ;;
    --max-len) MAX_LEN=$2; shift 2 ;;
    --gpu-util) GPU_UTIL=$2; shift 2 ;;
    --max-images) MAX_IMAGES=$2; shift 2 ;;
    --log-requests) LOG_REQ=1; shift ;;
    --force) FORCE=1; shift ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown option $1 (see the header of $0)" ;;
  esac
done
[ -n "$NAME" ] || die "--name is required"
case "$NAME" in *[!A-Za-z0-9._-]*) die "--name may use only [A-Za-z0-9._-]" ;; esac
DIR="$SERVE_ROOT/$NAME"

# pid of a replica if it is alive AND is a vLLM process (guards against pid reuse)
live_pid() {
  local f=$1 pid
  pid=$(cat "$f" 2>/dev/null) || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q vllm || return 1
  echo "$pid"
}

port_of() { local b; b=$(basename "$1" .pid); echo "${b#replica_}"; }

health() { curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$1/health" 2>/dev/null || true; }

gpu_mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ' ; }

# the vLLM interpreter: optional conda env, then MEDTRACE_VLLM_PYTHON (default python)
vllm_python() {
  if [ -n "${MEDTRACE_VLLM_CONDA_ENV:-}" ]; then
    local sh=${MEDTRACE_CONDA_SH:-}
    if [ -z "$sh" ]; then
      command -v conda >/dev/null || die "MEDTRACE_VLLM_CONDA_ENV is set but conda is not on PATH; set MEDTRACE_CONDA_SH"
      sh="$(conda info --base)/etc/profile.d/conda.sh"
    fi
    [ -f "$sh" ] || die "conda.sh not found at $sh"
    set +u
    # shellcheck disable=SC1090
    source "$sh"
    conda activate "$MEDTRACE_VLLM_CONDA_ENV"
    set -u
  fi
  PY=${MEDTRACE_VLLM_PYTHON:-python}
  "$PY" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('vllm') is None)" 2>/dev/null \
    || die "$PY cannot import vllm (set MEDTRACE_VLLM_PYTHON or MEDTRACE_VLLM_CONDA_ENV)"
}

start() {
  [ -n "$MODEL" ] || die "--model is required"
  [ -d "$MODEL" ] || die "model dir $MODEL not found"
  [ -n "$GPUS" ] || die "--gpus is required"
  local pool=() g
  IFS=',' read -r -a pool <<< "$GPUS"
  for g in "${pool[@]}"; do
    [[ "$g" =~ ^[0-9]+$ ]] || die "GPU id '$g' in --gpus $GPUS is not a non-negative integer"
  done
  [ "$(printf '%s\n' "${pool[@]}" | sort -u | wc -l)" -eq "${#pool[@]}" ] || die "duplicate GPU in --gpus $GPUS"
  [ "$TP" -ge 1 ] && [ $(( ${#pool[@]} % TP )) -eq 0 ] || die "${#pool[@]} GPUs is not a multiple of --tp $TP"
  local n=$(( ${#pool[@]} / TP ))

  shopt -s nullglob
  local f old
  for f in "$DIR"/replica_*.pid; do
    live_pid "$f" >/dev/null && die "$NAME already has a live replica ($(basename "$f")); stop it first"
    old=$(cat "$f" 2>/dev/null || true)
    if [[ "$old" =~ ^[0-9]+$ ]] && pgrep -g "$old" >/dev/null 2>&1; then
      die "a dead replica of $NAME left processes in group $old (engine/workers); run stop first"
    fi
  done
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  local mem used
  mem=$(gpu_mem)
  for g in "${pool[@]}"; do
    used=$(echo "$mem" | awk -F, -v g="$g" '$1==g {print $2}')
    [ -n "$used" ] || die "nvidia-smi does not list GPU $g"
    if [ "$used" -gt "$BUSY_MIB" ] && [ "$FORCE" = 0 ]; then
      die "GPU $g has ${used} MiB in use (someone else's job?); refusing (--force to override)"
    fi
  done
  local r port
  for ((r = 0; r < n; r++)); do
    port=$((PORT_BASE + r))
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then die "port $port is already in use"; fi
  done
  mkdir -p "$DIR"
  rm -f "$DIR"/replica_*.pid "$DIR"/replica_*.gpus    # stale entries of dead replicas

  vllm_python
  # no trailing ':' when unset (an empty entry would put the cwd on the library path)
  if [ -n "${MEDTRACE_VLLM_LIB:-}" ]; then
    export LD_LIBRARY_PATH="$MEDTRACE_VLLM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  # CUDA_VISIBLE_DEVICES must number GPUs as nvidia-smi does (the busy check above)
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  local logflag=--no-enable-log-requests
  [ "$LOG_REQ" = 1 ] && logflag=--enable-log-requests

  : > "$DIR/launch.txt"
  local devs k pid
  for ((r = 0; r < n; r++)); do
    port=$((PORT_BASE + r))
    devs=""
    for ((k = 0; k < TP; k++)); do devs+="${pool[$((r * TP + k))]},"; done
    devs=${devs%,}
    local args=(-m vllm.entrypoints.openai.api_server
      --model "$MODEL" --served-model-name "$NAME"
      --host 127.0.0.1 --port "$port"
      --tensor-parallel-size "$TP"
      --gpu-memory-utilization "$GPU_UTIL"
      --max-model-len "$MAX_LEN"
      --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}"
      --enable-prefix-caching
      --generation-config vllm
      --disable-uvicorn-access-log
      "$logflag")
    [ "${#EXTRA[@]}" -gt 0 ] && args+=("${EXTRA[@]}")
    echo "port=$port gpus=$devs cmd: $PY ${args[*]}" >> "$DIR/launch.txt"
    CUDA_VISIBLE_DEVICES=$devs setsid nohup "$PY" "${args[@]}" \
      > "$DIR/replica_$port.log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$DIR/replica_$port.pid"
    echo "$devs" > "$DIR/replica_$port.gpus"
    echo "replica $r: port $port  GPUs $devs  pid $pid  log $DIR/replica_$port.log"
  done
  echo "loading; '$0 wait --name $NAME' blocks until every replica answers /health"
}

status() {
  shopt -s nullglob
  local f any=0 port pid
  for f in "$DIR"/replica_*.pid; do
    any=1
    port=$(port_of "$f")
    if pid=$(live_pid "$f"); then
      echo "port $port  pid $pid  gpus $(cat "$DIR/replica_$port.gpus" 2>/dev/null)  health=$(health "$port")"
    else
      echo "port $port  pid $(cat "$f")  DEAD (see $DIR/replica_$port.log)"
    fi
  done
  [ "$any" = 1 ] || echo "no replicas registered for $NAME"
}

urls() {
  shopt -s nullglob
  local f out="" port
  for f in "$DIR"/replica_*.pid; do
    port=$(port_of "$f")
    live_pid "$f" >/dev/null && [ "$(health "$port")" = 200 ] && out+="http://127.0.0.1:$port,"
  done
  echo "${out%,}"
}

wait_ready() {
  shopt -s nullglob
  local files=("$DIR"/replica_*.pid) t0=$SECONDS f port ready
  [ "${#files[@]}" -gt 0 ] || die "no replicas registered for $NAME"
  while :; do
    ready=0
    for f in "${files[@]}"; do
      port=$(port_of "$f")
      live_pid "$f" >/dev/null || { echo "replica on port $port died; tail of its log:" >&2
                                     tail -n 20 "$DIR/replica_$port.log" >&2; exit 1; }
      [ "$(health "$port")" = 200 ] && ready=$((ready + 1))
    done
    if [ "$ready" = "${#files[@]}" ]; then echo "ready: $(urls)"; return 0; fi
    if [ $((SECONDS - t0)) -ge "$TIMEOUT" ]; then echo "not ready after ${TIMEOUT}s ($ready/${#files[@]})"; exit 2; fi
    sleep 5
  done
}

stop() {
  shopt -s nullglob
  local f pid port pgid groups=() gpus=""
  for f in "$DIR"/replica_*.pid; do
    port=$(port_of "$f")
    gpus+="$(cat "$DIR/replica_$port.gpus" 2>/dev/null),"
    if pid=$(live_pid "$f"); then
      pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
      if [ "$pgid" = "$pid" ]; then
        echo "stopping replica on port $port (process group $pid)"
        kill -TERM -- "-$pid" 2>/dev/null || true
        groups+=("$pid")
      else
        echo "stopping replica on port $port (pid $pid)"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    else
      pid=$(cat "$f" 2>/dev/null || true)
      # the API server may be gone while its engine/workers (same group) linger
      if [[ "$pid" =~ ^[0-9]+$ ]] && pgrep -g "$pid" >/dev/null 2>&1; then
        echo "port $port: server gone, signalling leftover process group $pid"
        kill -TERM -- "-$pid" 2>/dev/null || true
        groups+=("$pid")
      fi
    fi
  done
  local waited=0 alive g
  while :; do
    alive=0
    for g in ${groups[@]+"${groups[@]}"}; do pgrep -g "$g" >/dev/null 2>&1 && alive=1; done
    [ "$alive" = 0 ] && break
    if [ "$waited" -ge 60 ]; then
      for g in "${groups[@]}"; do
        pgrep -g "$g" >/dev/null 2>&1 && { echo "group $g ignored SIGTERM for 60s; SIGKILL"; kill -KILL -- "-$g" 2>/dev/null || true; }
      done
      sleep 3
      break
    fi
    sleep 2; waited=$((waited + 2))
  done
  for f in "$DIR"/replica_*.pid; do rm -f "$f"; done
  echo "stopped $NAME"
  gpus=${gpus%,}
  [ -n "$gpus" ] && command -v nvidia-smi >/dev/null && \
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$gpus" | sed 's/^/  GPU /'
  return 0
}

case "$cmd" in
  start) start ;;
  wait) wait_ready ;;
  status) status ;;
  urls) urls ;;
  stop) stop ;;
  *) die "usage: $0 {start|wait|status|urls|stop} --name NAME [options]" ;;
esac
