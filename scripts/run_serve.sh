#!/usr/bin/env bash
# Serve a config's generator G with exactly the arguments the config records, so the
# replicas, the served model name and Step 2's client can never disagree.
#
#   bash scripts/run_serve.sh <config> start [-- <extra serve_vllm.sh start args>]
#   bash scripts/run_serve.sh <config> wait|status|urls|stop
#
#   bash scripts/run_serve.sh configs/binary.yaml start
#   bash scripts/run_serve.sh configs/binary.yaml wait && python -m medtrace generate --config configs/binary.yaml
#
# Reads generation.{model, served_model, serve.{root, gpus, tp, port_base, max_len}} and calls
# scripts/serve_vllm.sh, which refuses busy cards and stops only the process groups it
# recorded. `served_model` is both --served-model-name and the state dir
# <serve.root>/<served_model>/ that Step 2 reads the live replicas from. Run it from the
# directory the pipeline runs in: relative roots (models_root, serve.root) resolve against it.
#
# Environment (nothing about the host is assumed):
#   MEDTRACE_PYTHON      interpreter with medtrace's dependencies, reads the config (default: python)
#   MEDTRACE_VLLM_*      the vLLM environment, see scripts/serve_vllm.sh
set -euo pipefail

CONFIG=${1:?usage: run_serve.sh <config> <start|wait|status|urls|stop> [args]}
CMD=${2:?usage: run_serve.sh <config> <start|wait|status|urls|stop> [args]}
shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
case "$CONFIG" in
  /*) ;;
  *) [ -f "$CONFIG" ] && CONFIG="$PWD/$CONFIG" || CONFIG="$REPO/$CONFIG" ;;
esac
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 2; }
[ "${1:-}" = "--" ] && shift

PY=${MEDTRACE_PYTHON:-python}
read -r MODEL NAME GPUS TP PORT MAXLEN ROOT < <(PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" \
                                                 "$PY" - "$CONFIG" <<'EOF'
import sys
from pathlib import Path
from medtrace.config import load_config
from medtrace.pipeline import stages
cfg = load_config(sys.argv[1])
g = cfg["generation"]
if g.get("backend", "vllm") != "vllm":
    sys.exit(f"generation.backend is {g.get('backend')!r}; nothing to serve")
s = g["serve"]
print(stages.resolve_paths(cfg).model(g["model"]), g["served_model"], ",".join(str(x) for x in s["gpus"]),
      s["tp"], s["port_base"], s["max_len"], Path(str(s.get("root") or "serve")).resolve())
EOF
)
[ -n "${ROOT:-}" ] || { echo "could not resolve generation.serve from $CONFIG" >&2; exit 2; }
export MEDTRACE_SERVE_ROOT="$ROOT"
case "$CMD" in
  start) exec bash "$REPO/scripts/serve_vllm.sh" start --model "$MODEL" --name "$NAME" --gpus "$GPUS" \
              --tp "$TP" --port-base "$PORT" --max-len "$MAXLEN" "$@" ;;
  wait|status|urls|stop) exec bash "$REPO/scripts/serve_vllm.sh" "$CMD" --name "$NAME" "$@" ;;
  *) echo "unknown command '$CMD'" >&2; exit 2 ;;
esac
