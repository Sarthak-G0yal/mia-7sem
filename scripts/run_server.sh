#!/usr/bin/env bash
# Run the experiments unattended on a workstation server and bundle the results.
#
#   bash scripts/run_server.sh miniproject          # ~1 h on a small GPU: Duke DME, 8 methods x 5 folds
#   bash scripts/run_server.sh project              # long: NA-DyCNN, D2 + D1 sources x 3 seeds x 6 methods
#   bash scripts/run_server.sh all                  # mini-project first, then the project
#   bash scripts/run_server.sh evaluate             # no training: re-score saved checkpoints of both parts
#
#   bash scripts/run_server.sh project --gpu 1      # pick a GPU by its nvidia-smi index (or --gpu cpu; GPU=1 works too)
#   bash scripts/run_server.sh miniproject --gpu 0 & bash scripts/run_server.sh project --gpu 1 &   # both GPUs at once
#   PYTHON="python" bash scripts/run_server.sh ...  # use the active env instead of uv
#   nohup bash scripts/run_server.sh all > logs/server.out 2>&1 &   # keep running after logout
#
# Extra arguments after the part name go to the training script, e.g.
#   bash scripts/run_server.sh project --sources d2 --seeds 0
set -euo pipefail
cd "$(dirname "$0")/.."

PART="${1:-all}"
shift || true

# --gpu N|cpu anywhere in the arguments (or GPU=N in the environment); removed before passing the rest on.
DEVICE="${GPU:-0}"
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) DEVICE="${2:?--gpu needs a GPU index or cpu}"; shift 2 ;;
        --gpu=*) DEVICE="${1#--gpu=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # same GPU numbering as nvidia-smi
if [ "$DEVICE" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES=""
elif [[ "$DEVICE" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="$DEVICE"
else
    echo "--gpu must be a GPU index (0, 1, ...) or cpu, got: $DEVICE"; exit 2
fi
echo ">>> device: ${DEVICE} (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}')"
export MPLBACKEND=Agg
PYTHON="${PYTHON:-uv run python}"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)_dev${DEVICE}"

run() {  # run <log name> <script> [args...]
    local name="$1"; shift
    echo ">>> $* (log: logs/${name}_${STAMP}.log)"
    $PYTHON -u "$@" 2>&1 | tee "logs/${name}_${STAMP}.log"
}

case "$PART" in
    miniproject) run miniproject scripts/miniproject_train.py "$@" ;;
    project)     run project scripts/project_train.py "$@" ;;
    all)         run miniproject scripts/miniproject_train.py
                 run project scripts/project_train.py "$@" ;;
    evaluate)    if [ -d runs ]; then run project_eval scripts/project_evaluate.py --profile; fi
                 if [ -d runs_seg ]; then run miniproject_eval scripts/miniproject_evaluate.py; fi ;;
    *) echo "usage: $0 {miniproject|project|all|evaluate} [script args]"; exit 2 ;;
esac

$PYTHON scripts/collect_results.py
