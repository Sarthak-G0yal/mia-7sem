#!/usr/bin/env bash
# Run the experiments unattended on a workstation server and bundle the results.
#
#   bash scripts/run_server.sh miniproject          # ~1 h on a small GPU: Duke DME, 8 methods x 5 folds
#   bash scripts/run_server.sh project              # long: NA-DyCNN, D2 + D1 sources x 3 seeds x 6 methods
#   bash scripts/run_server.sh all                  # mini-project first, then the project
#   bash scripts/run_server.sh evaluate             # no training: re-score saved checkpoints of both parts
#
#   GPU=1 bash scripts/run_server.sh project        # pick a GPU;  PYTHON="python" to use the active env, not uv
#   nohup bash scripts/run_server.sh all > logs/server.out 2>&1 &   # keep running after logout
#
# Extra arguments after the part name go to the training script, e.g.
#   bash scripts/run_server.sh project --sources d2 --seeds 0
set -euo pipefail
cd "$(dirname "$0")/.."

PART="${1:-all}"
shift || true
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export MPLBACKEND=Agg
PYTHON="${PYTHON:-uv run python}"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"

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
