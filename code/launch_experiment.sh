#!/bin/bash
# Launch one train_evaluate.py config via SLURM (if available), tmux (if
# available), or both -- SLURM for the GPU allocation, tmux inside that
# allocation so you can `ssh` to the node and attach if you ever need to
# look beyond WandB. Falls back to a plain foreground run if neither exists.
#
# Usage: ./launch_experiment.sh configs/uni2h_mil.yaml [--no-slurm] [--tmux]
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="$1"; shift
USE_SLURM=1
FORCE_TMUX=0
for arg in "$@"; do
  case "$arg" in
    --no-slurm) USE_SLURM=0 ;;
    --tmux) FORCE_TMUX=1 ;;
  esac
done

RUN_NAME=$(basename "$CONFIG" .yaml)
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
CMD="python3 train_evaluate.py --config $CONFIG"

if [ "$USE_SLURM" = "1" ] && command -v sbatch >/dev/null 2>&1; then
    SBATCH_SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=path_${RUN_NAME}"
        echo "#SBATCH --output=${LOG_DIR}/${RUN_NAME}_%j.out"
        echo "#SBATCH --error=${LOG_DIR}/${RUN_NAME}_%j.err"
        echo "#SBATCH --gres=gpu:1"
        echo "#SBATCH --cpus-per-task=8"
        echo "#SBATCH --mem=32G"
        echo "#SBATCH --time=04:00:00"
        echo "cd $(pwd)"
        if [ "$FORCE_TMUX" = "1" ] && command -v tmux >/dev/null 2>&1; then
            echo "tmux new-session -d -s path_${RUN_NAME} '$CMD'"
            echo "wait \$(tmux list-panes -t path_${RUN_NAME} -F '#{pane_pid}')"
        else
            echo "$CMD"
        fi
    } > "$SBATCH_SCRIPT"
    JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $NF}')
    echo "Submitted via SLURM: $RUN_NAME (job $JOB_ID) -- logs at ${LOG_DIR}/${RUN_NAME}_${JOB_ID}.out"
elif command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "path_${RUN_NAME}" "$CMD 2>&1 | tee ${LOG_DIR}/${RUN_NAME}.log"
    echo "Launched in tmux session: path_${RUN_NAME} (attach with: tmux attach -t path_${RUN_NAME})"
else
    echo "Neither SLURM nor tmux found; running in foreground."
    eval "$CMD"
fi
