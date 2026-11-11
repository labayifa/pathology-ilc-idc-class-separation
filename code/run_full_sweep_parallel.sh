#!/bin/bash
# Same 174-config sweep as run_full_sweep.sh, but runs several configs
# concurrently on one GPU (these are tiny models -- a single run barely
# touched GPU memory or utilization) instead of one at a time. Each job
# gets its own log file; checkpoints are already scoped per run_name so
# concurrent runs never collide there. Configs already present in
# results_summary.csv are skipped, so re-running this after a partial sweep
# (sequential or parallel) only does the remaining work.
#
# Usage: ./run_full_sweep_parallel.sh [concurrency, default 6]
set -uo pipefail
cd "$(dirname "$0")"
source ~/pathology_venv/bin/activate
export WANDB_BASE_URL=https://api.forge.coreweave.com

CONCURRENCY="${1:-6}"
RESULTS_CSV="../results/results_summary.csv"
mkdir -p ../logs

echo "=== Regenerating ablation configs ==="
python3 run_ablation.py --grid configs/ablation_svm_kernel.yaml --dry-run
python3 run_ablation.py --grid configs/ablation_mlp_mil.yaml --dry-run

run_name_of() {
    grep -m1 '^run_name:' "$1" | sed 's/run_name: *"\?\([^"]*\)"\?/\1/'
}

JOBLIST=$(mktemp)
# Shuffle so heavy (GAT-family) and light (lr/svm/mlp) configs interleave
# across the concurrency pool instead of clustering together alphabetically
# (all *_gat* configs sort adjacent, which would contend for the same CPU/GPU
# capacity at the same time instead of overlapping with lighter jobs).
for cfg in $(ls configs/*.yaml configs/ablation_generated/*.yaml | shuf); do
    base=$(basename "$cfg")
    case "$base" in
        base.yaml|ablation_svm_kernel.yaml|ablation_mlp_mil.yaml) continue ;;
        *_llm.yaml) entry="llm_classifier.py" ;;
        *) entry="train_evaluate.py" ;;
    esac
    run_name=$(run_name_of "$cfg")
    if [ -f "$RESULTS_CSV" ] && grep -q "^${run_name}," "$RESULTS_CSV"; then
        echo "skip (already done): $run_name"
        continue
    fi
    echo "${entry}::${cfg}" >> "$JOBLIST"
done

n_jobs=$(wc -l < "$JOBLIST")
echo "=== $n_jobs job(s) queued, concurrency=$CONCURRENCY ==="

run_one() {
    pair="$1"
    entry="${pair%%::*}"
    cfg="${pair##*::}"
    run_name=$(grep -m1 '^run_name:' "$cfg" | sed 's/run_name: *"\?\([^"]*\)"\?/\1/')
    echo "[$(date +%H:%M:%S)] START $run_name ($entry)"
    python3 "$entry" --config "$cfg" > "../logs/${run_name}.log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE  $run_name"
    else
        echo "[$(date +%H:%M:%S)] FAILED $run_name (exit $status) -- see ../logs/${run_name}.log"
    fi
}
export -f run_one

cat "$JOBLIST" | xargs -P "$CONCURRENCY" -I{} bash -c 'run_one "$@"' _ {}
rm -f "$JOBLIST"
echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
