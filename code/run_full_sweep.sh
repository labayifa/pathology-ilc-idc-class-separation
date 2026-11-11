#!/bin/bash
# Full sweep: 20 architecture configs + 2 LLM configs + the 152-config
# ablation grid (SVM kernels + MLP/MIL dropout x epochs x optimizer x
# scheduler). Meant to run inside tmux so it survives an SSH disconnect.
set -uo pipefail
cd "$(dirname "$0")"
source ~/pathology_venv/bin/activate
export WANDB_BASE_URL=https://api.forge.coreweave.com

echo "=== Regenerating ablation configs ==="
python3 run_ablation.py --grid configs/ablation_svm_kernel.yaml --dry-run
python3 run_ablation.py --grid configs/ablation_mlp_mil.yaml --dry-run

echo "=== Phase 1: architecture + LLM configs ==="
for cfg in configs/*.yaml; do
    base=$(basename "$cfg")
    case "$base" in
        base.yaml|ablation_svm_kernel.yaml|ablation_mlp_mil.yaml) continue ;;
        *_llm.yaml)
            echo ">>> [$(date +%H:%M:%S)] $base (llm_classifier.py)"
            python3 llm_classifier.py --config "$cfg" || echo "!!! FAILED: $base"
            continue
            ;;
    esac
    echo ">>> [$(date +%H:%M:%S)] $base (train_evaluate.py)"
    python3 train_evaluate.py --config "$cfg" || echo "!!! FAILED: $base"
done

echo "=== Phase 2: ablation sweep (152 configs) ==="
for cfg in configs/ablation_generated/*.yaml; do
    echo ">>> [$(date +%H:%M:%S)] $(basename "$cfg") (train_evaluate.py)"
    python3 train_evaluate.py --config "$cfg" || echo "!!! FAILED: $(basename "$cfg")"
done

echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
