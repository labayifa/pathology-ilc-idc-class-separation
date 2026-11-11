# Pathology Foundation Model Project

Frozen pathology foundation-model features (UNI2-h) vs. a baseline ImageNet
CNN, for Invasive Lobular Carcinoma (ILC) vs. Invasive Ductal Carcinoma (IDC)
classification on TCGA-BRCA whole-slide images. See `paper/paper.tex` for
the full write-up, `notes/external_benchmarks.md` for how results compare
to prior work, and `results/results_summary.csv` for the complete
172-configuration sweep.

**Headline result**: frozen UNI2-h + a gated-attention MIL head reaches
patient-level AUROC = 0.920, matching a fully fine-tuned Inception-v3 CNN
(Ektefaie et al. 2021, *npj Breast Cancer*) on the same dataset and task,
with zero backbone gradient updates.

## Repo structure

```
code/            Pipeline scripts (see "Reproducing the pipeline" below)
cohort/          Cohort definitions (patient IDs, classes, fold assignments)
results/         results_summary.csv (all 172 configs) + per-patient predictions
notebooks/       Interactive slide exploration / preprocessing walkthrough
notes/           Working notes: licensing verification, external benchmarks
paper/           NeurIPS-format workshop paper draft (paper.tex, references.bib)
```

**Not included** (by design, not oversight): raw whole-slide images, extracted
tile-feature tensors (`.pt` files), and model checkpoints. These are either
large binary derivatives of licensed, non-redistributable TCGA/foundation-model
data, or are fully regenerable from the steps below. See "Data and licensing".

## Data and licensing

- **TCGA-BRCA slides**: public, from the NCI Genomic Data Commons (GDC), under
  TCGA's standard data use policy. Downloaded via `code/download_tcga.py`
  (async, resumable, md5-verified) using `code/select_cohort.py`'s cohort
  definition and a GDC manifest.
- **UNI2-h** (Mahmood Lab): CC-BY-NC-ND 4.0, gated (institutional email
  required). Used strictly as a frozen feature extractor, no redistribution
  of weights or derived embeddings beyond the small `.pt` feature files this
  pipeline produces for its own training use.
- **Baseline features**: torchvision ResNet-18, ImageNet-pretrained, Apache
  2.0 / torchvision license.

## Reproducing the pipeline

All scripts live in `code/`. Python 3.11+ (3.13 recommended), see
`code/requirements.txt`. Tiling/extraction additionally needs OpenSlide
(`brew install openslide-bin && pip install openslide-python` on macOS;
`apt install openslide-tools && pip install openslide-python` on Ubuntu).

1. **Cohort selection** — `python select_cohort.py --tcga-dir <TCGA dir> --out ../cohort/cohort.csv --target-ilc 55`
   (deterministic, seed 42; the 109-patient cohort used in the paper is
   already committed at `cohort/cohort.csv`). `--target-ilc 999` selects the
   larger 348-patient expansion (`cohort/cohort_expanded.csv`).
2. **Download slides** — `python download_tcga.py --cohort ../cohort/cohort.csv --out-dir <slide dir>`
   (idempotent/resumable; or `stream_cohort_expansion.py` for the batched
   download+extract+delete pipeline used for the 348-patient expansion,
   which never holds more than one batch of raw slides on disk at once).
3. **Tile + extract frozen features** — `python tile_and_extract.py --cohort ../cohort/cohort.csv --slide-dir <slide dir> --out-dir <features dir>`
   (produces one `.pt` per patient with both `baseline_cnn` and `uni2h`
   tile-feature tensors, tile coordinates, and tissue fractions). Set
   `features_dir` in `configs/base.yaml` to this path.
4. **Fold assignment** — `python cv_folds.py --cohort ../cohort/cohort.csv --features-dir <features dir> --out ../cohort/folds.csv --n-folds 5 --n-seeds 5`
   (site-grouped, greedy bin-packed; already committed at `cohort/folds.csv`).
5. **Train + evaluate one config** — `python train_evaluate.py --config configs/<name>.yaml`
   (20 architecture configs in `configs/`, e.g. `uni2h_mil.yaml`). Logs to
   WandB, a SQLite experiment DB (`results/experiments.db`, not committed),
   and appends to `results/results_summary.csv`. Supports resuming from a
   checkpoint after interruption (e.g. a Spot-instance reclaim).
6. **Hyperparameter ablation sweep** — `python run_ablation.py --grid configs/ablation_mlp_mil.yaml`
   (or `ablation_svm_kernel.yaml`) expands the grid spec into per-combination
   configs under `configs/ablation_generated/` (not committed, regenerate with
   `--dry-run` to inspect without running) and runs each.
7. **Run everything in parallel on one GPU** — `run_full_sweep_parallel.sh [concurrency]`
   orchestrates all 172 configs with a bounded concurrency pool (these are
   small models; one A10G comfortably runs 6+ concurrently), skipping configs
   already present in `results_summary.csv`.
8. **Ensemble experiments** — `python ensemble_predictions.py --run-names <config names>`
   combines saved per-patient predictions (`results/predictions/*.csv`) into
   a mean-probability ensemble with bootstrap CIs.
9. **Exploratory LLM baseline** — `python llm_classifier.py --config configs/uni2h_llm.yaml`
   (local Qwen2.5-0.5B-Instruct, no external API).

### Notes on the real run

The sweep reported in `results/results_summary.csv` ran on an AWS g5.2xlarge
(1x A10G) with `run_full_sweep_parallel.sh 6`, inside `tmux` so it survived
SSH disconnects. All 172 configs completed with 0 failures. Training
checkpoints keep only the latest (for resuming) and best (by validation
AUROC) model per fold, deleting the latest once a fold completes — this keeps
checkpoint storage bounded but it is not committed here; a full checkpoint
re-run from these configs reproduces the same results_summary.csv up to
seed-dependent stochastic training variance.
