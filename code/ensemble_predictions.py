"""Combines the saved per-patient OOF predictions from several already-run
configs into a simple mean-probability ensemble -- no retraining needed,
since train_evaluate.py now saves every config's per-patient prediction to
results/predictions/{run_name}.csv. The goal is to see whether combining
classifier heads (and later, foundation models once
Phikon-v2/Virchow2 finish) beats any single config, and how that compares to
the Ektefaie et al. 2021 (npj Breast Cancer) external benchmark of 0.920
patient-level AUC for lobular-vs-ductal on TCGA-BRCA.

Usage:
    python ensemble_predictions.py --run-names uni2h_mil_... uni2h_gat_mil uni2h_mlp_...
    python ensemble_predictions.py --glob "uni2h_mil*"
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from train_evaluate import LABEL_MAP, append_csv_row_locked, bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ensemble_predictions")


def load_predictions(path: Path) -> dict[str, float]:
    with open(path) as f:
        return {row["patient_id"]: float(row["pred_prob_idc"]) for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, default=Path("../results/predictions"))
    parser.add_argument("--results-csv", type=Path, default=Path("../results/results_summary.csv"))
    parser.add_argument("--run-names", nargs="*", default=None, help="exact run_names to ensemble")
    parser.add_argument("--glob", type=str, default=None, help="glob pattern over run_names, e.g. 'uni2h_mil*'")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                         help="optional per-run weights, same order as --run-names; default: equal weight")
    parser.add_argument("--ensemble-name", type=str, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    if args.glob:
        run_names = sorted(p.stem for p in args.predictions_dir.glob(f"{args.glob}.csv"))
    elif args.run_names:
        run_names = args.run_names
    else:
        raise SystemExit("pass either --run-names or --glob")

    if len(run_names) < 2:
        raise SystemExit(f"need at least 2 configs to ensemble, got {len(run_names)}: {run_names}")

    log.info(f"Ensembling {len(run_names)} configs: {run_names}")
    per_config = {name: load_predictions(args.predictions_dir / f"{name}.csv") for name in run_names}

    patient_ids = set.intersection(*(set(preds.keys()) for preds in per_config.values()))
    missing = set.union(*(set(preds.keys()) for preds in per_config.values())) - patient_ids
    if missing:
        log.warning(f"{len(missing)} patient(s) not present in all configs, excluded from ensemble: {missing}")

    weights = args.weights if args.weights else [1.0] * len(run_names)
    if len(weights) != len(run_names):
        raise SystemExit(f"--weights has {len(weights)} values but {len(run_names)} run_names were given")
    weights = np.array(weights) / sum(weights)

    dx_class = {}
    with open(args.predictions_dir / f"{run_names[0]}.csv") as f:
        for row in csv.DictReader(f):
            dx_class[row["patient_id"]] = row["dx_class"]

    ensemble_pred = {}
    for pid in patient_ids:
        ensemble_pred[pid] = float(sum(w * per_config[name][pid] for w, name in zip(weights, run_names)))

    y_true = np.array([LABEL_MAP[dx_class[pid]] for pid in patient_ids])
    y_prob = np.array([ensemble_pred[pid] for pid in patient_ids])

    auroc = bootstrap_ci(y_true, y_prob, roc_auc_score, args.bootstrap_resamples, 0.95, 0)
    bal_acc = bootstrap_ci(y_true, y_prob, lambda yt, yp: balanced_accuracy_score(yt, (yp >= 0.5).astype(int)),
                            args.bootstrap_resamples, 0.95, 0)
    brier = bootstrap_ci(y_true, y_prob, brier_score_loss, args.bootstrap_resamples, 0.95, 0)

    log.info(f"FINAL (ensemble of {len(run_names)})  "
             f"AUROC={auroc[0]:.3f} [{auroc[1]:.3f},{auroc[2]:.3f}]  "
             f"BalAcc={bal_acc[0]:.3f} [{bal_acc[1]:.3f},{bal_acc[2]:.3f}]  "
             f"Brier={brier[0]:.3f} [{brier[1]:.3f},{brier[2]:.3f}]  "
             f"n={len(patient_ids)}")

    ensemble_name = args.ensemble_name or "ensemble_" + "_".join(run_names)[:200]
    append_csv_row_locked(
        args.results_csv,
        ["run_name", "feature_set", "classifier_head", "n_patients",
         "auroc", "auroc_ci_lo", "auroc_ci_hi",
         "balanced_accuracy", "balanced_accuracy_ci_lo", "balanced_accuracy_ci_hi",
         "brier", "brier_ci_lo", "brier_ci_hi", "wandb_url"],
        [ensemble_name, "+".join(sorted(set(run_names))), "ensemble", len(patient_ids),
         auroc[0], auroc[1], auroc[2], bal_acc[0], bal_acc[1], bal_acc[2],
         brier[0], brier[1], brier[2], ""],
    )
    log.info(f"Appended to {args.results_csv}")


if __name__ == "__main__":
    main()
