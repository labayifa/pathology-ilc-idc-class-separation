"""Exploratory LLM-based classification using summary statistics derived from
the already-extracted foundation-model features -- one concrete resolution
of the plan's flagged-as-ambiguous "LLM-based classification using extracted
features" idea.

Why this design: an LLM can't itself inspect a 1536-dim embedding, so a raw
feature dump would just be numeric noise to it. Instead, per CV fold, the K
feature dimensions that most separate the ILC/IDC TRAIN-set class means are
found, and each held-out patient's embedding is reduced to K signed z-scores
on those dimensions (oriented so positive always means "toward IDC", negative
"toward ILC"). A small local instruction-tuned LLM (no external API, no cost,
no credentials needed) is asked to weigh these K signals and decide.

This is explicitly a toy/sanity-check experiment, not a competitive method:
with the signals already class-oriented, a trivial sign-majority vote is a
natural baseline to compare against (computed here with no LLM involved) --
the real question is only whether the LLM's aggregation adds anything over
that trivial baseline, and a negative result (it doesn't) is a plausible,
reportable outcome per the plan's own ground rules.

Evaluated once (one fold-seed, not repeated across seeds like the
gradient-trained heads in train_evaluate.py) since nothing here is trained --
each patient appears in exactly one of the 5 held-out folds, covering the
full cohort exactly once.

Usage:
    python llm_classifier.py --config configs/uni2h_llm.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_evaluate import LABEL_MAP, append_csv_row_locked, bootstrap_ci, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm_classifier")

PROMPT_TEMPLATE = """You are a careful statistical reasoning assistant. Two breast cancer histology \
subtypes, ILC and IDC, have different average profiles across {k} numeric feature axes, measured from a \
pathology foundation model's embeddings (you cannot see the image itself). For each axis, the sign has \
already been oriented so that a POSITIVE value means this patient leans toward IDC on that axis, and a \
NEGATIVE value means this patient leans toward ILC on that axis. The magnitude is a z-score relative to \
the training population (roughly, how many standard deviations from the overall average).

This patient's {k} oriented signals: {signals}

Weigh all {k} signals together (do not just count how many are positive vs negative -- consider their \
magnitudes too) and decide which subtype this patient's slide more likely is. Respond with EXACTLY two \
lines, nothing else:
LABEL: <ILC or IDC>
CONFIDENCE: <integer 0-100>"""


def compute_oriented_signals(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray,
                              k: int) -> np.ndarray:
    """Returns [n_query, k] z-scores oriented so positive always means "toward IDC"."""
    mean_ilc = x_train[y_train == 0].mean(axis=0)
    mean_idc = x_train[y_train == 1].mean(axis=0)
    overall_mean = x_train.mean(axis=0)
    overall_std = x_train.std(axis=0) + 1e-8
    diff = mean_idc - mean_ilc
    top_idx = np.argsort(-np.abs(diff))[:k]
    signs = np.sign(diff[top_idx])
    z = (x_query[:, top_idx] - overall_mean[top_idx]) / overall_std[top_idx]
    return z * signs[None, :]


def parse_response(text: str) -> tuple[str | None, float | None]:
    label_match = re.search(r"LABEL:\s*(ILC|IDC)", text, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
    label = label_match.group(1).upper() if label_match else None
    conf = float(conf_match.group(1)) if conf_match else None
    return label, conf


def mean_pool(features_dir: Path, feature_set: str, pid: str) -> np.ndarray:
    data = torch.load(features_dir / f"{pid}.pt", weights_only=False)
    return data["features"][feature_set].mean(dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wandb-mode", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.wandb_mode:
        cfg["wandb"]["mode"] = args.wandb_mode
    llm_cfg = cfg["llm"]
    k = llm_cfg["k_signals"]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Config: {cfg['run_name']}  device: {device}")

    with open(Path(__file__).parent / cfg["folds_csv"]) as f:
        patients = list(csv.DictReader(f))
    labels = {p["patient_id"]: LABEL_MAP[p["dx_class"]] for p in patients}
    features_dir = Path(cfg["features_dir"])

    log.info(f"Loading {llm_cfg['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(llm_cfg["model_name"]).to(device)
    model.eval()

    seed = llm_cfg["seed"]
    fold_key = f"fold_seed{seed}"
    llm_preds: dict[str, float] = {}
    baseline_preds: dict[str, float] = {}
    n_label_parsed = 0
    n_conf_parsed = 0

    for fold in range(cfg["n_folds"]):
        train_ids = [p["patient_id"] for p in patients if int(p[fold_key]) != fold]
        val_ids = [p["patient_id"] for p in patients if int(p[fold_key]) == fold]
        log.info(f"fold {fold}: {len(train_ids)} train / {len(val_ids)} val")

        x_train = np.stack([mean_pool(features_dir, cfg["feature_set"], pid) for pid in train_ids])
        y_train = np.array([labels[pid] for pid in train_ids])
        x_val = np.stack([mean_pool(features_dir, cfg["feature_set"], pid) for pid in val_ids])
        signals = compute_oriented_signals(x_train, y_train, x_val, k)

        for pid, sig_row in zip(val_ids, signals):
            # trivial no-LLM control: majority vote of signs, ties broken toward IDC
            baseline_preds[pid] = float(np.mean(sig_row > 0) >= 0.5)

            signal_str = ", ".join(f"{v:+.2f}" for v in sig_row)
            prompt = PROMPT_TEMPLATE.format(k=k, signals=signal_str)
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
            with torch.no_grad():
                out = model.generate(**encoded, max_new_tokens=30, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(out[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            label, conf = parse_response(response)

            if label is not None:
                n_label_parsed += 1
                hard = 1.0 if label == "IDC" else 0.0
                if conf is not None:
                    n_conf_parsed += 1
                    p_idc = (conf / 100.0) if label == "IDC" else (1 - conf / 100.0)
                else:
                    p_idc = hard
            else:
                log.warning(f"  {pid}: could not parse LLM response: {response!r}; defaulting to 0.5")
                p_idc = 0.5
            llm_preds[pid] = p_idc

    log.info(f"Parsed a label for {n_label_parsed}/{len(patients)}, a confidence for {n_conf_parsed}/{len(patients)}")

    def final_metrics(preds: dict[str, float]) -> dict:
        y_true = np.array([labels[p["patient_id"]] for p in patients])
        y_prob = np.array([preds[p["patient_id"]] for p in patients])
        bcfg = cfg["bootstrap"]
        auroc = bootstrap_ci(y_true, y_prob, roc_auc_score, bcfg["n_resamples"], bcfg["ci"], bcfg["random_state"])
        bal = bootstrap_ci(y_true, y_prob, lambda yt, yp: balanced_accuracy_score(yt, (yp >= 0.5).astype(int)),
                            bcfg["n_resamples"], bcfg["ci"], bcfg["random_state"])
        brier = bootstrap_ci(y_true, y_prob, brier_score_loss, bcfg["n_resamples"], bcfg["ci"], bcfg["random_state"])
        return {"auroc": auroc, "balanced_accuracy": bal, "brier": brier}

    llm_metrics = final_metrics(llm_preds)
    baseline_metrics = final_metrics(baseline_preds)

    def fmt(m: dict) -> str:
        a, b, c = m["auroc"], m["balanced_accuracy"], m["brier"]
        return (f"AUROC={a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  "
                f"BalAcc={b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}]  "
                f"Brier={c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]")

    log.info(f"FINAL (LLM)              {fmt(llm_metrics)}")
    log.info(f"FINAL (sign-vote control) {fmt(baseline_metrics)}")

    import wandb
    wandb_run = wandb.init(
        project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], mode=cfg["wandb"]["mode"],
        name=cfg["run_name"], config=cfg, group=cfg["feature_set"], tags=["llm", cfg["feature_set"]],
    )
    wandb_run.summary["final/llm_auroc"] = llm_metrics["auroc"][0]
    wandb_run.summary["final/llm_balanced_accuracy"] = llm_metrics["balanced_accuracy"][0]
    wandb_run.summary["final/llm_brier"] = llm_metrics["brier"][0]
    wandb_run.summary["final/sign_vote_control_auroc"] = baseline_metrics["auroc"][0]
    wandb_run.summary["final/n_label_parsed"] = n_label_parsed
    wandb_run.summary["final/n_conf_parsed"] = n_conf_parsed
    wandb_run.finish()

    header = ["run_name", "feature_set", "classifier_head", "n_patients",
              "auroc", "auroc_ci_lo", "auroc_ci_hi",
              "balanced_accuracy", "balanced_accuracy_ci_lo", "balanced_accuracy_ci_hi",
              "brier", "brier_ci_lo", "brier_ci_hi", "wandb_url"]
    results_csv = Path(__file__).parent / cfg["results_csv"]
    a, b, c = llm_metrics["auroc"], llm_metrics["balanced_accuracy"], llm_metrics["brier"]
    append_csv_row_locked(results_csv, header, [cfg["run_name"], cfg["feature_set"], "llm_zeroshot", len(patients),
                                                 a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], wandb_run.url])
    a, b, c = baseline_metrics["auroc"], baseline_metrics["balanced_accuracy"], baseline_metrics["brier"]
    append_csv_row_locked(results_csv, header, [cfg["run_name"] + "_sign_vote_control", cfg["feature_set"],
                                                 "sign_vote_control", len(patients),
                                                 a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], ""])
    log.info(f"Appended results to {results_csv}")


if __name__ == "__main__":
    main()
