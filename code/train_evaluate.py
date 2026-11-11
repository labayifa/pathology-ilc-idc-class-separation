"""Plan step 6: train + evaluate all classifier-head variants on frozen tile
features, across 5-fold site-grouped CV repeated over several seeds (folds
from cv_folds.py). Each patient's final prediction is the average of its
out-of-fold probability across seeds (plan design decision 5 -- repeated
predictions from the same fold assignment aren't independent, but different
seeds' fold assignments give a reasonable repeat), then AUROC / balanced
accuracy / Brier score / reliability diagram are computed once on that
averaged prediction, with a patient-level bootstrap for 95% CIs.

Five classifier heads share one harness via a YAML config (--config):
  lr / svm   -- mean-pooled slide feature -> scikit-learn classifier
  mlp        -- mean-pooled slide feature -> small feedforward net
  mil        -- gated-attention MIL over all tiles (no pooling first)
  gat        -- graph attention over the tile spatial k-NN graph

Usage:
    python train_evaluate.py --config configs/uni2h_mil.yaml
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import experiment_db as edb
from models.attention_head import AttentionPooling
from models.gat_head import GATClassifier, GATThenMIL, GATThenMLP
from models.mil_head import GatedAttentionMIL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_evaluate")

LABEL_MAP = {"ILC": 0, "IDC": 1}


def append_csv_row_locked(path: Path, header: list[str], row: list) -> None:
    """Appends one row to a shared results CSV under an exclusive file lock
    -- several configs run concurrently on the same GPU all write to this
    one file, and an unlocked append can interleave rows or duplicate the
    header (two processes both see the file as empty-and-header-less at the
    same instant)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.tell() == 0:
                csv.writer(f).writerow(header)
            csv.writer(f).writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        base_path = path.parent / cfg.pop("defaults")
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_cohort(cfg: dict) -> tuple[list[dict], int]:
    with open(Path(__file__).parent / cfg["folds_csv"]) as f:
        rows = list(csv.DictReader(f))
    n_seeds = len(cfg["seeds"])
    return rows, n_seeds


class FeatureCache:
    """Loads each patient's {feature_set} tensor (and tile_coords, for GAT) once."""

    def __init__(self, features_dir: Path, feature_set: str, device: str):
        self.features_dir = features_dir
        self.feature_set = feature_set
        self.device = device
        self._cache: dict[str, dict] = {}

    def get(self, patient_id: str) -> dict:
        if patient_id not in self._cache:
            data = torch.load(self.features_dir / f"{patient_id}.pt", weights_only=False)
            self._cache[patient_id] = {
                "features": data["features"][self.feature_set].to(self.device),
                "tile_coords": data["tile_coords"].to(self.device),
            }
        return self._cache[patient_id]


# --------------------------------------------------------------------------
# Classifier heads
# --------------------------------------------------------------------------

def build_optimizer(params, tcfg: dict) -> torch.optim.Optimizer:
    opt_type = tcfg.get("optimizer", "adam")
    if opt_type == "adam":
        return torch.optim.Adam(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    if opt_type == "sgd":
        return torch.optim.SGD(params, lr=tcfg["lr"], momentum=tcfg.get("momentum", 0.9),
                                weight_decay=tcfg["weight_decay"])
    raise ValueError(f"unknown optimizer: {opt_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, tcfg: dict):
    sched_type = tcfg.get("scheduler", "none")
    if sched_type == "none":
        return None
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    if sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=tcfg.get("step_size", 20),
                                                gamma=tcfg.get("step_gamma", 0.5))
    raise ValueError(f"unknown scheduler: {sched_type}")


def atomic_torch_save(obj: dict, path: Path) -> None:
    """Writes to a temp file then renames over the target -- a Spot-instance
    reclaim mid-write must never leave a half-written checkpoint in place of
    a good one, since that checkpoint is what resuming loads from."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def finalize_checkpoints(latest_path: Path, best_path: Path) -> None:
    """Once a fold's training is fully done, `latest` (model + optimizer +
    scheduler state, kept only for resuming) is no longer needed -- only
    `best` (model weights only, needed later for e.g. error analysis)
    matters going forward. Keeps `latest` if `best` was never written (e.g.
    a degenerate fold whose val AUROC was always NaN), so a finished fold
    is never left with zero usable checkpoints."""
    if best_path.exists() and latest_path.exists():
        latest_path.unlink()


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mean_pool(feat_cache: FeatureCache, patient_id: str) -> torch.Tensor:
    return feat_cache.get(patient_id)["features"].mean(dim=0)


def run_sklearn_fold(cfg: dict, feat_cache: FeatureCache, train_ids: list[str], val_ids: list[str],
                      labels: dict[str, int]) -> dict[str, float]:
    x_train = np.stack([mean_pool(feat_cache, pid).cpu().numpy() for pid in train_ids])
    y_train = np.array([labels[pid] for pid in train_ids])
    x_val = np.stack([mean_pool(feat_cache, pid).cpu().numpy() for pid in val_ids])

    if cfg["classifier_head"] == "lr":
        clf = make_pipeline(StandardScaler(), LogisticRegression(**cfg["lr_params"]))
    else:
        # probability=True on SVC is deprecated (removed in sklearn 1.11); CalibratedClassifierCV
        # is the forward-compatible way to get predict_proba from an SVC.
        svm_params = {k: v for k, v in cfg["svm_params"].items() if k != "probability"}
        clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(**svm_params), ensemble=False))
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_val)[:, 1]
    return {pid: float(p) for pid, p in zip(val_ids, probs)}


def run_gat_sklearn_fold(cfg: dict, feat_cache: FeatureCache, train_ids: list[str], val_ids: list[str],
                          labels: dict[str, int], device: str, wandb_run, seed: int, fold: int,
                          conn, experiment_id: int, checkpoint_dir: Path) -> dict[str, float]:
    """gat_lr / gat_svm: LR and SVM aren't differentiable, so the GAT backbone
    can't be trained end-to-end with them directly. Instead train a GATThenMLP
    proxy (same backbone, a throwaway differentiable linear-ish head) via
    gradient descent, then discard the proxy's own head and fit sklearn LR/SVM
    on the trained backbone's mean-pooled refined embeddings -- the reported
    predictions come from the properly regularized/calibrated sklearn model,
    the proxy head only exists to supply a training signal for the backbone."""
    tcfg = cfg["train"]
    proxy = GATThenMLP(cfg["feature_dim"], tcfg["hidden_dim"], cfg["gat"]["n_heads"], cfg["gat"]["n_layers"],
                        cfg["gat"]["k_neighbors"], tcfg["dropout"]).to(device)
    optimizer = build_optimizer(proxy.parameters(), tcfg)
    scheduler = build_scheduler(optimizer, tcfg)
    rng = random.Random(seed * 1000 + fold)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / f"seed{seed}_fold{fold}_latest.pt"
    best_path = checkpoint_dir / f"seed{seed}_fold{fold}_best.pt"

    start_epoch = 0
    best_val_auroc = -1.0
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        proxy.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auroc = ckpt["best_val_auroc"]
        log.info(f"  seed{seed} fold{fold}: resuming GAT proxy from epoch {start_epoch}")

    for epoch in range(start_epoch, tcfg["epochs"]):
        proxy.train()
        order = list(train_ids)
        rng.shuffle(order)
        total_loss = 0.0
        for pid in order:
            entry = feat_cache.get(pid)
            optimizer.zero_grad()
            logits = proxy(entry["features"], entry["tile_coords"])
            loss = F.cross_entropy(logits, torch.tensor([labels[pid]], device=device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        train_loss = total_loss / len(order)

        proxy.eval()
        val_loss_total = 0.0
        val_probs = {}
        with torch.no_grad():
            for pid in val_ids:
                entry = feat_cache.get(pid)
                logits = proxy(entry["features"], entry["tile_coords"])
                val_loss_total += F.cross_entropy(logits, torch.tensor([labels[pid]], device=device)).item()
                val_probs[pid] = F.softmax(logits, dim=-1)[0, 1].item()
        val_loss = val_loss_total / len(val_ids)
        val_y = np.array([labels[pid] for pid in val_ids])
        val_p = np.array([val_probs[pid] for pid in val_ids])
        val_auroc = roc_auc_score(val_y, val_p) if len(set(val_y)) > 1 else float("nan")
        val_bal_acc = balanced_accuracy_score(val_y, (val_p >= 0.5).astype(int))
        val_brier = brier_score_loss(val_y, val_p)

        if wandb_run is not None:
            wandb_run.log({
                f"train/seed{seed}_fold{fold}/gat_proxy_loss": train_loss,
                f"train/seed{seed}_fold{fold}/gat_proxy_lr": current_lr,
                f"val/seed{seed}_fold{fold}/gat_proxy_loss": val_loss,
                f"val/seed{seed}_fold{fold}/gat_proxy_auroc": val_auroc,
                f"val/seed{seed}_fold{fold}/gat_proxy_balanced_accuracy": val_bal_acc,
                f"val/seed{seed}_fold{fold}/gat_proxy_brier": val_brier,
            })
        edb.log_step(conn, experiment_id, seed, fold, epoch, train_loss,
                     None if np.isnan(val_auroc) else val_auroc)

        atomic_torch_save({
            "model_state": proxy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_auroc": best_val_auroc,
        }, latest_path)
        edb.upsert_checkpoint(conn, experiment_id, seed, fold, "latest", str(latest_path), val_auroc, epoch)

        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            atomic_torch_save({"model_state": proxy.state_dict(), "epoch": epoch,
                                "val_auroc": val_auroc}, best_path)
            edb.upsert_checkpoint(conn, experiment_id, seed, fold, "best", str(best_path), val_auroc, epoch)
        conn.commit()  # per-epoch: a hard kill loses at most the in-progress epoch, not the whole fold
    finalize_checkpoints(latest_path, best_path)

    proxy.eval()
    with torch.no_grad():
        def pooled(pid: str) -> np.ndarray:
            entry = feat_cache.get(pid)
            return proxy.pooled_embedding(entry["features"], entry["tile_coords"]).cpu().numpy()

        x_train = np.stack([pooled(pid) for pid in train_ids])
        x_val = np.stack([pooled(pid) for pid in val_ids])
    y_train = np.array([labels[pid] for pid in train_ids])

    if cfg["classifier_head"] == "gat_lr":
        clf = make_pipeline(StandardScaler(), LogisticRegression(**cfg["lr_params"]))
    else:
        svm_params = {k: v for k, v in cfg["svm_params"].items() if k != "probability"}
        clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(**svm_params), ensemble=False))
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_val)[:, 1]
    return {pid: float(p) for pid, p in zip(val_ids, probs)}


def run_torch_fold(cfg: dict, feat_cache: FeatureCache, train_ids: list[str], val_ids: list[str],
                    labels: dict[str, int], device: str, wandb_run, seed: int, fold: int,
                    conn, experiment_id: int, checkpoint_dir: Path) -> dict[str, float]:
    tcfg = cfg["train"]
    head = cfg["classifier_head"]

    if head == "mlp":
        model = MLP(cfg["feature_dim"], tcfg["hidden_dim"], tcfg["dropout"]).to(device)
    elif head == "mil":
        model = GatedAttentionMIL(cfg["feature_dim"], tcfg["hidden_dim"],
                                   cfg["mil"]["attention_dim"], dropout=tcfg["dropout"]).to(device)
    elif head == "attention":
        model = AttentionPooling(cfg["feature_dim"], tcfg["hidden_dim"],
                                  cfg["attention"]["attention_dim"], dropout=tcfg["dropout"]).to(device)
    elif head == "gat":
        model = GATClassifier(cfg["feature_dim"], tcfg["hidden_dim"], cfg["gat"]["n_heads"],
                               cfg["gat"]["n_layers"], cfg["gat"]["k_neighbors"], tcfg["dropout"]).to(device)
    elif head == "gat_mlp":
        model = GATThenMLP(cfg["feature_dim"], tcfg["hidden_dim"], cfg["gat"]["n_heads"], cfg["gat"]["n_layers"],
                            cfg["gat"]["k_neighbors"], tcfg["dropout"]).to(device)
    elif head == "gat_mil":
        model = GATThenMIL(cfg["feature_dim"], tcfg["hidden_dim"], cfg["gat"]["n_heads"], cfg["gat"]["n_layers"],
                            cfg["gat"]["k_neighbors"], tcfg["dropout"], cfg["mil"]["attention_dim"]).to(device)
    else:
        raise ValueError(f"unknown classifier_head: {head}")

    optimizer = build_optimizer(model.parameters(), tcfg)
    scheduler = build_scheduler(optimizer, tcfg)
    rng = random.Random(seed * 1000 + fold)

    def forward_one(pid: str) -> torch.Tensor:
        entry = feat_cache.get(pid)
        if head == "mlp":
            return model(entry["features"].mean(dim=0).unsqueeze(0))
        if head in ("mil", "attention"):
            logits, _ = model(entry["features"])
            return logits
        if head == "gat_mlp":
            return model(entry["features"], entry["tile_coords"])
        logits, _ = model(entry["features"], entry["tile_coords"])  # gat, gat_mil
        return logits

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / f"seed{seed}_fold{fold}_latest.pt"
    best_path = checkpoint_dir / f"seed{seed}_fold{fold}_best.pt"

    start_epoch = 0
    best_val_auroc = -1.0
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auroc = ckpt["best_val_auroc"]
        log.info(f"  seed{seed} fold{fold}: resuming from epoch {start_epoch} (latest checkpoint found)")

    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        order = list(train_ids)
        rng.shuffle(order)
        total_loss = 0.0
        for pid in order:
            optimizer.zero_grad()
            logits = forward_one(pid)
            loss = F.cross_entropy(logits, torch.tensor([labels[pid]], device=device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        train_loss = total_loss / len(order)

        model.eval()
        val_loss_total = 0.0
        val_probs = {}
        with torch.no_grad():
            for pid in val_ids:
                logits = forward_one(pid)
                val_loss_total += F.cross_entropy(logits, torch.tensor([labels[pid]], device=device)).item()
                val_probs[pid] = F.softmax(logits, dim=-1)[0, 1].item()
        val_loss = val_loss_total / len(val_ids)
        val_y = np.array([labels[pid] for pid in val_ids])
        val_p = np.array([val_probs[pid] for pid in val_ids])
        val_auroc = roc_auc_score(val_y, val_p) if len(set(val_y)) > 1 else float("nan")
        val_bal_acc = balanced_accuracy_score(val_y, (val_p >= 0.5).astype(int))
        val_brier = brier_score_loss(val_y, val_p)

        if wandb_run is not None:
            wandb_run.log({
                f"train/seed{seed}_fold{fold}/loss": train_loss,
                f"train/seed{seed}_fold{fold}/lr": current_lr,
                f"val/seed{seed}_fold{fold}/loss": val_loss,
                f"val/seed{seed}_fold{fold}/auroc": val_auroc,
                f"val/seed{seed}_fold{fold}/balanced_accuracy": val_bal_acc,
                f"val/seed{seed}_fold{fold}/brier": val_brier,
            })
        edb.log_step(conn, experiment_id, seed, fold, epoch, train_loss,
                     None if np.isnan(val_auroc) else val_auroc)

        atomic_torch_save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_auroc": best_val_auroc,
        }, latest_path)
        edb.upsert_checkpoint(conn, experiment_id, seed, fold, "latest", str(latest_path), val_auroc, epoch)

        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            atomic_torch_save({"model_state": model.state_dict(), "epoch": epoch,
                                "val_auroc": val_auroc}, best_path)
            edb.upsert_checkpoint(conn, experiment_id, seed, fold, "best", str(best_path), val_auroc, epoch)
        conn.commit()  # per-epoch: a hard kill loses at most the in-progress epoch, not the whole fold
    finalize_checkpoints(latest_path, best_path)

    model.eval()
    with torch.no_grad():
        return {pid: F.softmax(forward_one(pid), dim=-1)[0, 1].item() for pid in val_ids}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_resamples: int, ci: float,
                  random_state: int) -> tuple[float, float, float]:
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(set(yt)) < 2:
            continue  # metric undefined (e.g. AUROC) for a single-class resample; skip, don't fabricate
        try:
            stats.append(metric_fn(yt, yp))
        except ValueError:
            continue
    stats = np.array(stats)
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return float(stats.mean()), float(lo), float(hi)


def compute_final_metrics(patients: list[dict], avg_pred: dict[str, float], cfg: dict) -> dict:
    y_true = np.array([LABEL_MAP[p["dx_class"]] for p in patients])
    y_prob = np.array([avg_pred[p["patient_id"]] for p in patients])
    y_pred = (y_prob >= 0.5).astype(int)

    bcfg = cfg["bootstrap"]
    auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(
        y_true, y_prob, roc_auc_score, bcfg["n_resamples"], bcfg["ci"], bcfg["random_state"])
    bal_acc_mean, bal_acc_lo, bal_acc_hi = bootstrap_ci(
        y_true, y_prob, lambda yt, yp: balanced_accuracy_score(yt, (yp >= 0.5).astype(int)),
        bcfg["n_resamples"], bcfg["ci"], bcfg["random_state"])
    brier_mean, brier_lo, brier_hi = bootstrap_ci(
        y_true, y_prob, brier_score_loss, bcfg["n_resamples"], bcfg["ci"], bcfg["random_state"])

    return {
        "y_true": y_true, "y_prob": y_prob, "y_pred": y_pred,
        "auroc": (auroc_mean, auroc_lo, auroc_hi),
        "balanced_accuracy": (bal_acc_mean, bal_acc_lo, bal_acc_hi),
        "brier": (brier_mean, brier_lo, brier_hi),
    }


def log_plots_to_wandb(wandb_run, wandb_module, metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, y_prob = metrics["y_true"], metrics["y_prob"]
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    wandb_run.log({"reliability_diagram": wandb_module.Image(fig)})
    plt.close(fig)

    wandb_run.log({
        "roc_curve": wandb_module.plot.roc_curve(y_true, np.stack([1 - y_prob, y_prob], axis=1),
                                                   labels=["ILC", "IDC"]),
        "confusion_matrix": wandb_module.plot.confusion_matrix(
            y_true=y_true.tolist(), preds=metrics["y_pred"].tolist(), class_names=["ILC", "IDC"]),
    })


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wandb-mode", type=str, default=None, help="override config's wandb.mode (e.g. disabled)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.wandb_mode:
        cfg["wandb"]["mode"] = args.wandb_mode

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Config: {cfg['run_name']}  device: {device}")

    patients, n_seeds = load_cohort(cfg)
    labels = {p["patient_id"]: LABEL_MAP[p["dx_class"]] for p in patients}
    features_dir = Path(cfg["features_dir"])
    feat_cache = FeatureCache(features_dir, cfg["feature_set"], device)

    import wandb
    wandb_run = wandb.init(
        project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], mode=cfg["wandb"]["mode"],
        name=cfg["run_name"], config=cfg, group=cfg["feature_set"], tags=[cfg["classifier_head"], cfg["feature_set"]],
    )

    per_patient_preds: dict[str, list[float]] = {p["patient_id"]: [] for p in patients}
    db_path = Path(__file__).parent / cfg.get("db_path", "../results/experiments.db")
    checkpoint_dir = Path(__file__).parent / cfg.get("checkpoint_dir", "../results/checkpoints") / cfg["run_name"]

    with edb.connect(db_path) as conn:
        experiment_id = edb.get_or_create_experiment(
            conn, cfg["run_name"], cfg["feature_set"], cfg["classifier_head"], cfg)

        for seed in cfg["seeds"]:
            fold_key = f"fold_seed{seed}"
            for fold in range(cfg["n_folds"]):
                train_ids = [p["patient_id"] for p in patients if int(p[fold_key]) != fold]
                val_ids = [p["patient_id"] for p in patients if int(p[fold_key]) == fold]
                log.info(f"seed {seed} fold {fold}: {len(train_ids)} train / {len(val_ids)} val")

                if cfg["classifier_head"] in ("lr", "svm"):
                    val_preds = run_sklearn_fold(cfg, feat_cache, train_ids, val_ids, labels)
                elif cfg["classifier_head"] in ("gat_lr", "gat_svm"):
                    val_preds = run_gat_sklearn_fold(cfg, feat_cache, train_ids, val_ids, labels, device,
                                                      wandb_run, seed, fold, conn, experiment_id, checkpoint_dir)
                else:
                    val_preds = run_torch_fold(cfg, feat_cache, train_ids, val_ids, labels, device,
                                                wandb_run, seed, fold, conn, experiment_id, checkpoint_dir)
                conn.commit()

                y_val = np.array([labels[pid] for pid in val_ids])
                p_val = np.array([val_preds[pid] for pid in val_ids])
                fold_auroc = roc_auc_score(y_val, p_val) if len(set(y_val)) > 1 else float("nan")
                wandb_run.log({f"fold_auroc/seed{seed}_fold{fold}": fold_auroc})

                for pid, p in val_preds.items():
                    per_patient_preds[pid].append(p)

        avg_pred = {pid: float(np.mean(preds)) for pid, preds in per_patient_preds.items()}
        metrics = compute_final_metrics(patients, avg_pred, cfg)

        # Per-patient OOF predictions, saved so configs can be combined into
        # ensembles later (across heads or feature sets) without retraining --
        # the aggregate metrics above are not enough for that on their own.
        preds_dir = Path(__file__).parent / cfg.get("predictions_dir", "../results/predictions")
        preds_dir.mkdir(parents=True, exist_ok=True)
        with open(preds_dir / f"{cfg['run_name']}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_id", "dx_class", "pred_prob_idc"])
            for p in patients:
                writer.writerow([p["patient_id"], p["dx_class"], avg_pred[p["patient_id"]]])

        auroc_mean, auroc_lo, auroc_hi = metrics["auroc"]
        bal_mean, bal_lo, bal_hi = metrics["balanced_accuracy"]
        brier_mean, brier_lo, brier_hi = metrics["brier"]
        log.info(f"FINAL  AUROC={auroc_mean:.3f} [{auroc_lo:.3f},{auroc_hi:.3f}]  "
                 f"BalAcc={bal_mean:.3f} [{bal_lo:.3f},{bal_hi:.3f}]  "
                 f"Brier={brier_mean:.3f} [{brier_lo:.3f},{brier_hi:.3f}]")

        wandb_run.summary["final/auroc"] = auroc_mean
        wandb_run.summary["final/auroc_ci_lo"] = auroc_lo
        wandb_run.summary["final/auroc_ci_hi"] = auroc_hi
        wandb_run.summary["final/balanced_accuracy"] = bal_mean
        wandb_run.summary["final/balanced_accuracy_ci_lo"] = bal_lo
        wandb_run.summary["final/balanced_accuracy_ci_hi"] = bal_hi
        wandb_run.summary["final/brier"] = brier_mean
        wandb_run.summary["final/brier_ci_lo"] = brier_lo
        wandb_run.summary["final/brier_ci_hi"] = brier_hi

        try:
            log_plots_to_wandb(wandb_run, wandb, metrics)
        except Exception as exc:
            log.warning(f"could not log plots to wandb: {exc}")

        edb.record_final_result(conn, experiment_id, metrics, wandb_run.url)

        results_csv = Path(__file__).parent / cfg["results_csv"]
        append_csv_row_locked(
            results_csv,
            ["run_name", "feature_set", "classifier_head", "n_patients",
             "auroc", "auroc_ci_lo", "auroc_ci_hi",
             "balanced_accuracy", "balanced_accuracy_ci_lo", "balanced_accuracy_ci_hi",
             "brier", "brier_ci_lo", "brier_ci_hi", "wandb_url"],
            [cfg["run_name"], cfg["feature_set"], cfg["classifier_head"], len(patients),
             auroc_mean, auroc_lo, auroc_hi, bal_mean, bal_lo, bal_hi,
             brier_mean, brier_lo, brier_hi, wandb_run.url],
        )
        log.info(f"Appended results to {results_csv}")

    wandb_run.finish()


if __name__ == "__main__":
    main()
