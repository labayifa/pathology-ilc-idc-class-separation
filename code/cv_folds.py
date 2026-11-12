"""Plan step 5 (Design decision 5): deterministic site-grouped stratified fold
assignment. Ports the greedy algorithm validated in the earlier feasibility-gate
simulation (sort sites by size, assign each whole site to the fold with the
smallest current patient count) -- never splits a tissue-source site across
folds, since slides from the same site can share batch effects (scanner,
staining) that would leak information between train/val if split.

Several seeds are produced (not just one) because downstream training repeats
across seeds and averages each patient's out-of-fold prediction before
bootstrapping (plan design decision 5) -- repeated predictions from the same
fold assignment aren't independent samples, but predictions from *different*
fold assignments are a reasonable way to get repeats.

Usage:
    python cv_folds.py \
        --cohort ../cohort/cohort.csv \
        --features-dir "/path/to/features" \
        --out ../cohort/folds.csv \
        --n-folds 5 --n-seeds 5
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cv_folds")


def assign_folds(patients: list[dict], n_folds: int, seed: int) -> dict[str, int]:
    """Greedy site-to-fold bin packing. Returns {patient_id: fold_idx}."""
    sites: dict[str, list[dict]] = defaultdict(list)
    for p in patients:
        sites[p["tss_site"]].append(p)

    site_names = list(sites.keys())
    rng = random.Random(seed)
    rng.shuffle(site_names)  # randomizes tie-break order among equal-size sites
    site_names.sort(key=lambda s: -len(sites[s]))  # stable sort preserves shuffled tie order

    fold_counts = [0] * n_folds
    fold_sites: list[list[str]] = [[] for _ in range(n_folds)]
    assignment: dict[str, int] = {}

    for site in site_names:
        fold_idx = min(range(n_folds), key=lambda i: (fold_counts[i], i))
        fold_sites[fold_idx].append(site)
        fold_counts[fold_idx] += len(sites[site])
        for p in sites[site]:
            assignment[p["patient_id"]] = fold_idx

    verify_folds(patients, assignment, fold_sites, n_folds)
    return assignment


def verify_folds(patients: list[dict], assignment: dict[str, int], fold_sites: list[list[str]], n_folds: int) -> None:
    by_patient = {p["patient_id"]: p for p in patients}

    seen_sites: set[str] = set()
    for sites in fold_sites:
        for s in sites:
            assert s not in seen_sites, f"site {s} assigned to more than one fold"
            seen_sites.add(s)

    for fold_idx in range(n_folds):
        fold_patient_ids = [pid for pid, f in assignment.items() if f == fold_idx]
        classes = {by_patient[pid]["dx_class"] for pid in fold_patient_ids}
        assert classes == {"ILC", "IDC"}, f"fold {fold_idx} missing a class: has {classes}"
        assert len(fold_sites[fold_idx]) >= 2, f"fold {fold_idx} has only {len(fold_sites[fold_idx])} site(s)"

    assert len(assignment) == len(patients), "not every patient got a fold"


def main() -> None:
    parser = argparse.ArgumentParser(description="Site-grouped stratified fold assignment")
    parser.add_argument("--cohort", type=Path, default=Path("../cohort/cohort.csv"))
    parser.add_argument("--features-dir", type=Path, required=True,
                         help="Only patients with a matching {patient_id}.pt here are included")
    parser.add_argument("--out", type=Path, default=Path("../cohort/folds.csv"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    with open(args.cohort) as f:
        rows = list(csv.DictReader(f))

    have_features = {p.stem for p in args.features_dir.glob("*.pt") if not p.name.startswith(".")}
    patients = [r for r in rows if r["patient_id"] in have_features]
    missing = [r["patient_id"] for r in rows if r["patient_id"] not in have_features]
    if missing:
        log.warning(f"{len(missing)} cohort patient(s) have no feature file, excluding: {missing}")
    log.info(f"{len(patients)} patients usable, across {len({p['tss_site'] for p in patients})} sites")

    seed_assignments = {}
    for seed in range(args.n_seeds):
        assignment = assign_folds(patients, args.n_folds, seed)
        seed_assignments[seed] = assignment
        counts = defaultdict(int)
        for fold_idx in assignment.values():
            counts[fold_idx] += 1
        log.info(f"seed {seed}: fold sizes {dict(sorted(counts.items()))} -- verified OK")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        fieldnames = ["patient_id", "dx_class", "tss_site"] + [f"fold_seed{s}" for s in range(args.n_seeds)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in patients:
            row = {"patient_id": p["patient_id"], "dx_class": p["dx_class"], "tss_site": p["tss_site"]}
            for seed in range(args.n_seeds):
                row[f"fold_seed{seed}"] = seed_assignments[seed][p["patient_id"]]
            writer.writerow(row)
    log.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
