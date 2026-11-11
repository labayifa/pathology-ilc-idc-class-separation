"""Cohort expansion (balanced 174 ILC + 174 IDC = 348 patients, vs the
original 55/55=110): downloads, tiles, and extracts features in small
batches, deleting each batch's raw .svs files right after extraction --
275 new slides at ~0.89GB avg is ~283GB, more than fits alongside the OS,
venv, and model caches on a 290GB disk if held all at once, so raw slides
are never kept longer than one batch's worth.

Reuses download_tcga.py and tile_and_extract.py as subprocesses (unchanged)
against a small per-batch cohort CSV, rather than modifying either script --
both are already idempotent, so a crash mid-batch just resumes cleanly on
rerun (this script's own patient-level skip check plus each sub-script's own
file-level skip check cover it twice over).

Usage:
    python stream_cohort_expansion.py \
        --full-cohort ../cohort/cohort_expanded.csv \
        --slide-dir /data/tcga_slides \
        --features-dir /data/features \
        --batch-size 20
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stream_cohort_expansion")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-cohort", type=Path, default=Path("../cohort/cohort_expanded.csv"))
    parser.add_argument("--slide-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--keep-raw", action="store_true", help="don't delete raw .svs after extraction (debug)")
    args = parser.parse_args()

    args.slide_dir.mkdir(parents=True, exist_ok=True)
    args.features_dir.mkdir(parents=True, exist_ok=True)

    with open(args.full_cohort) as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    pending = [r for r in rows if not (args.features_dir / f"{r['patient_id']}.pt").exists()]
    log.info(f"{len(pending)}/{len(rows)} patients still need processing "
             f"({len(rows) - len(pending)} already have features)")

    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        batch = pending[b * args.batch_size: (b + 1) * args.batch_size]
        batch_csv = args.slide_dir / "_batch_cohort.csv"
        with open(batch_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(batch)

        log.info(f"=== batch {b + 1}/{n_batches}: downloading {len(batch)} slides ===")
        subprocess.run(
            [sys.executable, "download_tcga.py", "--cohort", str(batch_csv),
             "--out-dir", str(args.slide_dir), "--concurrency", str(args.concurrency)],
            check=True,
        )

        log.info(f"=== batch {b + 1}/{n_batches}: tiling + extracting ===")
        subprocess.run(
            [sys.executable, "tile_and_extract.py", "--cohort", str(batch_csv),
             "--slide-dir", str(args.slide_dir), "--out-dir", str(args.features_dir)],
            check=True,
        )

        if not args.keep_raw:
            deleted = 0
            for r in batch:
                p = args.slide_dir / r["file_name"]
                if p.exists():
                    p.unlink()
                    deleted += 1
            log.info(f"batch {b + 1}/{n_batches}: deleted {deleted} raw slide(s)")

        batch_csv.unlink(missing_ok=True)

    log.info(f"All {n_batches} batch(es) complete.")


if __name__ == "__main__":
    main()
