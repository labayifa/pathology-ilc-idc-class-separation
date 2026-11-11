"""Stage B, plan step 1: finalize the real TCGA-BRCA ILC/IDC cohort.

Reproduces the feasibility-gate logic (verification_log.md sec. 1) deterministically
(fixed seed) and writes the exact case/slide IDs to a file, ready for the downloader.

Usage:
    python select_cohort.py --tcga-dir ../../TCGA --out ../cohort/cohort.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

IDC_LABELS = {"Infiltrating duct carcinoma, NOS"}
ILC_LABELS = {"Lobular carcinoma, NOS", "Infiltrating lobular carcinoma, NOS"}
DX_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]+-[A-Z0-9]+)-(\d+)[A-Z]-[A-Z0-9]+-DX\d+\.")
SEED = 42


def classify_patients(clinical_path: Path) -> dict[str, str]:
    with open(clinical_path) as f:
        clinical = json.load(f)

    patient_class: dict[str, str] = {}
    for p in clinical:
        sid = p["submitter_id"]
        diags = p.get("diagnoses", [])
        primary = [d for d in diags if d.get("diagnosis_is_primary_disease") == "true"]
        labels = {d.get("primary_diagnosis") for d in (primary or diags)}
        if labels <= IDC_LABELS and labels:
            patient_class[sid] = "IDC"
        elif labels <= ILC_LABELS and labels:
            patient_class[sid] = "ILC"
        else:
            patient_class[sid] = "OTHER"
    return patient_class


def find_dx_slides(manifest_path: Path) -> dict[str, list[dict]]:
    """Patient ID -> list of {file_id, file_name, md5, size} for Primary Tumor DX slides."""
    patient_dx: dict[str, list[dict]] = defaultdict(list)
    with open(manifest_path) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or ".svs" not in parts[1]:
                continue
            file_id, file_name, md5, size = parts[0], parts[1], parts[2], parts[3]
            m = DX_PATTERN.match(file_name)
            if not m:
                continue
            sample_type_code = m.group(2)
            if sample_type_code != "01":  # 01 = Primary Tumor; drop e.g. 06 = Metastatic
                continue
            patient_dx[m.group(1)].append(
                {"file_id": file_id, "file_name": file_name, "md5": md5, "size": size}
            )
    return patient_dx


def tss(patient_id: str) -> str:
    return patient_id.split("-")[1]


def select_cohort(
    patient_class: dict[str, str], patient_dx: dict[str, list[dict]], target_ilc: int
) -> list[dict]:
    usable = {
        sid: cls for sid, cls in patient_class.items() if cls in ("IDC", "ILC") and sid in patient_dx
    }

    site_classes: dict[str, set[str]] = defaultdict(set)
    for sid, cls in usable.items():
        site_classes[tss(sid)].add(cls)
    both_sites = {s for s, classes in site_classes.items() if {"ILC", "IDC"} <= classes}

    rng = random.Random(SEED)
    ilc_pool = sorted(sid for sid, cls in usable.items() if cls == "ILC" and tss(sid) in both_sites)
    idc_pool = sorted(sid for sid, cls in usable.items() if cls == "IDC" and tss(sid) in both_sites)

    ilc_cohort = ilc_pool if len(ilc_pool) <= target_ilc else rng.sample(ilc_pool, target_ilc)
    ilc_site_counts = Counter(tss(sid) for sid in ilc_cohort)

    idc_by_site: dict[str, list[str]] = defaultdict(list)
    for sid in idc_pool:
        idc_by_site[tss(sid)].append(sid)

    idc_cohort: list[str] = []
    for site, n in ilc_site_counts.items():
        avail = idc_by_site.get(site, [])
        idc_cohort.extend(rng.sample(avail, min(n, len(avail))))
    remaining = len(ilc_cohort) - len(idc_cohort)
    if remaining > 0:
        leftover = sorted(set(idc_pool) - set(idc_cohort))
        idc_cohort.extend(rng.sample(leftover, min(remaining, len(leftover))))

    rows = []
    for sid, cls in [(s, "ILC") for s in ilc_cohort] + [(s, "IDC") for s in idc_cohort]:
        slides = sorted(patient_dx[sid], key=lambda s: s["file_id"])
        chosen = slides[0]  # deterministic: lowest file_id when a patient has >1 DX slide
        rows.append({"patient_id": sid, "dx_class": cls, "tss_site": tss(sid), **chosen})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the TCGA-BRCA ILC/IDC cohort")
    parser.add_argument("--tcga-dir", type=Path, default=Path("../../TCGA"))
    parser.add_argument("--out", type=Path, default=Path("../cohort/cohort.csv"))
    parser.add_argument("--target-ilc", type=int, default=55)
    args = parser.parse_args()

    clinical_path = next(args.tcga_dir.glob("clinical.project-tcga-brca*.json"))
    manifest_path = next(args.tcga_dir.glob("gdc_manifest.*.txt"))

    patient_class = classify_patients(clinical_path)
    patient_dx = find_dx_slides(manifest_path)
    rows = select_cohort(patient_class, patient_dx, args.target_ilc)

    counts = Counter(r["dx_class"] for r in rows)
    sites = len({r["tss_site"] for r in rows})
    print(f"Cohort: {counts['ILC']} ILC, {counts['IDC']} IDC, across {sites} sites")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patient_id", "dx_class", "tss_site", "file_id", "file_name", "md5", "size"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
