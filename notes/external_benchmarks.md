# External Benchmarks for ILC vs IDC Classification (TCGA-BRCA)

Two external papers found (11 Nov 2026) for the same classification task this project runs. Only one is a fair target.

## 1. Ektefaie et al. 2021 — the fair benchmark to beat

Ektefaie Y, Yuan W, Dillon DA, Lin NU, Golden JA, Kohane IS, Yu KH. "Integrative multiomics-histopathology analysis for breast cancer classification." *npj Breast Cancer* 7:147, 2021. doi:10.1038/s41523-021-00357-y. PMID 34845230.

| | |
|---|---|
| Architecture | Inception-v3 CNN, ImageNet-pretrained, fully fine-tuned end-to-end (not frozen features) |
| Input | Top 200 tiles/slide (224×224), ranked by tissue (non-white pixel) density |
| Aggregation | Patient-level prediction via averaging tile predictions ("range voting") |
| Validation | Splits grouped by tissue source site — same safeguard this project's `cv_folds.py` implements |
| Cohort | TCGA-BRCA, ~1099 patients total, ~22% lobular (held-out test split size for this specific task not stated in the paper — wide CI suggests it's small) |
| **Result (TCGA held-out)** | **Patient-level AUC = 0.920** (95% CI 0.845–0.993); tile-level AUC = 0.800 |
| External validation cohort (different task, not the target) | Patient-level AUC = 0.996, tile-level AUC = 0.843 |

**Why this is the fair target**: same dataset (TCGA-BRCA), same task (ILC vs IDC / lobular vs ductal), same patient-level evaluation granularity, and the same site-grouping discipline this project already uses to avoid leakage.

**Final status (12 Nov 2026, full sweep complete)**: `uni2h + MIL` at its pre-specified default configuration reaches patient-level AUROC = 0.917 (95% CI 0.852–0.968), comparable to but not matching or exceeding Ektefaie's 0.920 (95% CI 0.845–0.993) given the overlapping intervals. Across the full hyperparameter sweep, MIL and plain-attention variants on `uni2h` land in 0.90–0.92, using a *frozen* foundation model + lightweight attention head rather than full end-to-end fine-tuning. That contrast (frozen-feature transfer learning comparable to task-specific fine-tuning) is the headline comparison for this project — framed as "comparable to," not "matches," per the paper's own confidence intervals.

## 2. Zhang 2025 — not a fair target, do not chase this number

Zhang M. "Hospital-Specific Bias in Patch-Based Pathology Models." arXiv:2508.14779 (ISBI 2026 submission). Code: github.com/MengRes/pfm_domain_bias

| | |
|---|---|
| Architecture | Frozen UNI features → 2-layer MLP (512, 256 units) baseline |
| Input | 4,029 patches, pre-filtered by CONCH zero-shot to keep only high-confidence (>0.8) "easy" patches |
| Cohort | Only 80 WSIs, 4 hospitals (the 4 with the most TCGA-BRCA WSIs), restricted to White female patients aged 60–79 |
| Evaluation | **Patch-level**, not patient-level; 5-fold CV grouped by WSI, not by hospital (so not a cross-site test) |
| **Result** | Disease accuracy 0.93±0.03, disease AUC **0.98±0.01** (UNI + MLP baseline) |

**Why this isn't comparable**: three things inflate this number relative to a real patient-level task — (a) patch-level evaluation is easier than patient-level (many redundant easy patches per patient), (b) CONCH-prefiltering removes ambiguous/hard patches before training even starts, (c) the cohort is small (80 WSIs) and demographically narrow (one race/sex/age band). The paper's own stated purpose is studying hospital-source bias in frozen features (relevant background, not a result to beat) — it found UNI's hospital-source AUC (how predictable the scanning hospital is from frozen features) was 0.95–0.96 at baseline, dropping to 0.58 after their adversarial debiasing adaptor, with disease AUC "largely maintained" (0.98 → 0.96).

## Open question this raises

Zhang 2025's finding that hospital/site is strongly encoded in frozen UNI features (AUC ~0.95) is a relevant cross-check for this project's own site-grouped CV discipline — it's indirect evidence that *not* grouping by site (or by tissue source site, our proxy for it) could inflate apparent performance via site leakage. Worth keeping in mind if any future ablation ever relaxes the site-grouping constraint.
