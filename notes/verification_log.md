# Pathology Foundation Model Project — Verification Log

Tracks which pre-work items from `markdown/Pathology_Foundation_Model_Project_Plan.md` are confirmed/resolved, so the main plan doesn't need re-reading to know current status.

## 1. TCGA-BRCA feasibility gate (Section 4, Stage B, step 0) — RESOLVED (9 Nov 2026)

Source data: `TCGA/clinical.project-tcga-brca.2026-11-09.json`, `TCGA/biospecimen.project-tcga-brca.2026-11-09.json`, `TCGA/gdc_manifest.2026-11-09.105113.txt`.

- **a. ILC/IDC counts with a usable diagnostic (DX) slide**: classified patients via `primary_diagnosis` + `diagnosis_is_primary_disease` flag (morphology-code-only classification was less reliable — it misclassified patients with a secondary diagnosis entry). Result: 778 pure IDC / 201 pure ILC / 119 other across 1,098 patients (71% ductal matches the external scoping-review figure already cited in the plan — a good sanity check). Cross-referenced against DX slides parsed from the manifest: **191 ILC** and **757 IDC** patients have ≥1 usable diagnostic slide. 31 patients (10 ILC, 21 IDC) have no DX slide and are dropped. 56 patients have 2 DX slides — pick one per the plan's rule.
- **b. Tabulation by tissue source site (TSS)**: 191 usable ILC patients span 31 sites; 20 sites have both ILC and IDC patients. 11 sites are ILC-only (17 patients), 7 are IDC-only (50 patients) — not needed for the cohort.
- **c. Grouped 5-fold split feasibility**: simulated a balanced 55 ILC / 55 IDC cohort drawn only from the 20 both-class sites, then did a greedy site-grouped assignment into 5 folds. **All 5 folds contained both classes and 2–4 distinct sites, no site split across folds.** Gate passes — no need to drop to fewer folds or give up the site-independence claim.
- Minor flag: 1 DX slide is from a Metastatic sample, not Primary Tumor — exclude it explicitly when building the real cohort ID file.

**Conclusion: the hard stop in step 0 is cleared.** Next action is actually building and recording the real cohort ID list (plan step 1) — the 55/55 numbers above are from a feasibility simulation, not the final selected patient list.

## 2. Model licensing and access (Section 3) — RESOLVED (9 Nov 2026)

| Model | License | Gating | Notes |
|---|---|---|---|
| UNI / UNI2-h (Mahmood lab) | CC-BY-NC-ND 4.0 | Gated, institutional email | No redistribution |
| CONCH / CONCH v1.5 (Mahmood lab) | CC-BY-NC-ND 4.0 | Gated, institutional email (personal domains rejected) | Commercial use needs written approval |
| Phikon / Phikon-v2 (Owkin) | Owkin non-commercial academic license | Gating unclear from card | github.com/owkin/HistoSSLscaling/LICENSE.txt |
| H-optimus-0 (Bioptimus) | Tag says Apache-2.0 | Gated (contact-info login required) — tag and gate conflict | Don't assume ungated |
| H-optimus-1 (Bioptimus) | CC-BY-NC-ND 4.0 | Gated, institutional email | |
| Virchow (paige-ai) | Apache 2.0 | Gated (institutional-email attestation, added on top of the license) | |
| Virchow2 (paige-ai) | CC-BY-NC-ND 4.0 (confirmed from model card) | Gated, institutional email | No redistribution, no commercial use, explicit ban on clinical/diagnostic use; ViT-H/14, 632M params, 3.1M WSIs (MSKCC) |
| CTransPath | GPL-3.0 (confirmed) | — | Non-commercial academic use; use official github.com/Xiyue-Wang/TransPath weights, not rehosts |

**Access status**: granted for all models above via CMU institutional email (csagbo@andrew.cmu.edu), non-commercial academic research. This resolves the "UNI2-h access is slow" risk in plan Section 9 — model choice is no longer access-constrained, pick by fit for the task.

## 3. Model technical specs (confirmed from official model cards, 10 Nov 2026)

| Model | Arch | Params | Embed dim | Required input MPP |
|---|---|---|---|---|
| UNI2-h | ViT-H custom, patch14 | 681M | 1536 | **not stated on card** — still open |
| CONCH v1.5 | ViT-L/16 (CoCa, from UNI ckpt) | — | — | not stated |
| Phikon | ViT-Base | 85.8M | — | not stated |
| Phikon-v2 | ViT-L/16 | ~0.3B | 1024 | trained at 20x; exact MPP not stated |
| H-optimus-0 | ViT | 1.1B | 1536 | **0.5 MPP (20x), explicit** |
| H-optimus-1 | ViT | 1.1B | 1536 | **0.5 MPP (20x), explicit** |
| Virchow | ViT-H/14 | 632M | 1280 (2560 concat) | **0.5 MPP (20x), explicit** |
| Virchow2 | ViT-H/14 | 632M | 1280 (2560 concat) | trained multi-scale: 0.25/0.5/1.0/2.0 MPP — flexible |
| CTransPath | Swin (not ViT) | — | 768 (known, not on this card) | not stated; standard ImageNet transform only |

**UNI2-h input MPP — checked 10 Nov 2026, genuinely unspecified**: checked github.com/mahmoodlab/UNI (README, README_old.md, and notebooks/uni_walkthrough.ipynb). None state a magnification or MPP — only pixel size (224x224) and ImageNet normalization. The paper's methods section and CLAM docs were not checked and might have it. Decision: default to 20x / 0.5 MPP (same convention H-optimus and Virchow state explicitly) and **document this as an assumption, not a confirmed spec**, per plan ground rule 2 (state exactly what was done).

**License correction**: H-optimus-0's own model card doesn't state a license at all (just calls itself "open-source"); the earlier Apache-2.0 tag is from the HF page metadata, not the card text, and the HF page is still gated behind contact-info sharing. Treat it the same as the other gated models.

## Open items (not yet verified)

- CPTAC-BRCA as an external validation cohort: need to confirm it actually carries ILC/IDC histologic-type labels (not just PAM50 molecular subtype) and has usable diagnostic slides per class/site, the same way TCGA-BRCA was checked above. Not done yet — do this once CPTAC-BRCA files are available locally.
- GCP GPU quota/pricing/credit balance (plan Section 5) — gcloud auth still broken in this environment, not yet checked.
- Official CTransPath repo (Xiyue-Wang/TransPath) license file itself — checked only via rehosted mirrors.
