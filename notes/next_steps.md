# Next Steps Checklist

Status: licensing/access and TCGA-BRCA feasibility gate done (see `verification_log.md`). Today is Day 3 of 8. Day 2 smoke test still pending.

- [ ] 1. GPU/GCP check — confirm GPU quota, pricing, credit balance.
- [ ] 2. BACH smoke test — download BACH training zip only (10.4GB, from https://zenodo.org/records/3632035 — skip the 3GB test zip, not needed for a smoke test). CC BY-NC-ND 4.0, cite Aresta et al. 2019 if ever written up. Run full pipeline on UNI2-h end-to-end (tile → features → mean-pool → LR → metrics). Just needs to run, not produce a result.
- [ ] 3. Finalize real TCGA-BRCA cohort — pick ~55 ILC / ~55 IDC from the 20 both-class sites, one DX slide per patient (drop the 1 metastatic DX slide; for the 56 patients with 2 DX slides, pick the lowest slide UUID). Write case/slide IDs to a file in `Pathology/`.
- [ ] 4. Download the ~110 selected DX slides only — async queue + semaphore, retries, timeouts.
- [ ] 5. Tile + resample — read MPP per slide, resample to UNI2-h's target MPP (verify on model card first), 224px tiles, cap ~1,000/slide.
- [ ] 6. Extract features — UNI2-h embeddings + small ImageNet-CNN baseline embeddings.
- [ ] 7. Train + evaluate — mean-pool+LR baseline vs. UNI2-h mean-pool+LR vs. UNI2-h+MIL, site-grouped stratified 5-fold CV. Report AUROC, balanced accuracy, Brier score, reliability diagram, 95% CI (patient-level bootstrap).
- [ ] 8. Error analysis — misclassified slides, attention maps (if MIL), confidently-wrong cases.
- [ ] 9. Deliverables — public repo (code, config, cohort IDs, results, README; no slides/features/weights), one-page write-up, updated letter/CV text.

## Verification per step
- Step 2: smoke test prints a metrics output with no errors.
- Step 3: cohort file has ~110 unique patient IDs, one DX slide each, balanced ILC/IDC, all from both-class sites.
- Step 7: every CV fold has both classes and ≥2 sites (matches the feasibility simulation).
