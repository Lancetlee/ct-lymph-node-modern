# Roadmap

This is a community research roadmap, not a promise of release dates or clinical performance. The current baseline has synthetic CPU tests and a new PyTorch segmentation architecture. Real Apple MPS/CUDA hardware results, real PyRadiomics integration, legacy-weight equivalence, and clinical model accuracy require separate evidence.

## 1. Reproducible accelerator validation

Contributors with Apple Silicon, including M5 Max, and NVIDIA GPUs are welcome to submit hardware reports.

- Run the device probe, CPU/backend numerical comparison, and synchronized benchmark with complete configuration and software versions.
- Verify the complete trained-model forward/backward and full-volume inference path on target hardware; the small probe alone is insufficient.
- Compare memory and latency across patch sizes and batch sizes using the same model and precision. Distinguish current tensor allocation from measured peak system memory.
- Propose optimizations only with numerical and timing comparisons. Track MPS operator failures without hidden CPU fallback.

Acceptance evidence: reproducible commands, sanitized JSON reports, numerical differences, and clearly scoped timing. MPS GPU support is not an ANE implementation or proof of M5-specific kernels.

## 2. Reliable optional radiomics installation

- Reproduce and resolve modern Python/Apple arm64 build problems with the optional PyRadiomics dependency.
- Evaluate a maintained upstream version or extraction backend in an isolated environment; document feature definitions and licensing before changing defaults.
- Run the real extraction integration tests without skips, then compare reference objects and fixed parameters. Keep mocked adapter tests separate.

Acceptance evidence: reproducible environment setup, actual feature output, parameter/version provenance, and numerical reference checks. A successful import alone is insufficient.

## 3. Segmentation training and evaluation

- Improve training controls and evaluation for small nodes, connected/fused nodes, and difficult anatomical regions.
- Add patient-isolated independent evaluation, detection sensitivity/false positives, Dice, and size/station subgroup reporting using datasets that permit the intended use.
- Document data provenance, annotation completeness, split construction, model cards, and any model-weight redistribution terms.

Acceptance evidence: reproducible evaluation on appropriately held-out data. Do not describe synthetic test checkpoints as clinical models or reuse development-validation scores as independent test performance.

## 4. Optional faithful legacy-model migration

The new Attention U-Net is not a converted TensorFlow model. A separate legacy migration effort could inventory the old architecture and custom layers, obtain weights under their applicable terms, and compare preprocessing, intermediate layers, and final outputs on identical inputs.

Acceptance evidence: weight/source hashes, a documented reference environment, numerical equivalence results, and independent segmentation evaluation. Renaming an H5 file or loading selected tensors is not a faithful conversion.

## 5. Annotation and research workflow

- Improve reviewed-mask exchange with common medical imaging tools while preserving integer labels and physical geometry.
- Refine candidate splitting/merging and the mapping between node identity, IASLC station, scan, and pathology reference standard.
- Invite radiologists and research teams to review anatomy and annotation guidance, supported by attributable sources and explicit uncertainty.
- If a later metastasis prediction module is proposed, keep its task, labels, patient splits, leakage controls, calibration, and validation separate from segmentation.

Use [CONTRIBUTING.md](CONTRIBUTING.md) and the issue forms to propose work or share a reproducible result. No patient data should be attached to public reports.
