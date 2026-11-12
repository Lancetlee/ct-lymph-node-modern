# Contributing

Contributions in English or Chinese are welcome. This project provides research tooling for thoracic lymph-node segmentation, reviewed ROI annotation, and optional radiomics extraction. It does not provide a validated metastasis classifier or a clinically validated checkpoint.

See [ROADMAP.md](ROADMAP.md) for useful starting points. Hardware reports, reproducible bug reports, documentation corrections, and small focused pull requests are all useful contributions.

## Development setup

Use Python 3.11 or 3.12 in a virtual environment. Install the appropriate PyTorch build for your operating system and backend, then install this project from the repository root:

```bash
python -m pip install -e '.[dicom,dev]'
python -m pytest tests -q
```

For native Apple Silicon development, use the `apple` extra:

```bash
python -m pip install -e '.[apple,dicom,dev]'
```

Keep optional radiomics dependency experiments in a separate environment. The current PyRadiomics 3.0.1 build limitation is documented in [the Apple Silicon guide](docs/APPLE_SILICON.md). A skipped integration test is not evidence that real feature extraction works.

## Before opening a pull request

1. Open an issue for a substantial architecture change, new dependency, or checkpoint format change so the approach can be discussed.
2. Create a branch in your fork and keep the change focused.
3. Add a regression test when fixing a substantive defect. Prefer small synthetic fixtures over downloaded clinical data.
4. Run checks relevant to the change and state the exact commands and results, including skips and unavailable hardware.
5. Update the relevant documentation when commands, schemas, numerical assumptions, or supported inputs change.

Explain the concrete problem, the resulting behavior, and any compatibility impact in the pull request. Do not claim MPS, CUDA, radiomics, or clinical validation based only on a CPU test or mocked dependency.

## Invariants to preserve

- Keep original HU data separate from network normalization and preserve physical image geometry.
- Reject incompatible checkpoints and unsupported input geometry explicitly. Do not silently substitute CPU after an explicit accelerator request.
- Keep segmentation scores separate from metastasis probabilities. Connected components are candidates, not automatic anatomical node identities or IASLC stations.
- Keep review approval separate from pathology verification. Record the unit of the reference standard and evidence used for matching.
- Keep patients isolated across dataset splits. Feature selection and other fitted processing belong inside the training folds of any later prediction study.
- Report failed measurements and unavailable validations honestly; do not replace them with fabricated values or zero-filled features.

## Hardware validation

Use the hardware validation issue form. Include the exact chip/GPU, operating system, native interpreter architecture, Python/PyTorch versions, requested and actual backend, precision, and command settings.

For an Apple Silicon report, start with:

```bash
ctln doctor --device mps --output reports/doctor-mps.json
ctln validate-device --device mps --output reports/parity-mps.json
ctln benchmark --device mps --roi-size 64 64 64 --batch-size 1 \
  --base-channels 16 --levels 4 --warmup 1 --iterations 3 \
  --output reports/benchmark-mps.json
```

Run the same benchmark configuration on CPU before reporting a speed comparison. State what the timing includes. The benchmark measures synthetic patch forward execution, not CT loading or full-volume sliding-window inference. Current allocated MPS tensor memory is not peak unified-memory usage. Keep `PYTORCH_ENABLE_MPS_FALLBACK` unset for the explicit MPS probe.

## Data and rights

Use synthetic reproductions in public issues and pull requests. Do not upload patient images, DICOM headers, identifiers, private file paths, access tokens, or institution-restricted data. Hardware JSON and logs should be reviewed before posting.

Only contribute code, text, data, or model artifacts you have the right to share. Preserve upstream attribution and licensing. Software contributions are distributed under the repository's [BSD 2-Clause license](LICENSE); externally sourced data and model weights retain their own terms. Do not add private teaching-video material or copyrighted figures as public fixtures.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
