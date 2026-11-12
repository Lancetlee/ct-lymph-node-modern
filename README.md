# CTLN Modern

[中文说明](README.zh-CN.md) · [Contributing](CONTRIBUTING.md) · [Roadmap](ROADMAP.md) · [Issues](https://github.com/Lancetlee/ct-lymph-node-modern/issues)

Research tooling for **thoracic lymph-node CT segmentation**, human-reviewed ROI annotation, and reproducible radiomics workflows. Built with PyTorch and explicit CPU, NVIDIA CUDA, and Apple Silicon MPS execution paths. Contributions in English and Chinese are welcome.

**Early research software. No trained clinical weights are included.** This is a new 3D Attention U-Net architecture, not a drop-in or numerically equivalent conversion of legacy Keras `.h5` weights. Training requires an appropriate CT cohort and expert segmentation masks. Segmentation scores are not metastasis probabilities; this package does not implement a validated metastasis classifier or automatic IASLC station assignment.

## What is included

- FP32 3D network with interpolation-based decoding, configurable sliding windows and CPU probability accumulation.
- Device diagnostics, synchronized synthetic benchmarks, and CPU-versus-target forward/backward numerical probes. Explicit accelerator requests fail rather than silently switching to CPU.
- NIfTI and optional regular classic DICOM CT loading, original HU preservation, physical geometry checks, and predictions restored to the source CT grid.
- Supervised binary segmentation training, patient-isolated development validation, checkpoint provenance and rejection of incompatible/untrained checkpoints.
- Pending-review instance candidates and optional original-HU PyRadiomics extraction with review gates and frozen configuration hashes.
- A [reusable skill](skill/thoracic-lymph-node-ct/SKILL.md) covering annotation, anatomical/evidence checks, data splitting and the execution workflow. The skill and detailed clinical research notes are currently in Chinese; English translations are welcome.

## Current evidence

| Area | Status |
|---|---|
| Windows CPU software tests | **51 passed, 2 skipped**; recorded report in [validation](validation/) |
| Synthetic train → checkpoint → infer | Executed with real optimizer steps, physical-grid and source-file checks |
| Synthetic classic DICOM IO | Executed using SimpleITK, including HU rescale, LPS/RAS and irregular-stack rejection |
| Apple MPS / M5 Max or CUDA hardware | Execution paths implemented; **not tested on actual accelerator hardware** in this initial release |
| Real PyRadiomics 3.0.1 extraction | **Not validated**; two actual integration tests skipped because the package was unavailable |
| Legacy H5 conversion / clinical accuracy | **Not completed** |

MPS targets the Mac GPU through PyTorch. No custom M5-specific Metal kernels or Apple Neural Engine implementation are included, and no M5 speedup is claimed. We especially welcome reproducible hardware reports using the issue form.

## Install

Use Python **3.11 or 3.12** in a virtual environment. Clone or download this repository first. Install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) for your machine, then install from the repository root:

```bash
python -m pip install -e '.[dicom,dev]'
ctln doctor --device cpu --output reports/doctor-cpu.json
python -m pytest tests -q
```

For native arm64 Apple Silicon:

```bash
python -m pip install -e '.[apple,dicom,dev]'
ctln doctor --device mps --output reports/doctor-mps.json
ctln validate-device --device mps --output reports/parity-mps.json
ctln benchmark --device mps --roi-size 64 64 64 --batch-size 1 \
  --base-channels 16 --levels 4 --warmup 1 --iterations 3 \
  --output reports/benchmark-mps.json
```

Run the same benchmark configuration on CPU when reporting speed comparisons. The benchmark measures random-network patch forward execution, not CT IO or full-volume inference. The numerical probe uses a small synthetic network and is not clinical validation. See the [Apple Silicon guide](docs/APPLE_SILICON.md) for native Python, macOS/PyTorch requirements and memory limits.

## Train and infer

Copy [the training config](configs/train.example.yaml) and [case manifest](configs/cases.example.csv), replace all placeholder paths, and provide `training.dataset_description`. Masks must be integer binary labels on the same physical grid as the HU CT. Confirm NIfTI HU provenance and unknown millimetre units explicitly in the training config.

Use a stable patient ID across scans and keep all scans from one patient in one split. The code requires train and val cases and does not load held-out test images. File/path guards do not replace identity reconciliation and cohort-wide deduplication.

```bash
ctln train --manifest cases.csv --config train.yaml --output models/ln.pt --device mps
ctln infer --image ct.nii.gz --checkpoint models/ln.pt --output results/case001 \
  --device mps --assume-hu --scan-id case001 \
  --roi-size 64 64 64 --batch-size 1 --overlap 0.5
```

Inference writes a segmentation probability volume, binary mask, candidate instances, pending-review JSON and a run manifest. Connected components are candidates, not automatically individual nodes, stations or pathology matches. Review/edit the masks and records before feature extraction; image approval does not establish pathology truth. See [the workflow and record schema](skill/thoracic-lymph-node-ct/references/integration.md).

## Optional radiomics

PyRadiomics runs on CPU and is not part of the core GPU installation. Version 3.0.1 lacks wheels for this project's modern Python/Apple arm64 environments. Its source build failed in the initial Windows/Python 3.12 environment before compilation. The `radiomics` extra declares the dependency; it does not guarantee a working build.

Validate a separate extraction environment with `python -m pytest tests/test_radiomics_integration.py -q` and ensure the tests actually run before claiming support. With that environment available:

```bash
ctln extract --image ct.nii.gz --labels results/case001/candidate_instances.nii.gz \
  --review results/case001/candidate_review.json --params configs/radiomics.example.yaml \
  --output features/case001 --assume-hu
```

Use the original HU CT, reviewed integer masks and explicit feature settings. Do not extract from network-normalized data or substitute a probability volume for a categorical ROI. Patient-level feature/model development and pathology-matched metastasis validation remain separate research tasks.

## Help improve the project

Useful starting points are [M5/MPS and CUDA acceptance reports, modern radiomics packaging, legacy-conversion research, geometry regression cases, and documentation](ROADMAP.md). Open an issue before a major architecture or dependency change. Small focused PRs with synthetic reproductions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

Never post patient images, identifiers, private teaching videos, credentials or institution-restricted material in public issues. Review logs and hardware reports for private paths before sharing.

## Attribution and license

Inspired by the audit and research workflow of [dbouget/ct_mediastinal_structures_segmentation](https://github.com/dbouget/ct_mediastinal_structures_segmentation). This is an independent engineering rewrite and does not reproduce the upstream architecture or published accuracy. The upstream BSD notice is retained. See [NOTICE.md](NOTICE.md) and the [BSD 2-Clause license](LICENSE); external data, publications and model weights retain their own terms.
