"""Command-line entry points. No downloads or legacy setup hooks are executed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml

from . import __version__


def _parser():
    parser = argparse.ArgumentParser(prog='ctln', description='Research CT lymph-node segmentation and reviewed radiomics workflow')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='Probe backend and run a small 3D forward/backward smoke test')
    doctor.add_argument('--device', default='auto')
    doctor.add_argument('--output', type=Path)
    doctor.add_argument('--overwrite', action='store_true')
    validate = sub.add_parser('validate-device', help='Compare the same synthetic FP32 network forward/backward with CPU')
    validate.add_argument('--device', default='mps')
    validate.add_argument('--output', required=True, type=Path)
    validate.add_argument('--roi-size', type=int, nargs=3, default=[9, 9, 9], metavar=('D', 'H', 'W'))
    validate.add_argument('--seed', type=int, default=42)
    validate.add_argument('--rtol', type=float, default=1e-3)
    validate.add_argument('--atol', type=float, default=1e-4)
    validate.add_argument('--overwrite', action='store_true')
    benchmark = sub.add_parser('benchmark', help='Synthetic network timing; not segmentation validation')
    benchmark.add_argument('--device', default='auto')
    benchmark.add_argument('--output', type=Path, required=True)
    benchmark.add_argument('--roi-size', type=int, nargs=3, default=[32, 32, 32], metavar=('D', 'H', 'W'))
    benchmark.add_argument('--batch-size', type=int, default=1)
    benchmark.add_argument('--warmup', type=int, default=1)
    benchmark.add_argument('--iterations', type=int, default=3)
    benchmark.add_argument('--base-channels', type=int, default=16, help='Match the training architecture when comparing devices')
    benchmark.add_argument('--levels', type=int, default=4)
    benchmark.add_argument('--no-attention', action='store_true')
    benchmark.add_argument('--amp', action='store_true', help='CUDA FP16 only; MPS uses FP32')
    benchmark.add_argument('--overwrite', action='store_true')
    infer = sub.add_parser('infer', help='Trained checkpoint -> original-grid segmentation and pending review candidates')
    infer.add_argument('--image', required=True, type=Path)
    infer.add_argument('--checkpoint', required=True, type=Path)
    infer.add_argument('--output', required=True, type=Path)
    infer.add_argument('--device', default='auto')
    infer.add_argument('--series-uid')
    infer.add_argument('--assume-hu', action='store_true', help='Confirm that the NIfTI intensity values are HU')
    infer.add_argument('--assume-mm', action='store_true', help='Assert millimetres only when NIfTI units are unknown')
    infer.add_argument('--roi-size', type=int, nargs=3, default=[64, 64, 64], metavar=('D', 'H', 'W'))
    infer.add_argument('--overlap', type=float, default=.5)
    infer.add_argument('--batch-size', type=int, default=1)
    infer.add_argument('--amp', action='store_true')
    infer.add_argument('--threshold', type=float, default=.5, help='Foreground segmentation threshold, not metastasis risk')
    infer.add_argument('--scan-id', default='scan')
    infer.add_argument('--max-accumulator-gib', type=float, default=4.)
    infer.add_argument('--max-case-voxels', type=int, default=134217728)
    infer.add_argument('--overwrite', action='store_true')
    extract = sub.add_parser('extract', help='Original HU + approved instance masks -> optional CPU PyRadiomics')
    extract.add_argument('--image', required=True, type=Path)
    extract.add_argument('--labels', required=True, type=Path)
    extract.add_argument('--review', required=True, type=Path)
    extract.add_argument('--params', required=True, type=Path)
    extract.add_argument('--output', required=True, type=Path)
    extract.add_argument('--series-uid')
    extract.add_argument('--assume-hu', action='store_true')
    extract.add_argument('--assume-mm', action='store_true')
    extract.add_argument('--overwrite', action='store_true')
    train = sub.add_parser('train', help='Train a new binary segmentation architecture using patient-isolated train/val data')
    train.add_argument('--manifest', required=True, type=Path)
    train.add_argument('--config', required=True, type=Path)
    train.add_argument('--output', required=True, type=Path)
    train.add_argument('--device', default='auto')
    train.add_argument('--overwrite', action='store_true')
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        from .pipeline import protect_outputs, write_json
        if args.command == 'doctor':
            from .device import device_report
            if args.output:
                protect_outputs([args.output], overwrite=args.overwrite)
            result = device_report(args.device)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['smoke_test']['passed'] else 1
        if args.command == 'validate-device':
            from .parity import validate_backend
            protect_outputs([args.output], overwrite=args.overwrite)
            result = validate_backend(args.device, roi_size=args.roi_size, seed=args.seed, rtol=args.rtol, atol=args.atol)
            write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['passed'] else 1
        if args.command == 'benchmark':
            from .benchmark import benchmark_model
            from .model import ModelConfig
            protect_outputs([args.output], overwrite=args.overwrite)
            result = benchmark_model(args.device, roi_size=args.roi_size, batch_size=args.batch_size,
                                     warmup=args.warmup, iterations=args.iterations, amp=args.amp,
                                     config=ModelConfig(base_channels=args.base_channels, levels=args.levels,
                                                        attention=not args.no_attention))
            write_json(args.output, result)
        elif args.command == 'infer':
            from .pipeline import infer_case
            kwargs = vars(args).copy()
            kwargs.pop('command')
            result = infer_case(**kwargs)
        elif args.command == 'train':
            from .training import train_segmentation
            result = train_segmentation(args.manifest, args.config, args.output,
                                        device=args.device, overwrite=args.overwrite)
        elif args.command == 'extract':
            from .imaging import load_ct, load_labelmap
            from .radiomics import extract_radiomics
            protect_outputs([args.output / 'features.csv', args.output / 'manifest.json'],
                            [args.image, args.labels, args.review, args.params], overwrite=args.overwrite)
            ct = load_ct(args.image, series_uid=args.series_uid,
                         assume_nifti_hu=args.assume_hu, assume_nifti_mm=args.assume_mm)
            labels = load_labelmap(args.labels, assume_nifti_mm=args.assume_mm)
            records = json.loads(args.review.read_text(encoding='utf-8'))
            if not isinstance(records, list):
                raise ValueError('--review must contain the reviewed JSON array produced by infer; the teaching template is not this schema.')
            params = yaml.safe_load(args.params.read_text(encoding='utf-8'))
            result = extract_radiomics(ct, labels, records, args.output, params=params, hu_confirmed=True)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError, MemoryError, yaml.YAMLError) as exc:
        print(f'ctln: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
