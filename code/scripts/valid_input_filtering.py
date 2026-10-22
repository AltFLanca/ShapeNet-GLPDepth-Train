import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import numpy as np
from functools import partial
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.shapenetsem_index import read_normalizations, read_records
from utils.exr import read_exr_red
from utils.legacy_depth_correction import correct_legacy_depth


def validate_record(record, root, metadata_errors, raw_depth_only):
    split, rgb_path, depth_path = record
    rgb_file = root / rgb_path
    depth_file = root / depth_path
    object_id = Path(depth_path).parent.name

    row = {
        'split': split,
        'rgb': rgb_path,
        'depth': depth_path,
        'status': 'valid',
    }

    if not rgb_file.is_file():
        row['status'] = 'missing_rgb'
        return row
    if not depth_file.is_file():
        row['status'] = 'missing_depth'
        return row
    if object_id in metadata_errors:
        row['status'] = 'invalid_metadata'
        return row

    try:
        depth = read_exr_red(depth_file)
        if not raw_depth_only:
            depth = correct_legacy_depth(depth)
    except Exception:
        row['status'] = 'invalid_depth'
        return row

    if not np.any(depth > 0):
        row['status'] = 'empty_depth'

    return row


def run(args):
    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    split_paths = {'train': args.train_txt_path, 'test': args.test_txt_path}
    records = [
        (split, rgb, depth)
        for split in splits
        for rgb, depth in read_records(split_paths[split])
    ]

    object_ids = {Path(depth).parent.name for _, _, depth in records}
    normalizations, metadata_errors = read_normalizations(args.metadata_path)
    metadata_errors |= {
        object_id
        for object_id, scale in normalizations.items()
        if not np.isfinite(scale) or scale <= 0
    }
    metadata_errors |= object_ids - normalizations.keys()

    root = args.data_path
    validate = partial(
        validate_record,
        root=root,
        metadata_errors=metadata_errors,
        raw_depth_only=args.raw_depth,
    )

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for first in range(0, len(records), 256):
            rows.extend(pool.map(validate, records[first:first + 256]))

    valid_samples = sum(row['status'] == 'valid' for row in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        lines = [
            f"/{row['rgb']} /{row['depth']}\n"
            for row in rows
            if row['split'] == split and row['status'] == 'valid'
        ]
        (args.output_dir / f'valid_{split}.txt').write_text(''.join(lines))

    status_counts = {}
    for row in rows:
        status = row['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        'selected_split': args.split,
        'depth_mode': 'raw' if args.raw_depth else 'corrected',
        'listed_samples': len(rows),
        'valid_samples': valid_samples,
        'status_counts': status_counts,
    }

    (args.output_dir / 'valid_input_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True, help='Path to the shapenetsem directory')
    parser.add_argument('--train_txt_path', type=Path, required=True)
    parser.add_argument('--test_txt_path', type=Path, required=True)
    parser.add_argument('--metadata_path', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--output_dir', type=Path, required=True, help='Path to the output directory')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--raw_depth', action='store_true', help='Validate raw depth without applying legacy depth correction')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())