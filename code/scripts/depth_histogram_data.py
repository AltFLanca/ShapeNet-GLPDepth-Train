import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.shapenetsem_index import load_normalizations, read_records
from utils.exr import read_exr_red
from utils.legacy_depth_correction import correct_legacy_depth


def normalize_depth(depth, scale):
    scale = np.float32(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid depth normalization scale: {scale}")
    return np.asarray(depth, dtype=np.float32) / scale


def measure_record(record, root, normalizations, raw_depth_only):
    split, rgb_path, depth_path = record
    object_id = Path(depth_path).parent.name

    raw_depth = read_exr_red(root / depth_path)
    depth = raw_depth if raw_depth_only else correct_legacy_depth(raw_depth)
    normalized_depth = normalize_depth(depth, normalizations[object_id])

    return {
        'split': split,
        'rgb': rgb_path,
        'depth': depth_path,
        'max_normalized_depth': float(normalized_depth.max()),
    }


def descriptive_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'count': int(values.size),
        'min': float(values.min()),
        'max': float(values.max()),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'percentiles': {
            str(p): float(np.percentile(values, p))
            for p in [50, 90, 95, 99, 99.9, 100]
        },
    }


def run(args):
    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    split_paths = {'train': args.train_txt_path, 'test': args.test_txt_path}
    records = [
        (split, rgb, depth)
        for split in splits
        for rgb, depth in read_records(split_paths[split])
    ]

    normalizations = load_normalizations(args.metadata_path)
    root = args.data_path

    measure = partial(measure_record, root=root, normalizations=normalizations, raw_depth_only=args.raw_depth)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for first in range(0, len(records), 256):
            rows.extend(pool.map(measure, records[first:first + 256]))

    values = np.asarray([row['max_normalized_depth'] for row in rows], dtype=np.float64)
    counts, bin_edges = np.histogram(values, bins=args.bins)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    with (output / 'per_image_maxima.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=['split', 'rgb', 'depth', 'max_normalized_depth'],
        )
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        output / 'histogram.npz',
        counts=counts,
        bin_edges=bin_edges,
        split=np.asarray(args.split),
    )

    summary = {
        'split': args.split,
        'overall': descriptive_statistics(values),
        'splits': {
            split: descriptive_statistics([
                row['max_normalized_depth'] for row in rows if row['split'] == split
            ])
            for split in splits
        },
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True, help='Path to the shapenetsem directory')
    parser.add_argument('--train_txt_path', type=Path, required=True)
    parser.add_argument('--test_txt_path', type=Path, required=True)
    parser.add_argument('--metadata_path', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parents[2] / "artifacts/depth_histogram")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bins', type=int, default=80)
    parser.add_argument('--raw_depth', action='store_true', help='Use raw depth without applying legacy depth correction')
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())