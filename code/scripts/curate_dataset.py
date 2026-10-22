import argparse
import json
from pathlib import Path


REMOVE_OBJECT_IDS = {
    '68af758e75c830d22ba9876b378f34a6',
}


def run(directory):
    summary = {}

    for split in ['train', 'test']:
        input_path = directory / f'valid_{split}.txt'
        output_path = directory / f'curated_{split}.txt'

        lines = input_path.read_text().splitlines()

        curated = [
            line
            for line in lines
            if Path(line.split()[1]).parent.name not in REMOVE_OBJECT_IDS
        ]

        output_path.write_text('\n'.join(curated) + '\n')
        summary[split] = len(curated)

    (directory / 'curated_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True, help='Path to the folder containing the valid split txts')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.directory)