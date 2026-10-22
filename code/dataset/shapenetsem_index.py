from pathlib import Path
import numpy as np
import csv


def read_records(txt_path):
    return [
        tuple(part.replace('\\', '/').lstrip('/') for part in line.split())
        for line in Path(txt_path).read_text().splitlines()
        if line.strip()
    ]


def read_normalizations(metadata_path):
    values = {}
    errors = set()

    with Path(metadata_path).open(newline='') as stream:
        for row in csv.DictReader(stream):
            object_id = row['fullId'].removeprefix('wss.')
            try:
                dims = np.array([
                    float(value) for value in row['aligned.dims'].replace('\\', '').split(',')
                ]) / 100
                values[object_id] = float(np.linalg.norm(dims))
            except (ValueError, TypeError, AttributeError):
                errors.add(object_id)

    return values, errors


def load_normalizations(metadata_path):
    values, _ = read_normalizations(metadata_path)
    return values
