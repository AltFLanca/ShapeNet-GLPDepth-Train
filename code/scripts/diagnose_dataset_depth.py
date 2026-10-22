import argparse
import sys
from pathlib import Path
import numpy as np
import json
from itertools import combinations
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.exr import read_exr_red
from utils.legacy_depth_correction import backproject_z, correct_legacy_depth

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CABINET = "1da9362bcaa79e65136d9cfd13835101"


def plane_stats(points):
    centered = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    normal = vectors[:, 0]
    return float(np.sqrt(np.mean((centered @ normal) ** 2))), normal


def write_ply(path, points, colors):
    vertices = np.empty(len(points), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, i]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def diagnose(dataset_root, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = dataset_root / CABINET / f"{CABINET}_top"
    raw = read_exr_red(str(prefix) + "_depth0001.exr")
    rgb = np.asarray(Image.open(str(prefix) + ".png").convert("RGB"))
    z = correct_legacy_depth(raw)

    candidates = {
        "Raw EXR": raw,
        "Recovered depth": z,
    }
    # An interior face, chosen from RGB; excludes edges, bevels, and the handle.
    roi = np.s_[95:385, 160:430]
    report = {"sample": str(prefix), "roi_xyxy": [160, 95, 430, 385],
              "camera_intrinsics": {"fx": 700, "fy": 700, "cx": 319.5, "cy": 239.5},
              "candidate_metrics": {}}
    for name, depth in candidates.items():
        patch = depth[roi]
        rms, _ = plane_stats(backproject_z(depth)[roi].reshape(-1, 3))
        report["candidate_metrics"][name] = {
            "z_mean": float(patch.mean()), "z_std": float(patch.std()),
            "z_peak_to_peak": float(np.ptp(patch)), "plane_rms": rms}

    points_grid = backproject_z(z)
    valid = z > 0
    points, colors = points_grid[valid], rgb[valid]
    np.savez_compressed(output_dir / "cabinet_top_corrected.npz", depth_z=z,
                        xyz=points, rgb=colors,
                        K=np.array([[700, 0, 319.5], [0, 700, 239.5], [0, 0, 1]]))
    write_ply(output_dir / "cabinet_top_corrected.ply", points, colors)

    # Compare the raw EXR directly with the recovered depth.
    raw_points = backproject_z(raw)[valid]
    indices = np.random.default_rng(7).choice(len(points), min(20000, len(points)), replace=False)
    combined = np.concatenate((raw_points, points))
    center = (combined.max(axis=0) + combined.min(axis=0)) / 2
    radius = np.ptp(combined, axis=0).max() / 2
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    for i, (title, cloud) in enumerate([
        ("Raw EXR", raw_points),
        ("Recovered depth: flat face + handle", points)
    ]):
        ax = fig.add_subplot(grid[0, i], projection="3d")
        ax.scatter(*cloud[indices].T, c=colors[indices]/255, s=1, depthshade=False)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z (depth units)", title=title,
               xlim=(center[0]-radius, center[0]+radius),
               ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-65)
    ax = fig.add_subplot(grid[1, :])
    for name in ("Raw EXR", "Recovered depth"):
        ax.plot(np.arange(160, 430), candidates[name][240, 160:430], label=name)
    ax.set(xlabel="Pixel column (row 240, inside the flat face)",
           ylabel="Z (depth units)", title="Measured Cross-Section")
    ax.legend()
    ax.grid(alpha=.2)
    # fig.savefig(output_dir / "cabinet_depth_recovery.png", dpi=160)
    plt.rcParams["pdf.fonttype"] = 42  # Embed TrueType fonts
    fig.savefig(
        output_dir / "cabinet_depth_recovery.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Planarity cannot calibrate focal length. Known perpendicular cabinet faces can.
    tilted_path = dataset_root / CABINET / f"{CABINET}_r_045_depth0001.exr"
    if tilted_path.exists():
        tilted = correct_legacy_depth(read_exr_red(tilted_path))
        patches = {"left": np.s_[210:330, 215:290],
                   "right": np.s_[180:350, 365:440],
                   "top": np.s_[65:95, 280:380]}
        checks = {}
        for focal in (588, 700):
            xyz = backproject_z(tilted, fx=focal, fy=focal)
            stats = {name: plane_stats(xyz[patch].reshape(-1, 3))
                     for name, patch in patches.items()}
            angles = {}
            for a, b in combinations(patches, 2):
                cosine_angle = np.clip(abs(stats[a][1] @ stats[b][1]), 0, 1)
                angles[f"{a}_{b}"] = float(np.degrees(np.arccos(cosine_angle)))
            checks[str(focal)] = {"acute_angles_deg": angles,
                                 "plane_rms": {k: s[0] for k, s in stats.items()}}
        report["cabinet_perpendicular_faces"] = checks
    (output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "artifacts/depth_recovery")
    args = parser.parse_args()
    diagnose(args.dataset_root, args.output_dir)