from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def legacy_projection_factor():
    """Return the fixed per-pixel factor used by the legacy depth encoding."""
    v, u = np.indices((480, 640), dtype=np.float64)
    cosine = 1 / np.sqrt(1 + ((u - 320) / 588) ** 2 + ((v - 240) / 588) ** 2)
    quantized = np.floor(255 * cosine + 0.5) / 255
    factor = np.where(
        quantized <= 0.04045,
        quantized / 12.92,
        ((quantized + 0.055) / 1.055) ** 2.4,
    )
    factor.setflags(write=False)
    return factor


def correct_legacy_depth(raw_depth):
    """Convert a legacy raw depth image to float32 axial depth in meters."""
    raw_depth = np.asarray(raw_depth)
    valid = np.isfinite(raw_depth) & (raw_depth > 0)
    corrected = np.zeros(raw_depth.shape, dtype=np.float64)
    np.divide(raw_depth, legacy_projection_factor(), out=corrected, where=valid)
    return corrected.astype(np.float32)


def backproject_z(depth, fx=700.0, fy=700.0, cx=319.5, cy=239.5):
    """Backproject axial Z depth to an HxWx3 camera-coordinate array."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be a 2-D axial-depth array")
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise ValueError("Intrinsics must be finite, with positive fx and fy")

    v, u = np.indices(depth.shape, dtype=np.float64)
    z = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    return np.stack(
        ((u - cx) * z / fx, (v - cy) * z / fy, z),
        axis=-1,
    )
