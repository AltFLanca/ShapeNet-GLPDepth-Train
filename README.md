Depth Diagnose:

uv run code/scripts/diagnose_dataset_depth.py --dataset-root /home/user/datasets/SHAPENETSEM/data/



Filter Splits:

uv run code/scripts/valid_input_filtering.py   --data_path /home/user/datasets/SHAPENETSEM/data/   --metadata_path /home/user/datasets/SHAPENETSEM/metadata.txt --train_txt_path /home/user/datasets/SHAPENETSEM/splits/train.txt --test_txt_path /home/user/datasets/SHAPENETSEM/splits/test.txt  --output_dir /home/user/datasets/SHAPENETSEM/splits/

uv run code/scripts/curate_dataset.py --directory /home/user/datasets/SHAPENETSEM/splits/



Histogram:

-> ShapenetSem_Normalized
uv run code/scripts/depth_histogram_data.py --data_path /home/user/datasets/SHAPENETSEM/data/ --metadata_path /home/user/datasets/SHAPENETSEM/metadata.txt --train_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_train.txt --test_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_test.txt --output_dir artifacts/depth_histogram_normalized --raw_depth

-> ShapenetSem_Normalized_Corrected
uv run code/scripts/depth_histogram_data.py --data_path /home/user/datasets/SHAPENETSEM/data/ --metadata_path /home/user/datasets/SHAPENETSEM/metadata.txt --train_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_train.txt --test_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_test.txt  --output_dir artifacts/depth_histogram_corrected



Train:

-> ShapenetSem_Normalized
uv run code/train.py --dataset shapenetsem_normalized --data_path /home/user/datasets/SHAPENETSEM/data/ --metadata_path /home/user/datasets/SHAPENETSEM/metadata.txt --train_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_train.txt --test_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_test.txt --max_depth 2.63 --max_depth_eval 2.63 --ckpt_dir /home/user/models/cardoso/GLPDepth/best_model_nyu.ckpt --save_model --save_freq 1 --start_epoch 1 --epochs 25

-> ShapenetSem_Normalized_Corrected
uv run code/train.py --dataset shapenetsem_normalized_corrected --data_path /home/user/datasets/SHAPENETSEM/data/ --metadata_path /home/user/datasets/SHAPENETSEM/metadata.txt --train_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_train.txt --test_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_test.txt --max_depth 2.63 --max_depth_eval 2.63 --ckpt_dir /home/user/models/cardoso/GLPDepth/best_model_nyu.ckpt --save_model --save_freq 1 --start_epoch 1 --epochs 25



Test:

-> ShapenetSem_Normalized
uv run code/test.py --dataset shapenetsem_normalized --data_path /home/user/datasets/SHAPENETSEM/data/ --metadata_path /home/user/datasets/SHAPENETSEM/metadata.txt --test_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_test.txt --ckpt_dir /home/user/models/glp-depth/normalized/epoch_20_model.ckpt --do_evaluate  --max_depth 2.63 --max_depth_eval 2.63

-> ShapenetSem_Normalized_Corrected
uv run code/test.py --dataset shapenetsem_normalized_corrected --data_path /home/user/datasets/SHAPENETSEM/data/ --metadata_path /home/user/datasets/SHAPENETSEM/metadata.txt --test_txt_path /home/user/datasets/SHAPENETSEM/splits/curated_test.txt --ckpt_dir /home/user/models/glp-depth/corrected/epoch_25_model.ckpt --do_evaluate  --max_depth 2.63 --max_depth_eval 2.63



# Recovering the legacy ShapeNetSem depth maps

The supplied cabinet EXR can be recovered without Blender. Its depth values are
axial camera depth multiplied by a **quantized, sRGB-decoded cosine image**.
Reconstructing and dividing by that multiplier removes the curvature and bands.
This inverse is pixelwise: it does not fit the object to a plane or smooth details.

![Original and recovered point clouds](../artifacts/depth_recovery/cabinet_depth_recovery.png)

## What went wrong

There are two separate issues in the older repository's `render_blender.py`:

1. **The correction image is stored as color, with byte precision.** At line 122,
   `bpy.data.images.new(...)` omits `float_buffer=True` and `is_data=True`.
   Both default to `False` in the
   [Blender 2.90 API](https://docs.blender.org/api/2.90/bpy.types.BlendDataImages.html).
   The comment on line 123 that images always store 32-bit floating-point pixels
   is therefore misleading: the Python property exposes floats even for a byte buffer.
   Blender automatically converts color images to linear values in compositing;
   data images require the appropriate color-space setting. See the
   [Blender compositor documentation](https://docs.blender.org/manual/en/3.1/compositing/types/converter/color_space.html).

2. **The two sets of intrinsics disagree.** The correction map uses `fx=fy=588`.
   Later, the actual camera is assigned `lens=35` and `sensor_width=32`, overriding
   the earlier `angle=57°` setting. At 640×480 with square pixels, no shift and
   horizontal sensor fit, this gives `fx=fy=640*35/32=700` pixels. Setting
   `lens_unit='FOV'` changes the parameter display; it does not make `lens=35`
   mean a 35-degree field of view. The resulting horizontal FOV is about 49.13°.

**The supplied script is not a complete account of the actual rendering run.**
Its current default is EEVEE, and lines 83–88 use an all-ones correction image for
that branch. Those settings would not produce the multiplier observed here.
The data are consistent with an older/different configuration that applied the
color-managed cosine image to already-axial depth. The exact original engine and
script revision cannot be proven from these EXRs. Their headers contain no camera
intrinsics or rendering history.

## Exact inverse measured in these files

For the original top-to-bottom 480×640 arrays, with zero-based column `u` and row `v`:

```python
v, u = np.indices((480, 640), dtype=np.float64)
c = 1 / np.sqrt(1 + ((u - 320) / 588)**2 + ((v - 240) / 588)**2)
q = np.floor(255 * c + 0.5) / 255
factor = np.where(q <= 0.04045, q / 12.92, ((q + 0.055) / 1.055)**2.4)
Z = D / factor
```

Here `D` is the raw EXR and `Z` is recovered axial depth, the distance along the
camera's forward axis. Use only finite, positive depths. The helper handles
invalid values and background by returning zero there.

The rounding is essential: a smooth sRGB correction leaves about 0.00507 depth
units of standard deviation on the flat cabinet patch. Dividing by one or two
cosines also leaves errors. Applying an inverse gamma transform to **D itself**
would be incorrect: the transform affected the multiplier before multiplication.

Backproject the recovered `Z` with:

```python
X = (u - 319.5) * Z / 700
Y = (v - 239.5) * Z / 700
xyz = np.stack((X, Y, Z), axis=-1)
```

These camera coordinates are X right, Y down, Z forward. In this formula integer
indices refer to pixel centers. Equivalently, use `(u+0.5, v+0.5)` and principal
point `(320,240)`. The faulty correction image really was centered at integer
`(320,240)`: do not change its center to match the backprojection convention.

The output is already axial depth. **Do not apply another radial-to-axial
conversion**, such as `depth_is_radial=True` in the old notebook.

## Numerical evidence

On cabinet top pixels `depth[95:385, 160:430]`, an interior region excluding edges
and the handle:

| Interpretation | Standard deviation of Z | Peak-to-peak Z |
|---|---:|---:|
| Raw EXR as axial depth | 0.0533716 | 0.264866 |
| Old notebook's radial conversion | 0.0749547 | 0.367267 |
| Undo one cosine | 0.0308819 | 0.160288 |
| Undo two cosines | 0.0084294 | 0.0486888 |
| Undo quantized sRGB multiplier | **2.28 × 10⁻⁷** | **8.09 × 10⁻⁷** |

The restored plane has mean depth **1.9355471** in the EXR's original units. If
the original metric scaling is correct, those units are meters. The tiny residual
measures consistency with a plane; it is not a claim of submicrometer accuracy.

Independent checks:

- The cabinet's left face also becomes flat (standard deviation about 2.23 × 10⁻⁷).
- The second object's (`1a8d161d93764a8996cba1a88ab59678`) top and bottom faces
  become flat, with standard deviations about 5.8 × 10⁻⁷ and 5.4 × 10⁻⁷.
- All foreground values in all **42 local EXRs, across three objects**, recover
  values within 4.10 × 10⁻⁷ relative error of representable float16 values. This
  is strong evidence of a shared encoding and earlier depth quantization, even
  though the final EXRs have FLOAT (32-bit) channels. It does not prove that every
  file in a larger dataset uses the same rendering configuration.
- Three tilted patches in cabinet `r_045` have recovered plane RMS errors of
  0.000251, 0.000230, and 0.000173. Existing depth quantization remains on slopes.
- With focal length 700, the acute angles between these faces are **89.973°,
  89.973°, 89.965°**; with 588 they are **82.195°, 83.479°, 83.471°**. Assuming
  the cabinet faces are perpendicular, this independently supports 700.

The tilted patches use row/column slices `210:330,215:290`, `180:350,365:440`, and
`65:95,280:380`. Plane normals come from the smallest eigenvector of each centered
point covariance matrix. These fits are diagnostic only; they do not change depth.
Planarity by itself cannot determine focal length: changing X/Y scale preserves
planes. This is why optimizing focal length only for plane flatness can mislead.

## Use the fix

The newer training repository already owns the EXR reader and legacy-depth correction: `utils.exr.read_exr_red` and `utils.legacy_depth_correction.correct_legacy_depth`. The migrated diagnostics and notebooks use those functions directly. `backproject_z` lives alongside the correction utility because no equivalent geometry helper was present in the newer repository.

The reconstruction section of [ground_truth_pointcloud.ipynb](ground_truth_pointcloud.ipynb)
now applies the correction explicitly and uses the camera's 700-pixel focal length.
It keeps `sample['depth']` raw and stores corrected depth in `depth_z`.
The viewer centers its cloud for display; `xyz_camera` preserves camera coordinates.

From Python with `code` on the import path:

```python
from utils.exr import read_exr_red
from utils.legacy_depth_correction import backproject_z, correct_legacy_depth

raw = read_exr_red("object_top_depth0001.exr")
z = correct_legacy_depth(raw)
xyz_image = backproject_z(z)
points = xyz_image[z > 0]
```

Apply this exactly once, at the original resolution, before resizing, cropping,
flipping, or training augmentation. Normalize corrected depth afterwards if needed.
The functions do not modify the input arrays or overwrite the source dataset.
This correction is specific to the measured legacy encoding; validate a few flat
surfaces before applying it to files from a different rendering batch. Masked or
clipped geometry and prior depth precision cannot be reconstructed by this inverse.

To reproduce the figures, diagnostics, corrected NPZ and full-resolution colored
PLY point cloud, run from the repository root:

```bash
MPLCONFIGDIR=/tmp/shapenet-matplotlib .venv/bin/python code/diagnose_blender_depth.py
.venv/bin/python -m unittest discover -s code -p 'test_blender_depth.py' -v
```

Artifacts are saved in `artifacts/depth_recovery/`. The NPZ contains `depth_z`,
camera-coordinate `xyz`, uint8 `rgb`, and `K`. The PLY uses the same coordinates.
The eight tests cover measured pixels, tilted planes, local geometric details,
invalid inputs, pixel centers, and the 42 available EXRs.
