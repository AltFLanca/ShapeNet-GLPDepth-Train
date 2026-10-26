# ShapeNet-GLPDepth-Train

Fine-tune [GLPDepth](https://github.com/vinvino02/GLPDepth), a monocular depth estimation model, on rendered [ShapeNetSem](https://github.com/RavineWindteer/ShapenetSem-to-RGBD) product images. In addition to the two training variants available in [GLPDepth-Edited](https://github.com/RavineWindteer/ShapenetSem-to-RGBD), this repository offers training with the correction of a legacy rendering artifact.

The original ShapeNetSem depth maps contain a systematic distortion introduced during rendering. Instead of storing the true axial depth, the rendering pipeline multiplied depth values by a quantized, sRGB-decoded cosine correction map. This produces artificial curvature and banding, causing planar surfaces to appear curved when reconstructed as 3D point clouds. The correction reverses this transformation to recover axial depth before resizing and normalization.

For further details on the rendering artifact, mathematical correction, and numerical validation, see [Legacy Depth Correction](docs/legacy_depth_correction.md).


## Model Performance

The following table compares GLPDepth models fine-tuned on the original normalized depth maps and the corrected depth maps, evaluated on the curated ShapeNetSem test set. Lower values indicate better performance across all metrics.

| Model | mAPE ↓ | mSPE ↓ | RMSE ↓ | RMSE Log ↓ | Log10 ↓ | SILog ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Previous (20 epochs) | 0.0725 | 0.0157 | 0.1067 | 0.0748 | 0.0305 | 0.0573 |
| Corrected (20 epochs) | 0.0614 | 0.0128 | 0.0906 | 0.0647 | 0.0261 | 0.0502 |
| **Corrected (25 epochs)** | **0.0564** | **0.0109** | **0.0840** | **0.0591** | **0.0238** | **0.0459** |

Training with corrected depth maps consistently improves performance across all evaluation metrics. The corrected model trained for 25 epochs achieves the best results, reducing RMSE by approximately 21.3% compared to the previous model trained for 20 epochs.

## Setup

Install the project dependencies using [uv](https://docs.astral.sh/uv/). From the repository root, run:

```bash
uv sync --locked
```

The rendered ShapeNetSem dataset and pretrained GLPDepth checkpoints are not included in this repository.

## Repository Structure

- `code/train.py` - Fine-tunes GLPDepth on ShapeNetSem depth maps.
- `code/test.py` - Evaluates trained checkpoints.
- `code/model_comparison.ipynb` - Compares original-depth and corrected-depth checkpoints, including visualizations and evaluation metrics.
- `code/dataset/` - Contains loaders for raw, normalized, and legacy-corrected ShapeNetSem depth maps.
- `code/models/` - Contains the GLPDepth model architecture.
- `code/configs/` - Defines training and evaluation arguments.
- `code/scripts/valid_input_filtering.py` - Filters invalid samples from the dataset splits.
- `code/scripts/curate_dataset.py` - Produces curated training and testing splits.
- `code/scripts/diagnose_dataset_depth.py` - Diagnoses the legacy depth-rendering distortion.
- `code/scripts/depth_histogram_data.py` - Generates depth distribution statistics.
- `code/utils/legacy_depth_correction.py` - Implements the legacy depth correction procedure.
- `docs/legacy_depth_correction.md` - Documents the rendering artifact, mathematical correction, and numerical validation.

## Prepare Dataset

Dataset preparation requires the ShapeNetSem rendered images and depth maps, a metadata file, and the original training and testing splits, which can be generated using the [ShapeNetSem](https://github.com/RavineWindteer/ShapenetSem-to-RGBD) repository.

The [valid_input_filtering.py](code/scripts/valid_input_filtering.py) script validates the original dataset splits by checking for missing RGB images or depth maps, invalid normalization metadata, unreadable depth maps, and depth maps without positive values. By default, validation applies the legacy depth correction, which can be disabled using `--raw_depth`.

Run validation with:

```bash
uv run code/scripts/valid_input_filtering.py \
  --data_path /path/to/ShapeNetSem/data \
  --metadata_path /path/to/ShapeNetSem/metadata.txt \
  --train_txt_path /path/to/ShapeNetSem/splits/train.txt \
  --test_txt_path /path/to/ShapeNetSem/splits/test.txt \
  --output_dir /path/to/ShapeNetSem/splits
```

This produces `valid_train.txt` and `valid_test.txt`, along with `valid_input_summary.json` containing validation statistics.

The [curate_dataset.py](code/scripts/curate_dataset.py) script further filters the validated splits. Run curation with:

```bash
uv run code/scripts/curate_dataset.py \
  --directory /path/to/ShapeNetSem/splits
```

This produces `curated_train.txt` and `curated_test.txt`, which are used for training and evaluation, along with `curated_summary.json` containing the number of retained samples.

Currently, curation excludes only a specific problematic ShapeNetSem object.

## Analyze Depth Distributions

The [depth_histogram_data.py](code/scripts/depth_histogram_data.py) script generates depth distribution statistics for the ShapeNetSem dataset. Analyzing these distributions helps determine appropriate values for the `--max_depth` and `--max_depth_eval` parameters when fine-tuning GLPDepth.

By default, the script analyzes depth maps after applying the legacy depth correction. To analyze the original normalized depth maps instead, use the `--raw_depth` argument.

Generate statistics for the corrected depth maps with:

```bash
uv run code/scripts/depth_histogram_data.py \
  --data_path /path/to/ShapeNetSem/data \
  --metadata_path /path/to/ShapeNetSem/metadata.txt \
  --train_txt_path /path/to/ShapeNetSem/splits/curated_train.txt \
  --test_txt_path /path/to/ShapeNetSem/splits/curated_test.txt \
  --output_dir artifacts/depth_histogram_corrected
```

Based on the depth distribution analysis, both training variants use a maximum depth of `2.63`.

**Note:** When analyzing both the original and corrected depth maps, use separate output directories to avoid overwriting the generated statistics.

## Train Models

Training requires the curated ShapeNetSem dataset splits and an initial GLPDepth checkpoint, provided through the `--ckpt_dir` argument.

The training process can be configured using the following arguments:

- `--dataset` - Dataset variant used for training (`shapenetsem_normalized` or `shapenetsem_normalized_corrected`).
- `--exp_name` - Experiment name used to distinguish training runs.
- `--max_depth` - Maximum depth used by the model.
- `--max_depth_eval` - Maximum depth used during evaluation.
- `--ckpt_dir` - Path to the initial model checkpoint to finetune.
- `--epochs` - Number of training epochs.
- `--start_epoch` - Starting epoch.
- `--save_model` - Enables checkpoint saving.
- `--save_freq` - Checkpoint saving frequency.

To fine-tune GLPDepth using normalized depth maps with legacy depth correction, run:

```bash
uv run code/train.py \
  --dataset shapenetsem_normalized_corrected \
  --exp_name normalized_corrected \
  --data_path /path/to/ShapeNetSem/data \
  --metadata_path /path/to/ShapeNetSem/metadata.txt \
  --train_txt_path /path/to/ShapeNetSem/splits/curated_train.txt \
  --test_txt_path /path/to/ShapeNetSem/splits/curated_test.txt \
  --max_depth 2.63 \
  --max_depth_eval 2.63 \
  --ckpt_dir /path/to/initial_checkpoint.ckpt \
  --save_model \
  --save_freq 1 \
  --start_epoch 1 \
  --epochs 25
```

**Note:** Use separate experiment names for the two variants to avoid mixing checkpoints and training outputs.

## Evaluate Models

The [test.py](code/test.py) script evaluates trained GLPDepth checkpoints using the corresponding ShapeNetSem dataset variant.

Evaluate the corrected-depth checkpoint:

```bash
uv run code/test.py \
  --dataset shapenetsem_normalized_corrected \
  --data_path /path/to/ShapeNetSem/data \
  --metadata_path /path/to/ShapeNetSem/metadata.txt \
  --test_txt_path /path/to/ShapeNetSem/splits/curated_test.txt \
  --ckpt_dir /path/to/corrected/epoch_25_model.ckpt \
  --do_evaluate \
  --max_depth 2.63 \
  --max_depth_eval 2.63
```

To visually compare the depth predictions from both variants, use [model_comparison.ipynb](code/model_comparison.ipynb).