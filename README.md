# project_Copernicus_FM

Step-based experiment scaffold for **SAR(VV/VH) + S2(B4,B3,B2,B8) + Fmask**.

## Repo layout

- `loader/`: scene discovery, BigTIFF window read, patch indexing, preprocessing
- `scripts/`: Step 1 discovery / Step 2 index build (first scene)
- `config/`: YAML templates
- `output/`: runtime outputs (git-kept via `.gitkeep`)

## Quick start (indexing)

1) Install deps:

```bash
python -m pip install -r requirements.txt
```

2) Step 1: discover scenes (excludes `._*.tif`)

```bash
python scripts/01_discover_scenes.py
```

3) Step 2: build patch index for the first scene (requires `rasterio`)

```bash
python scripts/02_build_index_first_scene.py
```

Notes:
- `scripts/02_build_index_first_scene.py` uses `INDEX_SCOPE` at file top.
- Set `INDEX_SCOPE = "all"` for all scenes or `"first"` for first scene only.

## Copernicus-FM checkpoint wiring

List locally discoverable Copernicus-FM checkpoints:

```bash
python scripts/03_check_copernicus_fm.py
```

Try model construction + checkpoint load:

```bash
python scripts/03_check_copernicus_fm.py
```

## Stage 1 tiny runs

GAN baseline:

```bash
python scripts/02_train_stage1_gan.py
```

Copernicus-FM + decoder:

```bash
python scripts/02_train_stage1_copfm.py
```

Notes:
- Per-epoch val sample PNG is saved to `output/<exp_id>/samples/` when `eval.save_val_png_each_epoch: true`.
- Per-epoch train/val metrics are stored under `metrics.json` -> `history`.
