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
python scripts/01_discover_scenes.py --data-root ~/data_2/SARtoRGB/Korea
```

3) Step 2: build patch index for the first scene (requires `rasterio`)

```bash
python scripts/02_build_index_first_scene.py --data-root ~/data_2/SARtoRGB/Korea --limit-patches 64
```
