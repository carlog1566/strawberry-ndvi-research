# Standalone legacy regression

[Project overview](../README.md) | [Current comparison pipeline](workflows.md) | [Repository map](repository-map.md)

The earlier standalone linear-regression workflow is implemented in [`src/uav_ndvi/legacy_linear_regression.py`](../src/uav_ndvi/legacy_linear_regression.py). [`models/LinearReg_Baseline.py`](../models/LinearReg_Baseline.py) remains a launcher and preserves imports of its functions and constants. Existing command arguments and default checkpoint paths are unchanged.

## Which regression workflow to use

| Detail | Current comparison pipeline | Standalone legacy workflow |
| --- | --- | --- |
| Entry point | `python -m uav_ndvi` | `python models/LinearReg_Baseline.py` |
| Implementation | `src/uav_ndvi/models/linear_regression.py` | `src/uav_ndvi/legacy_linear_regression.py` |
| Manifest | `manifest.csv`: `date,path,season_id` | `ndvi_manifest.csv`: `date,with_soil` |
| Configuration | JSON/YAML experiment files | CLI flags stored in the checkpoint |
| Spatial partition | Default geographic 70/15/15 strips | Southern 60%, middle 20%, northern 20% |
| Historical inputs | Variable-length within-season history | Two preceding acquisitions subject to day limits |
| Grid | Current pipeline's aligned cache | Reference extent with explicit meter resolution; default 0.5 m |
| Prediction output | Native outputs with diagnostics | Predictions clipped to [-1, 1] |
| Checkpoint format | Run directory with separate model and metadata files | One joblib dictionary containing model, grid, settings and feature names |

The standalone implementation is retained for earlier work; it is not the supervised baseline fitted by the current three-model comparison. Its history construction does not enforce the current pipeline's explicit season blocks. A `linear.joblib` from a current run is not interchangeable with a standalone checkpoint dictionary.

## Existing files

The root `ndvi_linear_model.joblib`, `ndvi_linear_01m.joblib` and `ndvi_linear_0016m.joblib` have distinct file hashes. Their names alone do not establish grid spacing or model equivalence. They remain at their original paths. The standalone prediction command defaults to `ndvi_linear_model.joblib`; use `--model` to select another compatible saved file.

`ndvi_manifest.csv` resolves relative raster paths from the manifest's directory. Keep that relationship when moving or copying a manifest.

## Commands

Run these from the project root using the existing environment:

```powershell
python models/LinearReg_Baseline.py --help
python models/LinearReg_Baseline.py train --help
python models/LinearReg_Baseline.py predict --help
```

To train a new standalone model without replacing a root checkpoint, choose a fresh output path:

```powershell
python models/LinearReg_Baseline.py train --manifest ndvi_manifest.csv --model-out runs/legacy/v2/model.joblib
```

For prediction, select a compatible model and target date:

```powershell
python models/LinearReg_Baseline.py predict --manifest ndvi_manifest.csv --model runs/legacy/v2/model.joblib --date 2025-04-15 --out predictions/legacy/v2/2025-04-15.tif
```

The script checks acquisition gaps, available historical observations and raster coverage. These example paths do not indicate that a model has already been trained there. Checkpoints, source rasters and root manifests were not moved during organization.
