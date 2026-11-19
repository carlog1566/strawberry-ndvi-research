> Public sharing snapshot: raw UAV rasters, aligned caches, sample arrays and trained research checkpoints are not included. Create a fresh virtual environment in this Research folder. See the repository-root README for setup and scope.

# Running the pipeline

[Project overview](../README.md) | [Research methods](research-methods.md) | [Repository map](repository-map.md)

Run all commands from the project root. Review the provisional season labels and storage requirements in [Research methods](research-methods.md) before a full research run.

## Run from Windows PowerShell

From the project root, use the existing interpreter or activate it. Dependency declarations are in `requirements.txt`; these instructions do not require modifying the existing environment.

```powershell
.\.venv\Scripts\Activate.ps1
python -m uav_ndvi --help
python -m pytest -q
```

Prepare the shared union-grid cache once. This is a fresh cache; the original `cache/uav` remains untouched. Confirm/edit the provisional labels before a final research run.

```powershell
python -m uav_ndvi prepare --manifest manifest.csv --out cache/season_blocks_v1 --config config.yaml
```

Experiment A:

```powershell
python -m uav_ndvi examples --cache cache/season_blocks_v1 --out artifacts/spatial_generalization/v1 --config configs/spatial_generalization.json
python -m uav_ndvi train --examples artifacts/spatial_generalization/v1 --out runs/spatial_generalization/v1 --config configs/spatial_generalization.json
python -m uav_ndvi evaluate --run runs/spatial_generalization/v1
python -m uav_ndvi forecast --cache cache/season_blocks_v1 --run runs/spatial_generalization/v1 --date 2025-04-14 --season provisional_2025_feb_apr --out predictions/spatial_generalization/v1/2025-04-14 --model all --classification
```

Experiment B:

```powershell
python -m uav_ndvi examples --cache cache/season_blocks_v1 --out artifacts/cross_year_2025/v1 --config configs/cross_year_2025.json
python -m uav_ndvi train --examples artifacts/cross_year_2025/v1 --out runs/cross_year_2025/v1 --config configs/cross_year_2025.json
python -m uav_ndvi evaluate --run runs/cross_year_2025/v1
python -m uav_ndvi forecast --cache cache/season_blocks_v1 --run runs/cross_year_2025/v1 --date 2025-04-01 --season provisional_2025_feb_apr --out predictions/cross_year_2025/v1/2025-04-01 --model all --classification
python -m uav_ndvi forecast --cache cache/season_blocks_v1 --run runs/cross_year_2025/v1 --date 2025-04-14 --season provisional_2025_feb_apr --out predictions/cross_year_2025/v1/2025-04-14 --model all --classification
```

`preprocess` remains an alias for `prepare`; existing command arguments remain available. `--model transformer`, `--model linear`, and `--model persistence` export one model. `--mask` accepts an independently defined aligned positive raster mask. `--reference` chooses alignment origin/spacing. GPU training is automatic when CUDA is available; training seeds, CPU/GPU device, selected epoch, and configuration are recorded. Inference currently runs in bounded CPU batches.

All output-producing stages reject nonempty destinations. Use a new version directory to repeat a run. Evaluation defaults to `RUN/evaluation_test`; use `--out FRESH_DIRECTORY` to repeat it without replacing reports. `--split validation` evaluates validation explicitly. Old run paths in `run.json`/`examples.json` are absolute; moving the entire workspace requires updating those references in copies, not silently rewriting preserved artifacts.

