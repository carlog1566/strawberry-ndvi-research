# Historical project documentation

[Current overview](../README.md) | [Current commands](workflows.md) | [Current research methods](research-methods.md)

This archive preserves the earlier README and validation report together. These are historical descriptions, not current research conclusions. Commands and plain-text paths in the archived material refer to the project root. The two original files were removed with approval after verifying that their contents were preserved here.


## Earlier README

### UAV NDVI forecasting for strawberry fields

Runnable per-pixel AgriMatNet adaptation for irregular DJI Phantom 4 Multispectral NDVI acquisitions at Cal Poly Pomona. Compares a quantile transformer with scikit-learn ordinary least squares using identical spatially separated pixel/date examples. No measured research results are supplied.

#### Installation

From this directory in PowerShell (Python 3.10+):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

Training automatically uses CUDA when available; install a compatible CUDA-enabled PyTorch build if your environment currently has a CPU build. Evaluation and tiled inference currently use CPU. Windows DataLoader workers default to zero; raise `num_workers` after checking disk throughput. Run all commands from this directory.

#### Original implementation and changes

Source: [ArCo Group ndvi-forecasting](https://github.com/arco-group/ndvi-forecasting), cloned into `upstream/ndvi-forecasting`, inspected at commit `93ca6fae1acb4cbde9f4f8494fb8ee0029f205fe`. Paper: Iele et al., *Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates* (2026), linked by the repository at [arXiv](https://arxiv.org/abs/2602.17683). The detailed source review is in [docs/upstream-review.md](upstream-review.md). Reused architecture is attributed in `uav_ndvi/model.py`; the original copyright and MIT terms are preserved in [src/uav_ndvi/licenses/AgriMatNet-LICENSE](../src/uav_ndvi/licenses/AgriMatNet-LICENSE).

| Component | Adaptation and reason |
| --- | --- |
| `model_quantile.py` | Retains separate history/future projections and encoders, pre-norm ReLU transformer blocks, masked mean history pooling, concatenation fusion, and quantile head. Predicts one requested date per example. |
| `layers.py` | Replaces sequential index sinusoidal positions with continuous actual elapsed days. Three versus forty days produces different encodings. |
| Future stream | A single known date token with elapsed days and calendar sine/cosine. It never contains future NDVI or future measured covariates. |
| History stream | NDVI, optional past spectral features, explicit per-feature missing indicators, and calendar sine/cosine. NDVI-missing and padding tokens are excluded from attention/pooling. |
| `train_quantile_ablation.py` | Preserves Adam and pinball loss; adds validation-median-RMSE early stopping, GPU training, fixed spatial partitions, and reproducible example selection. Removes satellite/weather ablations. |
| `torch_dataset.py` | Replaces eager satellite cache loading with memory-mapped aligned raster storage and season-specific, variable-length per-pixel sequences. |
| `scaler.py` | No fitted NDVI scaling is needed: native [-1,1] values are retained, calendar features bounded, elapsed days explicit. The baseline's StandardScaler is fitted only on training examples. No statistics from validation/test are fitted. |

Defaults: d_model=64, four layers **in each encoder**, eight heads, feedforward=256, dropout=.1, Adam learning rate=.0003, quantiles [.1,.5,.9]. The median is the point forecast. This is a dual-encoder fusion architecture, not an encoder-decoder cross-attention transformer. A singleton future stream keeps the original structure but has little temporal-attention work; simplify it only as a separately evaluated ablation. No pretrained weights are used.

The original `models/LinearReg_Baseline.py`, original manifest, raster files, saved models, and existing predictions are preserved. Its older split and output semantics differ; use the new CLI for this comparison. No proposal document was found in the workspace during implementation.

#### Prepare your real acquisitions

Use single-band, floating-point NDVI GeoTIFFs with values in [-1,1], embedded CRS/transform, and proper NoData metadata or internal validity mask. The expected site CRS is EPSG:32611; processing also accepts other declared CRSs and reprojects to the reference. Integer scaled NDVI must first be converted to physical NDVI units; scale/offset metadata is not automatically applied. NDVI maps are inputs; raw multiband orthomosaics are not interpreted automatically.

Suggested structure:

```text
dataset/
  2024/2024_04_05_NDVI.tif
  2024/2024_04_19_NDVI.tif
  2024/2024_05_10_NDVI.tif
  2024/2024_06_02_NDVI.tif
  2025/...
acquisitions.csv
```

```csv
date,season,path
2024-04-05,2024_strawberry,dataset/2024/2024_04_05_NDVI.tif
2024-04-19,2024_strawberry,dataset/2024/2024_04_19_NDVI.tif
2024-05-10,2024_strawberry,dataset/2024/2024_05_10_NDVI.tif
2024-06-02,2024_strawberry,dataset/2024/2024_06_02_NDVI.tif
```

Dates must be actual acquisitions in YYYY-MM-DD format. One mosaic per date per season. Gaps are allowed; do not invent monthly entries or interpolate by default. Set season IDs from actual planting/harvest cycles: multiple cycles per year should have distinct IDs; a cycle crossing New Year can share a single ID. Different years are separate by default if `season` is omitted, with a warning. Your existing `date,with_soil` manifest is accepted as a legacy alias. Review season boundaries before relying on the results. A season with fewer than `min_history + 1` dates produces no supervised examples; its earlier observations can still support inference if enough history exists.

Paths are relative to the manifest's directory, or `--data-root` when provided; absolute paths work too. `--reference` selects an explicit study-area grid. Otherwise the first acquisition after sorting by season/date defines extent, transform, resolution, and CRS, preserved in all outputs. Areas outside that grid are excluded. For a later inference cache, use exactly the same reference. Alignment uses nearest-neighbor WarpedVRT reads in windows; it does not fix inaccurate UAV georegistration. Inspect the saved alignment report and overlays, especially at 0.016 m resolution.

Declared NoData, masks, NaN, infinity, and out-of-range NDVI become NaN in the cache. Real NDVI zero remains zero unless the source itself declares zero as NoData (a warning explains this irreversible ambiguity). Background stored as valid zero without metadata remains valid numerically, but does not meet the vegetation prediction mask. Soil-removed maps are supported, although valid negative/low NDVI values removed at source cannot be recovered. Histories retain valid soil values as measurements, distinct from missing observations.

#### Preprocess and build identical forecasting examples

Edit a copy of `configs/default.json` as needed and use that configuration consistently:

```powershell
.venv/Scripts/python.exe -m uav_ndvi preprocess --manifest ndvi_manifest.csv --out cache/uav --config configs/default.json
.venv/Scripts/python.exe -m uav_ndvi examples --cache cache/uav --out artifacts/examples --config configs/default.json
```

Optional reference and data location:

```powershell
.venv/Scripts/python.exe -m uav_ndvi preprocess --manifest acquisitions.csv --data-root D:/UAV --reference D:/UAV/field_reference.tif --out cache/reference_grid --config configs/default.json
```

The disk-backed `values.npy` has shape (acquisitions, rows, columns, features); missing values remain NaN. Estimated storage is dates × height × width × channels × 4 bytes, plus source rasters and temporary disk/OS cache overhead. Raster reads and example searches use tiles, not a complete in-memory temporal image stack. `tile_size` bounds raster working memory; a tile still includes all selected history dates. Allow adequate disk space. Preprocessing refuses to overwrite a completed cache; use a fresh directory after changing inputs. No silent downsampling occurs.

For every observed target date, example construction uses all strictly earlier same-season acquisitions within `max_history_days`, truncated to the latest `max_history` acquisition dates. It requires `min_history` finite per-pixel NDVI values. The latest finite historical NDVI must exceed `threshold` (default .2); the per-pixel horizon from that observation must be <= `max_horizon_days`. Targets must be finite vegetation NDVI > .2. Multiple target dates produce multiple examples at a location. Sequences are padded in batches with separate masks; there is no regular monthly grid. Missing scheduled observations can appear as masked tokens, while entirely absent acquisitions are represented by elapsed time gaps.

Spatial partition: first 70% of grid rows train, next 15% validation, last 15% test (integer rounding). `split_axis="column"` uses columns instead. This creates contiguous geographic regions; all dates at a coordinate stay in one partition. These percentages describe **grid area**, not an exact count of eligible vegetation pixels or sampled examples. Cropland shape and missingness can change usable proportions. `buffer_pixels` excludes a strip on both sides of each boundary; at .016 m, 63 pixels is about 1 m. Choose a meaningful buffer after examining spatial autocorrelation. Boundaries should pass through the actual field footprint; choose a suitable reference extent and axis. Empty partitions fail explicitly.

Uniform random-priority reservoirs cap each split at `samples_per_split` examples (20,000 default), across eligible pixel/date pairs. Memory is bounded by this cap and tile size. Saved index arrays are the single source for both models; no model-specific resampling occurs. Large temporal caches can still take time to scan, and shuffled pixel reads can be disk-bound. Sampling more pixels does not create more independent acquisition dates.

#### Train and evaluate

```powershell
.venv/Scripts/python.exe -m uav_ndvi train --examples artifacts/examples --out runs/strawberry --config configs/default.json
.venv/Scripts/python.exe -m uav_ndvi evaluate --run runs/strawberry --split validation
.venv/Scripts/python.exe -m uav_ndvi evaluate --run runs/strawberry --split test
```

`train` fits both models. The transformer uses mini-batches and validates every epoch; `patience` stops on unimproved validation RMSE of the unmodified median output, saving the best checkpoint. The linear baseline is standardized ordinary least squares using last/first/mean/std NDVI, observation count, history duration, lead time, fitted historical NDVI/day trend, lead interactions, and target calendar features. Optional historical features add available-value means and availability indicators. Neither model gets a coordinate feature.

Run artifacts: `transformer.pt`, `linear.joblib`, `run.json`, `history.csv`, `training.png`. Evaluation adds `test_comparison.csv`, `test_per_date.csv`, `test_predictions.csv`, per-model scatter density plots, health confusion matrices, and `test_interval.json` (or the selected split prefix). Metrics are RMSE, MAE, R², fixed-three-class macro F1 and accuracy. Undefined R² is blank. Health classes are Unfavorable <.4, Marginal [.4,.6), Healthy >=.6; macro F1 includes all three classes with zero contribution for undefined classes. Metrics use the saved valid vegetation examples only. `test_predictions.csv` preserves coordinates and quantiles for audit.

Use fresh cache, example, training-run, and forecast output directories for changed experiments. Completed artifacts are protected against accidental replacement; evaluation reports for an existing run can be regenerated.

Intervals use raw q.1/q.9 bounds: empirical coverage, mean width, nominal .8 coverage, and quantile crossing rate. Quantiles are not constrained to be ordered or calibrated. Crossings are reported instead of silently sorted. Point outputs and metrics are not clipped to [-1,1], so out-of-range forecasts remain visible for diagnosis. The test split should remain untouched during tuning. Synthetic scores are software checks only and are not evidence that either model is superior on the UAV data.

#### Forecast a chosen date

```powershell
.venv/Scripts/python.exe -m uav_ndvi forecast --cache cache/uav --run runs/strawberry --date 2025-06-30 --season 2025 --out predictions/2025-06-30-transformer --classification
.venv/Scripts/python.exe -m uav_ndvi forecast --cache cache/uav --run runs/strawberry --date 2025-06-30 --season 2025 --model linear --out predictions/2025-06-30-linear
```

Use the season ID actually recorded in your manifest. Target-date and later rasters can exist in the cache; they are never used as input. No target-date raster is required. A default **explicit prediction-mask rule** allows only pixels with sufficient finite historical NDVI, latest valid historical NDVI > .2, and an acceptable per-pixel horizon. It cannot forecast unseen vegetation in historical background. Supply a stricter field mask if desired:

```powershell
.venv/Scripts/python.exe -m uav_ndvi forecast --cache cache/uav --run runs/strawberry --date 2025-06-30 --season 2025 --mask dataset/field_mask.tif --out predictions/masked --classification
```

The mask must already match the reference grid. Positive finite unmasked values are included, intersected with historical eligibility; derive the field mask independently of future NDVI. All output rasters retain the reference CRS, extent, resolution, and transform; NoData=-9999. `predicted.tif` is median or linear NDVI, optional `q0.1.tif`/`q0.9.tif` are transformer bounds, and `health.tif` codes are 0/1/2. Missing background is NoData, never class 0. If a matching target acquisition exists, `actual.tif` contains valid vegetation truth, and `error.tif` is prediction minus actual on their valid intersection. PNG thumbnails accompany predicted, actual, and error rasters. Maps and quantiles are generated tile-by-tile and prediction-batch-by-batch; thumbnails are bounded to 1200 pixels per dimension. Preview color limits are fixed for comparability and may saturate extremes; inspect the GeoTIFF values directly.

To add newer acquisitions, preprocess a new manifest into a new cache using the training reference grid, then forecast with that cache. Season labels and feature order must remain consistent. Model/checkpoint files use pickle-based loading; load files you created or trust.

#### Optional multispectral inputs

Default `extra_features=[]` needs only NDVI. To experiment with historical red-edge reflectance, set `"extra_features": ["rededge"]` and add `rededge_path` for every acquisition. Each extra input must be an already calibrated **single-band** raster in comparable physical units; extract/calibrate original multiband mosaics separately. Missing pixels are supported; missing file paths are not. Additional features are aligned to the same grid and used only from historical dates. Rebuild the cache/examples and retrain after changing feature order. Avoid mixing arbitrary raw digital numbers across flights; transformer extra features are not automatically fitted/scaled.

#### Reproduce the synthetic integration run

```powershell
.venv/Scripts/python.exe -m tests.test_pipeline artifacts/synthetic
.venv/Scripts/python.exe -m uav_ndvi preprocess --manifest artifacts/synthetic/manifest.csv --out cache/synthetic --config configs/smoke.json
.venv/Scripts/python.exe -m uav_ndvi examples --cache cache/synthetic --out artifacts/synthetic_examples --config configs/smoke.json
.venv/Scripts/python.exe -m uav_ndvi train --examples artifacts/synthetic_examples --out runs/synthetic --config configs/smoke.json
.venv/Scripts/python.exe -m uav_ndvi evaluate --run runs/synthetic
.venv/Scripts/python.exe -m uav_ndvi forecast --cache cache/synthetic --run runs/synthetic --date 2025-05-21 --season 2025 --out predictions/synthetic --classification
```

Tests cover GeoTIFF alignment, missing values versus real zero, real day intervals, seasonal and temporal isolation, spatial location isolation, masked forward behavior, a small actual training/evaluation run, and georeferenced outputs for both models. Pytest discovery is scoped to `tests/`: the upstream repository includes a test-named evaluation script that depends on an absent `cache_builder_unified` module and satellite caches.

#### Research limits

Spatial holdout shares acquisition dates with training. It measures transfer to withheld areas, not generalization to unseen future dates or years. A separate chronological/season holdout experiment is needed to make that stronger claim; this CLI implements the requested spatial experiment. Do not infer that a high pixel count compensates for few dates. In particular, four 2025 acquisitions supply very few independent temporal outcomes. Forecasts at longer horizons than the training distribution are uncertain even within the configured maximum. At 0.016 m, alignment error and neighboring-pixel correlation can dominate apparent performance. Buffers and careful field registration matter.

Vegetation-only truth filtering excludes NDVI <= .2, including severe crop loss or soil; reported metrics do not measure those cases. Health thresholds are operational NDVI categories, not independently validated disease diagnoses. Sparse observations and soil-removal masks limit what can be learned about deterioration. No weather, disease labels, satellite inputs, interpolation, or pretrained satellite weights are required. No claim of transformer superiority or agronomic forecasting accuracy has been made.


## Earlier validation report

### Executed validation

Executed in the project virtual environment on Windows, using Python 3.13 and CPU PyTorch. The real UAV imagery was not used to train a model or estimate research performance.

Final project suite command:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Seven tests passed before this report was written. They execute:

1. Synthetic GeoTIFF preprocessing with two seasons, irregular dates, a one-pixel shifted acquisition, source NoData, explicit NaN, genuine NDVI zero, and missing background.
2. Variable-length strict-past histories; same-season boundaries; actual [0, 3, 43] elapsed-day offsets; separate missing indicators.
3. Pairwise disjoint train/validation/test coordinate sets, across all dates.
4. Dual-encoder quantile forward pass, finite outputs, elapsed-day sensitivity, and output invariance to changed padded token values.
5. Dataset input invariance after changing target and later stored NDVI values.
6. Optional historical spectral-channel cache construction, baseline features, and transformer forward pass.
7. Two-epoch transformer training and baseline fitting, evaluation, quantile reports, both models' GeoTIFF export, retained geotransform/CRS/shape, background masking, unchanged forecast after future-truth mutation, external field-mask restriction, forecasting without truth, and rejection of insufficient same-season history.

A separate retained command-line integration run also completed every documented stage using `configs/smoke.json`:

- Input GeoTIFFs and manifest: `artifacts/synthetic/`.
- Windowed aligned cache: `cache/synthetic/`.
- Shared sampled examples: `artifacts/synthetic_examples/` (30 per partition).
- Both fitted models, history, plots, and evaluation reports: `runs/synthetic/`.
- Median, quantile, actual, error, health GeoTIFFs and previews: `predictions/synthetic/` (238 eligible forecasts).

The initial unrestricted pytest discovery encountered an upstream evaluation script whose imported `dataset_builder.cache_builder_unified` is absent from the cloned repository. `pytest.ini` limits discovery to this project's tests; the adaptation does not require that upstream cache builder or its satellite/weather environment.

The two-epoch smoke configuration is deliberately tiny and verifies that the commands run. Its scores must not be reported as research evidence. GPU execution, full-scale UAV throughput, actual season labels, raster georegistration quality, long-horizon calibration, and forecasting performance on the real dataset remain unvalidated.
