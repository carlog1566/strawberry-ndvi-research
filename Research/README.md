> Public sharing snapshot: raw UAV rasters, aligned caches, sample arrays and trained research checkpoints are not included. Create a fresh virtual environment in this Research folder. See the repository-root README for setup and scope.

# Strawberry UAV NDVI forecasting

This folder is the project root. From its parent directory, start with:

```powershell
cd Research
.\.venv\Scripts\Activate.ps1
python -m uav_ndvi --help
```

The existing virtual environment remains in the parent directory. Data, code, configurations, saved models and results are inside this folder. Saved JSON references were migrated; `.relocation.json` lets the loader resolve historical checkpoint paths without rewriting model binaries.

Compare persistence, supervised linear regression, and a UAV adaptation of [AgriMatNet](https://github.com/arco-group/ndvi-forecasting) on identical per-pixel prediction targets.

## Start here

| Task | Guide |
| --- | --- |
| Find an implementation or understand the folders | [Repository map](docs/repository-map.md) |
| Prepare data, train, evaluate or export maps | [Pipeline commands](docs/workflows.md) |
| Understand data splits, model settings and research limitations | [Research methods](docs/research-methods.md) |
| Check prior validation evidence | [Validation](docs/validation.md) |
| Review approved cleanup and compatibility checks | [Cleanup review](docs/cleanup-review.md) |
| Compare the adaptation with the original model | [Upstream review](docs/upstream-review.md) |
| Use the older standalone regression workflow | [Legacy regression](docs/legacy-regression.md) |
| Choose the correct configuration file | [Configuration guide](configs/README.md) |
| Compare models across repeated training seeds | [Repeated comparison and paired t-test](docs/repeated-comparison.md) |
| Prepare a presentation using verified methods, results and gaps | [Presentation brief](docs/presentation-brief.md) |

## Source layout

```text
src/uav_ndvi/                    Current pipeline
  __main__.py                    CLI commands
  config.py, defaults.json       Configuration loading and validation
  preprocessing.py               Raster alignment and disk-backed caches
  dataset.py                     Histories, partitions, samples and batching
  models/
    transformer.py               Adapted AgriMatNetUAV transformer
    linear_regression.py         Current supervised regression baseline
  train.py                       Fitting and checkpoint loading
  evaluate.py                    Metrics and diagnostic reports
  forecast.py                    Forecast rasters and coverage maps
  utils.py                       Reproducibility and tensor-device helpers
  legacy_linear_regression.py    Separate legacy regression implementation
  licenses/                      Packaged attribution
uav_ndvi/__init__.py              Root CLI support without installation
models/LinearReg_Baseline.py      Compatibility launcher for legacy regression
scripts/audit_existing.py         Audit retained research outputs
configs/                         Explicit experiment configurations
config.yaml                      Automatically loaded root configuration
manifest.csv                     Current acquisition and season inventory
ndvi_manifest.csv                Legacy acquisition inventory
requirements.txt, pyproject.toml Dependencies and packaging
tests/                           Automated synthetic and compatibility checks
docs/                            Guides, validation and research history
upstream/                        Original AgriMatNet reference repository
```

The former modules `model.py`, `baseline.py`, `preprocess.py`, `training.py`, `evaluation.py` and `inference.py` remain as compatibility imports. Current implementations have a single location in the layout above. The existing `.venv` supplies the interpreter and dependencies.

## Run from Windows PowerShell

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uav_ndvi --help
python -m pytest -q
```

Follow [Pipeline commands](docs/workflows.md) for both experiment workflows. `prepare` and `preprocess` are aliases; existing command arguments remain available.

## Dataset and provisional crop cycles

`manifest.csv` contains provisional within-cycle observation blocks. Confirm those labels before final research runs. See [Dataset and crop cycles](docs/research-methods.md#dataset-and-provisional-crop-cycles).

## Alignment, missing data, and sampling

The pipeline aligns rasters to a common grid, retains missing-value masks and samples eligible pixel-date pairs. Fine-resolution multi-year caches can require substantial disk space. See [Alignment and sampling](docs/research-methods.md#alignment-missing-data-and-sampling).

## Experiments and model choices

The geographic holdout and held-out 2025 experiments compare all three models on shared samples. See [Experiments and settings](docs/research-methods.md#experiments-and-model-choices).

## Outputs and diagnosis

`cache/` holds aligned arrays, `artifacts/` holds sample indices and resolved settings, `runs/` holds checkpoints and evaluation reports, and `predictions/` holds forecast rasters and previews. Saved metadata links these directories by absolute paths; keep them in place unless those references are migrated and checked. See [Output interpretation](docs/research-methods.md#outputs-and-diagnosis).

## Verification and research limits

Synthetic checks establish pipeline execution. The real-data spatial models were retrained and evaluated in [the November 18 run](runs/spatial_generalization/retrain_2026-11-18/README.md), using the previous geographic splits. A full real-data cross-year experiment remains separate and pending. See [Research limits](docs/research-methods.md#verification-and-research-limits) and [Validation](docs/validation.md).

Original imagery, saved models, research results and upstream reference code are retained. Earlier documentation is collected in [Historical project documentation](docs/history.md). AgriMatNet attribution is recorded in the [packaged AgriMatNet license](src/uav_ndvi/licenses/AgriMatNet-LICENSE).
