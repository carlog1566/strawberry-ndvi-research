# Repository map

`Research/` is now the project root. Run commands from this folder. The existing virtual environment is the only runtime dependency left in the parent directory (`../.venv/`). Saved JSON paths were migrated; `.relocation.json` lets existing checkpoint metadata resolve to this root without changing saved weights.

Use `src/uav_ndvi/` for current pipeline development. The root `models/` directory contains a separate, older standalone regression workflow; `upstream/` contains the original reference project.

## Current pipeline

| File under `src/uav_ndvi/` | Responsibility |
| --- | --- |
| `__init__.py` | Package identity. |
| `__main__.py` | CLI: prepare/preprocess, examples, train, evaluate, forecast. |
| `config.py` | Load defaults and JSON/YAML overrides; validate settings. |
| `defaults.json` | Packaged defaults available outside the project root. |
| `preprocessing.py` | Read manifests, align rasters, mask invalid values, write/open caches. |
| `dataset.py` | Season histories, partitions, sample indices, tensors and batch padding. |
| `models/transformer.py` | Adapted transformer, elapsed-day encoding and quantile loss. |
| `models/linear_regression.py` | Current regression features and fitted scikit-learn pipeline. |
| `train.py` | Batching, model fitting and checkpoint loading. |
| `evaluate.py` | Persistence comparison, metrics, classifications and diagnostic plots. |
| `forecast.py` | Tiled forecast maps, coverage, errors and PNG previews. |
| `utils.py` | Shared reproducibility and tensor-device helpers. |
| `legacy_linear_regression.py` | Separate standalone regression implementation, preserved for the earlier workflow. |
| `licenses/AgriMatNet-LICENSE` | Canonical adaptation license, linked by documentation and included in distributed packages. |

The following files expose existing implementations through additional public import paths. They are intentionally small; keep both import paths working during any future restructuring.

| Public import file | Implementation it exposes |
| --- | --- |
| `model.py` | `models/transformer.py` |
| `baseline.py` | `models/linear_regression.py` |
| `models/__init__.py` | Models subpackage identity. |
| `preprocess.py` | `preprocessing.py` |
| `training.py` | `train.py` and shared helpers from `utils.py` |
| `evaluation.py` | `evaluate.py` |
| `inference.py` | `forecast.py` |

## Project inputs, tools and reference material

| Location | Responsibility |
| --- | --- |
| `uav_ndvi/__init__.py` | Root import shim adds `src/uav_ndvi` to the package search path. Enables running without installation. |
| `pyproject.toml` | Package build metadata, dependencies, CLI entry point and bundled resources. |
| `requirements.txt` | Python dependency declarations consumed by packaging. |
| `pytest.ini` | Restricts tests to this project, excluding upstream evaluation scripts. |
| `tests/test_pipeline.py` | Synthetic raster, dataset, model and end-to-end tests; synthetic data generator. |
| `tests/test_experiments.py` | Experiment isolation, union-grid, configuration and old-checkpoint checks. |
| `tests/__init__.py` | Supports imports of shared fixtures from `tests.test_pipeline`. |
| `scripts/audit_existing.py` | Audit retained real outputs without retraining. |
| `scripts/repeated_comparison.py` | Repeated fixed-split fits, paired RMSE scores, averages and a paired t-test. |
| `tests/test_repeated_comparison.py` | Statistical calculation, pairing, duplicate-seed and output-preservation checks. |
| `config.yaml` | Root configuration automatically loaded when no explicit config is provided. |
| `configs/default.json` | Earlier explicit defaults used by archived workflow instructions. |
| `configs/smoke.json` | Small synthetic workflow settings. |
| `configs/spatial_generalization.json` | Geographic partition experiment. |
| `configs/cross_year_2025.json` | Held-out 2025 experiment. |
| `manifest.csv` | Current acquisition inventory with provisional season labels. |
| `ndvi_manifest.csv` | Older inventory using the standalone script's `with_soil` schema. |
| `dataset/` | Original NDVI imagery referenced by manifests. |
| `models/LinearReg_Baseline.py` | Compatibility launcher and function imports for `src/uav_ndvi/legacy_linear_regression.py`. |
| `ndvi_linear_model.joblib`, `ndvi_linear_01m.joblib`, `ndvi_linear_0016m.joblib` | Retained standalone regression artifacts; filename alone does not establish redundancy. |
| `upstream/ndvi-forecasting/` | Original AgriMatNet source, weights, documentation, license and nested Git history. |
| `../.venv/` | Existing Python interpreter and installed dependencies, retained in the parent directory. |
| `.relocation.json` | Previous project locations used to resolve paths embedded in saved checkpoints. |
| `.gitignore` | Excludes generated files if the project is placed under Git. |

## Generated outputs and research history

| Location | Responsibility |
| --- | --- |
| `cache/` | Aligned NDVI arrays plus metadata; saved examples depend on these paths. Also contains Matplotlib's font cache. |
| `artifacts/` | Saved example indices, synthetic inputs and resolved experiment settings. |
| `runs/` | Trained models, run settings, histories, evaluations and figures. |
| `predictions/` | Forecast rasters, previews, errors and coverage reports; some rasters have GIS sidecars. |
| `docs/upstream-review.md` | Original model review and adaptation decisions. |
| `docs/workflows.md` | Preparation, training, evaluation and forecast commands. |
| `docs/research-methods.md` | Dataset assumptions, split definitions, model settings and output interpretation. |
| `docs/repeated-comparison.md` | Five-seed comparison protocol, commands and interpretation limits. |
| `docs/presentation-brief.md` | Self-contained presentation source covering models, experiments, results, software tests and proposal gaps. |
| `docs/legacy-regression.md` | Differences, commands and saved artifacts for the standalone regression workflow. |
| `configs/README.md` | Configuration selection and default-loading behavior. |
| `docs/validation.md` | Prior validation evidence and research limitations. |
| `docs/history.md` | Consolidated earlier README and validation report; replaces the two approved original archive files. |
| `docs/existing_audit.json` | Consolidated audit of previous real research outputs, including persistence metrics. |

Keep generated inputs and outputs connected: `runs/*/run.json` stores an absolute examples path, and `artifacts/*/examples.json` stores an absolute cache path. Manifests resolve relative raster paths from their own directory. Moving these files requires updating and validating the corresponding references.

## Organization sequence

1. Use this map and the root README to distinguish current code, legacy code, upstream reference material and generated results.
2. Review the exact deletion candidates in [Cleanup review](cleanup-review.md). Delete only approved files.
3. Current implementations live in `models/`, `preprocessing.py`, `train.py`, `evaluate.py`, `forecast.py` and `utils.py`. Former module names remain as compatibility imports; the CLI uses current modules directly.
4. Relocate legacy artifacts or generated experiment folders only with an explicit path migration and successful evaluation/forecast checks. They are not disposable solely because they are old or synthetic.
