# Configuration guide

[Project overview](../README.md) | [Pipeline commands](../docs/workflows.md)

Run pipeline commands from the project root. Pass the same explicit configuration to preparation, sample construction and training when those stages depend on its settings.

| File | Purpose |
| --- | --- |
| [`../config.yaml`](../config.yaml) | Automatic root configuration when `--config` is omitted; currently the season-aware spatial experiment. |
| [`spatial_generalization.json`](spatial_generalization.json) | Explicit geographic holdout settings; requires season labels. |
| [`cross_year_2025.json`](cross_year_2025.json) | Explicit held-out 2025 settings, including conservative exclusions of late-2024 acquisitions. |
| [`smoke.json`](smoke.json) | Small model and sample settings for synthetic checks; not a final research configuration. |
| [`default.json`](default.json) | Earlier explicit base settings retained for historical commands. |
| [`../src/uav_ndvi/defaults.json`](../src/uav_ndvi/defaults.json) | Packaged base defaults used by the loader; also works when the package is installed outside this workspace. |

## How settings are resolved

1. `uav_ndvi.config` loads packaged defaults and adds the experiment/compatibility defaults defined in `config.py`.
2. With an explicit `--config FILE`, that file overrides the base settings. It does **not** also inherit the root `config.yaml`.
3. Without `--config`, the loader uses `config.yaml` if present in the current working directory. Otherwise it uses the base settings.

This distinction matters: the root and experiment configurations require explicit seasons; the base compatibility settings permit calendar-year fallback. The current `smoke.json` only overrides size/training settings, so passing it explicitly also uses that base season behavior. Configuration file names do not select a different parser: JSON and native YAML are accepted.

The old `default.json` and packaged `defaults.json` currently contain matching JSON values but serve different paths and workflows. Keep both available unless those consumers are migrated. Do not edit packaged defaults just to configure a single experiment; use a new experiment file instead.

Saved run configurations determine evaluation and forecasting behavior. Changing a configuration file after training does not change an existing checkpoint. Sample/cache compatibility checks still apply when constructing or training a new run.
