# Research methods and interpretation

[Project overview](../README.md) | [Commands](workflows.md) | [Repository map](repository-map.md)

All paths and commands below are relative to the project root. This guide preserves the detailed dataset assumptions, model settings, output interpretation and research limits from the project README.

## Dataset and provisional crop cycles

The supplied manifest has 43 acquisitions: 14 in 2022, one in 2023, 24 in 2024, and four in 2025. Paths are relative to the manifest directory; `--data-root` can override that base. Input rasters must have one NDVI band in physical [-1, 1] units, a CRS, and correct NoData metadata. Optional historical spectral covariates use `NAME_path` columns and `extra_features` configuration. No future measured covariates enter a forecast.

Edit `manifest.csv` columns `date,path,season_id,season_status`. Dates use YYYY-MM-DD. `season_id` identifies a crop cycle or a conservative within-cycle observation block, not a year. The supplied labels are explicitly provisional:

| Observations | Provisional ID |
| --- | --- |
| April?August 2022 | `provisional_2022_apr_aug` |
| August 2023 singleton | `provisional_2023_aug` |
| February?August 2024 | `provisional_2024_feb_aug` |
| September?December 2024 | `provisional_2024_sep_dec` |
| February?April 2025 | `provisional_2025_feb_apr` |

The possible late-summer 2024 boundary and the late-2024/2025 relationship are unverified. These blocks are analysis assumptions, not inferred planting records. Sequences never cross their boundaries. January 1 does not automatically define a crop cycle. Update IDs from farm records, including adding further boundaries within a block if needed; rebuild into fresh cache/example/run directories. Do not merge late 2024 with 2025 without confirmation. New experiment configurations require explicit labels. The legacy manifest reader can still warn and fall back to calendar years when `require_seasons=false`, solely for compatibility.

The cross-year configuration conservatively excludes all listed September?December 2024 dates from both targets and histories. This avoids fitting on a crop cycle that might continue into 2025. It supports an **unseen-year** claim; unconfirmed season labels cannot establish an **unseen-growing-season** claim. Even the provisional within-block continuity should be checked before final research conclusions. One 2023 acquisition cannot independently supply the default two historical observations; it therefore contributes no supervised examples.

## Alignment, missing data, and sampling

New preprocessing creates a north-up union of all acquisition footprints, snapped to the first raster's lattice (or `--reference`), retaining its CRS and measured pixel spacing. An explicitly configured `grid_extent=reference` reproduces reference-footprint clipping for compatibility. Union export extent can be larger than an individual source raster; no geographic coordinates or reference pixel spacing are discarded. Nearest-neighbor warping preserves observed NDVI values without temporal interpolation. Large raster differences should be inspected for registration errors; shared CRS alone does not guarantee accurate field registration.

The original cache spacing is about **0.0108023 m**, not the approximate 0.016 m stated in the dataset description. Inspect `metadata.json` and its alignment report to establish the correct resolution for the thesis. The code does not silently resample to an assumed spacing. Memory-mapped float32 arrays and tiled I/O limit memory use. Preprocessing reports estimated bytes in metadata and checks free disk space before allocating; a fine-resolution union across 43 dates may require many GiB and substantial processing time.

NoData, nonfinite values, out-of-range NDVI, and NDVI <=0.2 become NaN. Validity is retained per pixel/date. Zero is used only as a masked tensor placeholder, never as an observed NDVI. A pixel needs `min_history` valid observations in the same block, within `max_history_days`, among the last `max_history` acquisitions. Its horizon is measured from its own latest valid observation, not the latest field-wide acquisition. Targets must also be valid vegetation pixels for supervised evaluation; inference does not use target validity to decide where to predict. All usable aligned values remain in the cache, including dates with partial coverage.

A seeded uniform pixel-date reservoir caps each split at `samples_per_split` (default 20,000) for tractable training/evaluation. Coverage counts are computed over **all** valid targets before sampling, including acquisition dates without enough history. Geographic 70/15/15 fractions describe grid strips; irregular field coverage means eligible sample fractions need not equal those proportions. Increase the cap for final analyses. Overall metrics weight sampled pixel-date pairs equally; per-date reports expose unequal representation. Samples at neighboring pixels/dates are correlated, so ordinary independent-sample confidence intervals would be inappropriate.

## Experiments and model choices

Spatial generalization fixes 70%/15%/15% geographic row strips on the common grid for training/validation/testing across all eligible years. `split_axis=column` changes orientation; `buffer_pixels` adds a geographic exclusion buffer. A pixel location belongs to exactly one partition across dates and years. History always precedes its target and stays in its explicit season block. Dates recur across geographic partitions by design.

Cross-year generalization fits 2022 and usable 2023/2024 data, reserves a deterministic chronological tail of eligible 2024 dates for validation, and tests only 2025. With the supplied conservative exclusions, the automatically selected validation dates are the last 15% (rounded up) of the remaining 2024 acquisition dates. Resolved dates are saved in `examples.json` and checkpoints; explicit `validation_dates` can override this choice using training-year dates only. All validation dates are excluded from every historical input as well as training targets. Training targets in 2024 must also precede the earliest validation date, including when validation dates are set manually. Nothing is fitted or selected using 2025 NDVI. The frozen model can use earlier observed 2025 NDVI for later forecasts. With two required histories, the supplied 2025 block first permits targets on April 1 and April 14; February 27 and March 7 supply context. A history missing at an individual pixel can make that target ineligible.

Persistence predicts the last valid NDVI. Linear regression fits training-only standardized past-history features (last/first/mean/std, elapsed spans, trend, lead time, and target calendar features). It is a supervised regression baseline, separate from the preserved standalone legacy regression script. The transformer preserves AgriMatNet's dual encoders and fusion head, replacing sequence-index positions with actual elapsed days. Dates and day-of-year are the only future features.

Defaults: d_model=64, four layers per encoder, eight heads, dropout=0.1, Adam lr=3e-4, validation RMSE early stopping. New models predict residuals around persistence; their output head starts at zero residual. The point objective is median-output MSE plus 0.1 times the original quantile pinball loss, so point training aligns with RMSE and auxiliary intervals remain available. `loss=pinball` and `residual_prediction=false` reproduce the prior objective/output interpretation. These are configurable hypotheses to validate, not evidence the transformer will win. Old checkpoints lacking these settings retain their original behavior. All NDVI targets and predictions stay in native units, with no target normalization or inverse scaling. Linear feature normalization fits only training examples. Predictions are not silently clipped or quantiles sorted; out-of-range rates and quantile crossings are reported.

## Outputs and diagnosis

Evaluation uses exactly the same saved coordinates/targets for all three models. It writes overall, per-date, and exact-horizon RMSE/MAE/R?, accuracy, macro F1, per-class precision/recall/F1/support, confusion matrices, native-unit distributions, bias/spread/out-of-range diagnostics, scatter plots, horizon/date RMSE figures, physical pixel-center coordinates, and full-grid target coverage counts/percentages. R? is undefined for constant or singleton targets. Health labels are unfavorable <0.4, marginal [0.4,0.6), healthy >=0.6; macro F1 always includes all three classes. Quantile interval diagnostics are secondary and not calibrated guarantees.

Each forecast model directory contains GeoTIFFs for predicted NDVI, latest historical NDVI, history counts, forecast coverage, per-pixel horizon, and coverage reasons. With ground truth it also includes actual NDVI, signed error, absolute error, and target validity. Optional health and transformer quantile maps are retained. `coverage_reason.tif` codes: 0=no historical coverage, 1=insufficient usable history, 2=excessive horizon, 3=external mask exclusion, 4=forecastable. `coverage.json` reports counts and valid-target forecast percentage. PNG NDVI panels use the same 0?1 range; absolute errors use 0?0.5; gray means NoData. GeoTIFFs retain full numeric precision and the common grid's CRS, transform, extent, and spacing. Full-field maps include all eligible locations; only the saved held-out sample indices determine experiment metrics. Compare model panels with that distinction in mind.

The preserved earlier confusion matrices show **every one of 20,000 predictions was unfavorable** for both fitted models. Actual NDVI std=0.0901, linear std=0.0216, transformer std=0.0242: strong underdispersion consistent with meanward predictions and low R?. Neither model reached 0.4. On those same preserved 20,000 samples, a read-only reconstruction of persistence gives RMSE=0.121254 and R?=-0.809882 (`docs/existing_audit.json`). These are audits of the old experiment, not results for the new splits. There is no target scaling/inverse-scaling step in the inspected implementation. These findings do not establish that architecture alone caused the weak predictions.

A read-only audit of `predictions/june30` found 2,125,161 predictions out of 49,000,000 grid cells; the remaining 46,874,839 are NoData. Of those, 28,853,584 have no usable history and 18,021,255 fail history eligibility. There were no unexplained missing pixels under the old rules. June 30, 2025 is not an observed manifest date, so this map cannot be ground-truth scored. The audit is saved in `docs/existing_audit.json`; repeat into a new report with:

```powershell
python -m scripts.audit_existing --out docs/existing_audit_repeat.json
```

## Verification and research limits

See `docs/validation.md` for executed checks and their scope. Synthetic smoke runs under each experiment directory establish pipeline execution only; they are not strawberry results. The full new union-grid multi-year experiments must be trained and evaluated before reporting comparative performance. No new scientific performance claim is made here. In particular, few 2025 acquisitions, unverified crop cycles, partial coverage, spatial autocorrelation, fine-resolution storage demands, and large forecast gaps limit conclusions. Hyperparameters and partition choices must be finalized using training/validation data before inspecting final 2025 results.

Original code is retained under `upstream/ndvi-forecasting`; its MIT license is copied at `src/uav_ndvi/licenses/AgriMatNet-LICENSE`. Adaptation details are recorded in `docs/upstream-review.md`.
