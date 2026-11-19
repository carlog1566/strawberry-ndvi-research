# Executed validation ? November 13, 2026

The pre-change pipeline passed 7 tests. The expanded suite passed 14 tests using the existing Windows virtual-environment interpreter. The environment itself was not installed into or edited. The previous validation report is preserved in [Historical project documentation](history.md#earlier-validation-report).

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite exercises alignment and a union-grid shifted edge, NaN/NoData and soil exclusion, partial coverage, actual elapsed days and variable sequence lengths, same-season boundaries, spatial isolation, explicit season_id handling, held-out 2024 dates absent from all histories, 2025 values absent from training inputs, frozen rolling-origin model files, finite model inference, padding invariance, original checkpoint loading, optional historical features, metrics for identical three-model samples, and GeoTIFF CRS/transform/shape/mask/error export. A final chronological-validation guard exposed two test-fixture issues (an earliest possible validation date left no training targets, and a misplaced assertion); both fixtures were corrected and the full suite rerun successfully. The end-to-end tests train two-epoch small transformers on synthetic data. Pending deprecation warnings from Affine multiplication and Matplotlib colormaps do not fail the suite.

Separate root CLI workflows completed successfully for BOTH experiments, using shifted synthetic GeoTIFFs, two epochs, d_model=8, one layer, two heads, and 30 samples per partition. Each workflow ran `examples`, `train`, `evaluate`, and `forecast --model all --classification`, following a shared `prepare` command. Retained artifacts:

- Inputs and configuration: `artifacts/research_validation/`.
- Union cache: `cache/research_validation/`.
- Final model checkpoints, history curves, metrics and diagnostics: `runs/spatial_generalization/cli_validation/` and `runs/cross_year_2025/cli_validation/`.
- Three-model raster maps: `predictions/research_validation/spatial_generalization/` and `predictions/research_validation/cross_year_2025/`.
- An earlier synthetic smoke pass remains under each experiment's `smoke/` directory; it predates zero-initializing the residual head. It is retained rather than overwritten.

These synthetic checks establish execution and data contracts; they do not estimate strawberry forecasting performance. Synthetic dates differ from the real manifest and are not research acquisition dates.

Read-only audits of the preserved real results and June 30 forecast were run with `python -m scripts.audit_existing`. The consolidated report is `docs/existing_audit.json`. It includes the expanded audit, which reconstructs persistence on exactly the saved 20,000 old evaluation coordinates/dates without retraining or rewriting previous results. Original linear/transformer metrics match the user's reported values. All unexplained missing forecast pixels count to zero under the old history rules. The original `runs/strawberry/transformer.pt` successfully loads with its historical non-residual output interpretation.

The full new union cache and full-size strawberry experiments have NOT been executed. Old cached values use a clipped reference footprint and calendar-year season assumptions; the audit must not be represented as validating new crop-cycle or cross-year claims. No new performance improvement is claimed. Farm-confirmed crop-cycle labels, footprint/registration inspection, adequate disk space, full training, validation-based model selection, and final frozen 2025 evaluation remain necessary for the scientific study.

## Organization checks - November 18, 2026

After source organization, the existing Windows interpreter passed all 16 tests in 11.02 seconds with the same 126 Affine/Matplotlib deprecation warnings. The command was `python -B -m pytest -q -p no:cacheprovider`, avoiding regeneration of housekeeping caches. The two added checks cover identity of old/new imports and resolution of serialized references using former module paths.

Complete synthetic CLI workflows passed for both `spatial_generalization` and `cross_year_2025`: shared `prepare`, then `examples`, `train`, `evaluate`, and `forecast --model all --classification` for each experiment. These used two epochs, d_model=8, one layer, two heads, and 30 samples per partition. Outputs were written outside the project under `<USER_HOME>/AppData/Local/Temp/uav_ndvi_organized_cli_sgtpvebd`, retaining project research outputs unchanged. Temporary outputs are machine-local verification evidence, not durable research artifacts.

Root and all subcommand/alias help screens passed, along with the standalone legacy regression help screen. The old strawberry checkpoint still produced exactly the original 24 quantile values across eight saved test samples. This comparison does not establish scientific forecasting performance. Current documentation links were checked for local targets.

## Standalone regression organization - November 18, 2026

The latest suite passed 17 tests in 15.79 seconds using `python -B -m pytest -q -p no:cacheprovider`, with the same 126 deprecation warnings. In addition to existing current-pipeline checks, the new legacy integration test trains through `models/LinearReg_Baseline.py` from an external working directory without installation. Prediction through that launcher and `python -m uav_ndvi.legacy_linear_regression` produced equal raster profiles, masks and pixel values using the synthetic checkpoint. Real data/checkpoint paths were unchanged. Current source-package license resolution, its package-data declaration and 45 local documentation links were also checked.

## Research folder migration - November 18, 2026

The project was moved into `Research/`, including source, tests, configuration, input rasters, caches, saved models, output maps, documentation and the upstream repository with its Git history. The existing `.venv` remains in the parent directory to preserve its installed launchers. Use `..\.venv\Scripts\python.exe` from Research, or activate the parent environment.

Absolute paths in 21 JSON metadata files were updated. Checkpoint binaries were preserved; `.relocation.json` and the loader normalize their embedded metadata paths. All seven saved transformer checkpoints were loaded and their metadata matched their relocated test datasets. Both manifests resolved all raster files. Root CLI help and all local guide links passed. The complete suite passed all 17 tests in 15.87 seconds with the same 126 existing deprecation warnings.

## Real-data spatial retraining - November 18, 2026

Retraining completed at `runs/spatial_generalization/retrain_2026-11-18`, preserving the previous checkpoints. Both fitted models used `artifacts/spatial_generalization/v1` and its real-data `cache/season_blocks_v1` cache: 20,000 samples per partition, disjoint geographic pixel locations, 70/15/15 row strips, and dates shared across partitions. This is a spatial comparison, not unseen-date evaluation. The existing configuration used seed 42, d_model=64, four layers per encoder, eight heads, residual predictions and MSE plus auxiliary quantile loss. CPU execution used four threads.

The transformer early-stopped after 48 epochs and selected epoch 36 by minimum validation RMSE (0.087416). Linear regression was fitted only on the same training samples. Validation and test evaluations completed, writing comparison tables, per-date/horizon reports, predictions, interval diagnostics, plots and provenance. On the 20,000 test samples, RMSE/MAE/R2 were persistence 0.128357/0.100304/-0.814347, linear 0.090768/0.074245/0.092718, and transformer 0.089750/0.072664/0.112942. No significance claim is made for the small fitted-model difference.

Both fitted artifacts reloaded successfully and matched their evaluation CSV predictions on eight checked test samples. The selected checkpoint epoch matched the validation minimum in history.csv. Full protocol, results and limitations are in [the run summary](../runs/spatial_generalization/retrain_2026-11-18/README.md). The earlier statements about unexecuted real-data experiments predate this spatial retraining; a full real-data cross-year run is still separate and pending.

## Five paired seed repetitions - November 18, 2026

Completed `runs/model_comparison/five_seed_2026-11-18` using seeds 42-46 on the unchanged saved spatial examples. The completed seed-42 retraining was reused once; seeds 43-46 were new full fits of both models with validation-selected transformer checkpoints. Two separate CPU processes ran concurrently, each using four threads. Initial tests passed all 25 cases, including eight new statistical/pairing checks; a separate two-seed synthetic workflow verified parallel training, evaluation and summary generation. No repeated real-data run was a synthetic smoke model.

Mean test RMSE across five pairs was transformer 0.090802118 (sample SD 0.000877748) and linear 0.090767573 (SD 0). The two-sided paired test gave t(4)=0.088003036, p=0.934103998, and a 95% CI for mean transformer-minus-linear RMSE of [-0.001055323, 0.001124413]. It did not detect a mean difference at alpha=0.05. This does not establish equivalence.

All five pairs had distinct seeds, matching non-seed configurations and identical 20,000 test coordinates/targets. Sample-file hashes remained unchanged. Recomputed run-level RMSEs matched saved scores; the equivalent one-sample test on paired differences independently reproduced the statistic and p-value. Deterministic linear scores were identical across all five fits. The uncertainty concerns initialization/training randomness conditional on this fixed dataset; no generalization-to-new-fields/years claim follows. The test set had already been inspected, so this is exploratory. With five pairs, approximate normality of seed-level differences is not established.

See [comparison report and plot](../runs/model_comparison/five_seed_2026-11-18/comparison.md), [per-run scores](../runs/model_comparison/five_seed_2026-11-18/paired_scores.csv), and [method](repeated-comparison.md).
