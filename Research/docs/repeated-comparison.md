# Repeated transformer versus regression comparison

[Project overview](../README.md) | [Research methods](research-methods.md)

The five-run experiment uses the existing spatial split and saved samples. Both models see identical training samples and are evaluated on the same 20,000 held-out-location test samples. The geographic 70/15/15 partition is unchanged; acquisition dates remain shared across partitions.

## What changes between runs

Seeds 42, 43, 44, 45 and 46 control transformer initialization, batch shuffling and dropout. Architecture, loss, samples, feature construction and early stopping remain fixed. Each transformer checkpoint is selected using validation RMSE. Each pair also fits the linear baseline on exactly those training samples.

The completed `retrain_2026-11-18` seed-42 run is reused as one pair. Seeds 43-46 are new fits. The two older seed-42 checkpoints must not be counted as separate independent seed repetitions. The runner rejects duplicate seeds, mismatched configurations and mismatched test samples/targets.

Ordinary least-squares regression is deterministic on fixed data. Its five scores should therefore be identical, with zero standard deviation. Refitting it five times does not create five independent datasets. Variation in the paired differences comes from transformer training randomness.

## Scores, averages and test

One repetition contributes one transformer test RMSE and one linear test RMSE. Average these five RMSE scores separately and report their sample standard deviations. Do not average predictions first: that would evaluate an ensemble, which is a different comparison.

For each pair, the difference is transformer RMSE minus linear RMSE. Negative differences favor the transformer. The two-sided paired t-test uses the five differences, so its degrees of freedom are four. The report includes the mean difference, its 95% confidence interval, the t statistic and the p-value at alpha=0.05. A zero-variance difference vector is reported as undefined rather than fabricating a p-value.

The test does not treat the 20,000 neighboring pixel-date errors as independent repetitions. Its scope is training-seed variability conditional on the fixed data. It does not establish uncertainty across new fields, seasons, years or independent datasets. With five pairs, normality of the seed-level differences is difficult to assess. The existing test set has already been inspected, so this is an exploratory comparison.

If the intended repetitions instead use different training/test splits, agree on a spatially appropriate split design and inference procedure first. Ordinary paired t-tests on overlapping resampled splits need special care; changing splits is not equivalent to changing initialization seeds.

## Execute and inspect

From Research with the parent virtual environment activated:

```powershell
python -m scripts.repeated_comparison run --examples artifacts/spatial_generalization/v1 --config configs/spatial_generalization.json --seeds 42 43 44 45 46 --reference-run runs/spatial_generalization/retrain_2026-11-18 --cpu-threads 4 --jobs 2 --out runs/model_comparison/five_seed_2026-11-18
```

This command was launched for the five-run experiment; use a fresh output directory to repeat it. `--jobs 2` runs two separate training processes at a time. Each worker uses four CPU threads. Individual training logs are under `logs/`; each seed directory contains checkpoints, history and evaluation results. `protocol.json` records the settings and sample-file hashes before training.

Final reports are `paired_scores.csv`, `paired_test.json` and `comparison.md` in the comparison directory. To analyze completed runs again without retraining, use the `summarize` subcommand with all five run paths and a fresh `--out` directory. The analyzer recomputes each RMSE from the paired saved predictions.

## Statistical references

- [NIST: analysis of paired observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) describes inference from paired differences.
- [SciPy: ttest_rel](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html) documents the implemented test and confidence interval.
- [Dietterich, 1998](https://doi.org/10.1162/089976698300017197) discusses the problems with uncorrected paired tests across random train/test splits.
