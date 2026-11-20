# AI review and presentation context

Read `Research/docs/presentation-brief.md` first. It is the self-contained presentation source: implemented methods, current results, historical/synthetic distinctions and proposal gaps.

- Current models are in `Research/src/uav_ndvi/models/`. The legacy standalone regression is a separate workflow; do not assign the current baseline scores to it.
- The actual split is 70/15/15 geographic row strips, with 20,000 sampled pixel-date examples per split. Acquisition dates are shared across partitions.
- Five seed results refer to 42–46, including one reused completed seed-42 run. Same-seed checkpoints or repeat evaluations are not additional independent repetitions.
- Mean test RMSE: transformer 0.090802118; OLS 0.090767573. Paired p=0.934104. No mean difference was detected. This does not establish equivalence or superiority.
- The t-test concerns seed variability conditional on fixed data, not inference across independent fields, years or datasets.
- OLS predicts every test sample as unfavorable; no transformer run predicts healthy samples. Include class imbalance and macro F1 when interpreting accuracy.
- Health categories are NDVI thresholds, not independently measured agronomic diagnoses.
- New post-hoc results are in `Research/analysis_results/five_seed_posthoc_2026-11-19/`. Read `presentation_summary.txt` and `ranking_metrics_summary.csv`. Binary ROC-AUC/AP at actual NDVI >=0.4 and >=0.6 are now complete; they use continuous scores, not hard labels or a calibrated three-class probability model.
- Historical self-attention was recovered for all five seeds, four layers and eight heads on 20,000 test samples per seed. Per-date summaries, per-head matrices and support counts are included. These describe historical-token mixing, not causal influence or direct future-target-to-history cross-attention.
- A full real-data date/cross-year holdout, causal temporal attribution and controlled ablations remain incomplete. The earlier RMSE p-value does not test the new AUC/AP differences.
- Synthetic smoke scores test workflow execution, not real strawberry performance.
- Explain differences from the proposal, including acquisition years and baseline features. Do not present proposed methods as completed if the implementation differs.
- Raw rasters, caches, trained research checkpoints and per-pixel predictions are intentionally absent. A public clone cannot reproduce real-data training without the original inputs.

For slides, use motivation, objectives, data, preprocessing, split, models, training, five-run results, paired test, diagnostics, limitations and conclusion. Keep implemented features, verified behavior, observations and future work distinct. Do not invent experiments or metrics.
