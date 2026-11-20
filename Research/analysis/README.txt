SEPARATE FIVE-SEED POST-HOC ANALYSIS

From the Research directory, using the existing parent environment:

python -B -m analysis.five_seed_posthoc --out analysis_results/five_seed_posthoc_2026-11-19

Use a different fresh output directory to repeat this analysis. Existing outputs
are never overwritten. Do not add --attention-limit for the full test analysis;
that optional flag explicitly restricts attention to a prefix for diagnostics.

New isolated tests (no model fitting):
python -B -m pytest analysis/test_five_seed_posthoc.py -q -p no:cacheprovider

Files and meaning:
- report.html: presentation-oriented summary, figures and interpretation limits.
- presentation_summary.txt: copy-ready metric definitions, results and scope.
- ranking_metrics_summary.csv: per-threshold/model mean and sample SD over seeds.
- ranking_metrics_all_seeds.csv: every seed/model/threshold score and availability.
- seed_XX/*_roc.csv and *_precision_recall.csv: complete saved curves.
- seed_XX/attention/: historical attention tables, support counts and illustrations.
- attention_date_seed_summary.csv: head/layer-averaged date attention across seeds.
- healthy_precision_recall_zoom.png: precision axis 0-0.05; full curves are also saved.
- input_integrity.json and protected_inputs_before.json: original-file audit.
- analysis_protocol.json: definitions, version information and settings.
- availability.json: missing inputs or failed attention extraction, if any.

The positive target is actual NDVI >= 0.4 (marginal-or-healthy) or >= 0.6
(healthy). Scores are the unmodified continuous point NDVI predictions saved by
the previous experiment. These are two binary tasks, not a three-class AUC.
Average Precision is sklearn's AP, not trapezoidal area under the PR curve.
The PR baseline is the positive prevalence, especially important for rare healthy
targets. Means/SDs concern fixed-data training-seed variability.

Historical attention is reconstructed by observational Q/K/V hooks and separate
equal-weight attention copies. Original model forward methods and parameters are
not replaced. Ordinary/instrumented/saved predictions are checked, masks are
respected, hooks are removed, and checkpoint/state hashes must remain unchanged.
Illustrative NPZ files retain all heads; illustrative heatmaps average heads by
layer. NPZ axes are [layer, head, query_date, key_date]. Full-test attention is
streamed into per-date and per-date-pair summaries.

Uniform-relative attention divides received mass by the uniform 1/history_length
baseline within each example. Acquisition support is reported. Dates absent from
valid histories are unavailable, not zero-importance. Pairwise means condition
on co-occurrence and are not necessarily globally row-normalized.

Attention describes historical-to-historical mixing, not causal importance or
target-to-history cross-attention. Residuals, values, feedforward blocks, pooling,
future-date features and the persistence anchor also affect final predictions.
No intervention, new data split, preprocessing change, model tuning, retraining
or change to existing experimental results is part of this analysis.
