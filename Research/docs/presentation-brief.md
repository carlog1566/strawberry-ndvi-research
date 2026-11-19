# Presentation brief: repository work, results and remaining gaps

Prepared November 19, 2026, from the source code, saved experiments and supplied proposal: **NDVI-Based Crop Health Forecasting for Strawberry Fields Using UAV Multispectral Imagery and Transformer-Based Models**.

This document is self-contained input for generating a presentation. Results distinguish software checks, synthetic experiments, historical real-data experiments and the current five-seed comparison.

## Instructions for ChatGPT creating the presentation

Use the verified facts below to create slides and speaker notes. Make the current five-seed real-data comparison the main result. Explain model inputs, spatial sampling, training and the paired test. Include classification weaknesses and unfinished proposal objectives. Do not invent AUC-ROC, attention analyses, agronomic labels, temporal-holdout results or evidence that the transformer is superior. Historical and synthetic results belong in background or an appendix. Use NDVI units for regression error, not “percentage accuracy.”

## 1. Research purpose and completed work

The project predicts a pixel's later NDVI using its earlier UAV-derived NDVI observations in a strawberry field in Pomona. The motivation is earlier information about changes in vegetation condition. It compares an adapted AgriMatNet transformer with ordinary least-squares linear regression and a persistence baseline.

Implemented work includes:

1. Dated-raster manifests and explicit provisional season/block labels.
2. Raster alignment across different acquisition footprints and missing-data masks.
3. Per-pixel, variable-length histories using actual acquisition-day gaps.
4. Geographic train/validation/test regions and saved sample indices.
5. Transformer and regression fitting on the same training samples.
6. Validation-based transformer checkpoint selection and common test evaluation.
7. NDVI, derived health-category, error, coverage and uncertainty outputs.
8. Synthetic end-to-end tests, historical-output audits and compatibility checks.
9. Five training-seed repetitions, average scores and a paired t-test.
10. Repository consolidation into Research while preserving saved models and historical import paths.

This is **per-pixel temporal modeling**. The current transformer does not ingest a spatial image patch or learn a convolution over neighboring pixels.

## 2. Dataset and preprocessing

The Python pipeline starts from already-computed single-band NDVI GeoTIFFs. The proposal discusses computing NDVI from NIR and red bands in external UAV/ArcGIS processing. Do not describe the model pipeline as reconstructing orthomosaics or computing indices from raw multispectral bands itself.

NDVI = (NIR - Red) / (NIR + Red).

The manifest selects 43 acquisitions:

| Year | Acquisitions |
| --- | ---: |
| 2022 | 14 |
| 2023 | 1 |
| 2024 | 24 |
| 2025 | 4 |

This is broader than the proposal's stated 2024–2025 scope. The isolated 2023 block cannot supply sufficient history and contributes no current supervised examples by itself.

The provisional observation blocks are April–August 2022, August 2023, February–August 2024, September–December 2024, and February–April 2025. These are analysis assumptions, not farm-verified planting/harvest cycles. Histories do not cross block boundaries.

Preprocessing:

- Requires georeferencing and single-band input rasters.
- Aligns acquisitions to a north-up union grid snapped to the reference lattice. The current real cache is 7,599 by 7,599 cells.
- Retains measured spacing of approximately 0.0108023 m, about 1.08 cm. Reconcile this with the approximate 0.016 m dataset description before presenting resolution.
- Uses nearest-neighbor alignment, not temporal interpolation.
- Stores float32 values in a disk-backed memory-mapped array; the current 43-date cache is approximately 9.93 GB in decimal units.
- Excludes declared NoData, nonfinite/out-of-range values and configured NDVI <= 0.2. The cutoff is an experimental vegetation-selection rule, not independent crop-health diagnosis.
- Uses zero only as a masked tensor placeholder, not as an observed replacement for missing NDVI.

Targets and predictions retain native NDVI units; there is no learned target normalization or inverse scaling.

Current example eligibility requires a valid target, at least two valid earlier NDVI observations in the same block, no more than 32 retained acquisitions, a history window of 365 days, and a maximum forecast horizon of 120 days from that pixel's latest valid observation.

Only target-date/time-of-year information is available as future input. No future NDVI, measured future weather or future spectral observation enters prediction. Optional historical spectral channels are supported, but the reported experiments set extra_features to an empty list.

## 3. Split and sampling protocol

The completed real-data experiment measures **spatial generalization**:

- First 70% of grid rows: training region.
- Next 15%: validation region.
- Final 15%: test region.
- Each pixel location stays in one partition across dates; disjointness was checked.
- Current buffer between strips: zero pixels.

Percentages describe grid regions, not percentages of the final sampled examples or guaranteed percentages of usable field area. Different footprints and missing coverage produce these counts:

| Partition | Eligible pixel-date pairs before sampling | Saved samples |
| --- | ---: | ---: |
| Training | 24,776,859 | 20,000 |
| Validation | 5,269,011 | 20,000 |
| Test | 363,220 | 20,000 |

Sampling is a seeded uniform pixel-date reservoir within each partition. It is not class-balanced or balanced by date. All models receive the same saved test targets/coordinates; some dates have very few examples.

**Acquisition dates are shared across partitions.** April 1 and April 14, 2025 occur as training targets at training-region pixels, as well as targets in other regions. Configuration fields named train_years/test_year do not make spatial mode chronological; the experiment mode controls partitioning.

Every individual example predicts a later observation from earlier history, but the fitted model can have seen other pixels from the target date. These results therefore do not demonstrate performance on an acquisition date completely absent from training.

## 4. The models and how they work

### Persistence

Prediction = the latest valid historical NDVI at that pixel. It has no fitted parameters or training stage. It tests whether learning improves on a “no change” forecast and uses the same eligible test examples as the fitted models.

### Current supervised linear regression

Source: src/uav_ndvi/models/linear_regression.py.

The model is ordinary least squares, not Ridge, logistic regression or a neural network. It converts a variable-length history into 12 engineered features in the NDVI-only experiment:

1. Latest valid NDVI.
2. Earliest retained valid NDVI.
3. Historical NDVI mean.
4. Historical NDVI standard deviation.
5. Number of valid observations.
6. Elapsed history span.
7. Lead time from the latest valid observation to the target.
8. Historical NDVI trend per day.
9. Trend multiplied by lead time.
10. Latest NDVI multiplied by lead time.
11. Sine of target day-of-year.
12. Cosine of target day-of-year.

A StandardScaler is fitted on training features only, then scikit-learn LinearRegression learns coefficients and an intercept by least squares. Targets remain native NDVI. Current comparison predictions are not clipped.

This differs from the proposal's description of flattened historical values and accumulated/running-sum features. The change must be described accurately.

OLS is deterministic on fixed data. Refitting it in the five repetitions produces identical predictions and zero variation in its scores.

### Adapted transformer: AgriMatNetUAV

Source: src/uav_ndvi/models/transformer.py.

The purpose is to learn nonlinear relationships among earlier NDVI observations while representing irregular time gaps and handling missing/variable-length histories.

The architecture was adapted from the ArCo Group AgriMatNet repository. Reported fits train from scratch; they do not initialize from the upstream pretrained satellite checkpoint.

Information flow:

1. Historical tokens contain NDVI, a missing-value indicator and calendar sine/cosine. A linear projection maps them into a 64-dimensional embedding.
2. Sinusoidal positional encoding uses actual elapsed days since the first retained acquisition. A three-day and a forty-day gap have different representations.
3. The history encoder processes the sequence with padding/missing masks. A masked mean summarizes valid historical embeddings.
4. A separate encoder processes one target-date token containing known timing features. It receives no measured future NDVI. A single-token future branch has limited temporal-attention work within that branch.
5. Historical and future embeddings are concatenated and passed through a dropout/linear/ReLU prediction head.
6. The head outputs the 0.1, 0.5 and 0.9 quantiles; the median is the point prediction.
7. Current predictions are residuals around persistence: latest valid NDVI plus learned residual. Zero initialization of the final head makes the initial prediction start at persistence.

This is a **dual-encoder fusion architecture**, not an encoder-decoder cross-attention network. Actual-day encoding is an adaptation documented in this repository. The inspected upstream positional code uses sequence-index positions; do not claim the actual-day implementation was inherited unchanged from that code.

The model predicts NDVI/quantiles, not trained three-class health probabilities. Categories are assigned afterward. Attention weights have not yet been exported/analyzed to substantiate which flights matter most.

### Older standalone linear-regression workflow

Source: src/uav_ndvi/legacy_linear_regression.py. The original models/LinearReg_Baseline.py path remains a launcher and compatibility import.

This is separate from the baseline in the main comparison. It uses two earlier acquisitions and nine features: latest/previous NDVI, trend, acquisition gap, lead time, target calendar sine/cosine and two lead interactions. It uses a configurable meter-resolution reference grid with average aggregation and a 60/20/20 south/middle/north spatial split. Predictions are clipped to [-1, 1]. It does not enforce the current explicit season blocks.

Three standalone checkpoints remain: ndvi_linear_model.joblib, ndvi_linear_01m.joblib and ndvi_linear_0016m.joblib. Their hashes differ. Filenames alone do not verify resolution or equivalence. No separate reproducible saved metric report for those exact three artifacts was identified; do not assign the current baseline's scores to them.

### Upstream reference implementation

upstream/ndvi-forecasting preserves original source, weights, license and Git history. Its presence does not mean an unmodified upstream model was evaluated on this UAV dataset. Retaining architecture/source provenance is different from pretrained transfer learning.

## 5. Current training settings

| Setting | Value |
| --- | --- |
| Embedding dimension | 64 |
| Encoders | Separate history and target-date encoders |
| Layers | Four per encoder |
| Attention heads | Eight |
| Feedforward dimension | 256 |
| Activation / normalization | ReLU; pre-normalization |
| Dropout | 0.1 |
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Batch size | 256 |
| Maximum epochs | 100 |
| Early-stopping patience | 12 epochs without validation-RMSE improvement |
| Point objective | Median-output MSE + 0.1 times quantile pinball loss |
| Quantiles | 0.1, 0.5, 0.9 |
| Output | Residual added to latest valid NDVI |
| Checkpoint selection | Minimum validation RMSE |
| Execution | CPU; CUDA unavailable; four PyTorch threads per worker |

The regression is fitted on the same training examples after transformer fitting. Validation/test examples do not fit its scaler or coefficients. Current stages preserve earlier results by using fresh output directories.

## 6. Main real-data results: five paired seeds

Five repetitions were requested. No alternative split design was specified, so the existing spatial split was held fixed and seeds 42–46 were used. The completed seed-42 retraining supplies one pair; seeds 43–46 are new fits of both models. Seeds change transformer initialization, shuffling and dropout, not sample membership. Earlier additional seed-42 checkpoints are not independent repetitions in this analysis.

Two training processes ran concurrently, each with four CPU threads. All five pairs use the same 20,000 test examples. Each transformer's checkpoint is selected from validation performance independently.

### Individual runs

| Seed | Epochs completed | Selected epoch | Best validation RMSE | Transformer test RMSE | Linear test RMSE |
| --- | --- | --- | --- | --- | --- |
| 42 | 48 | 36 | 0.087416 | 0.089750 | 0.090768 |
| 43 | 34 | 22 | 0.088515 | 0.092037 | 0.090768 |
| 44 | 49 | 37 | 0.087898 | 0.090904 | 0.090768 |
| 45 | 51 | 39 | 0.087806 | 0.091106 | 0.090768 |
| 46 | 52 | 40 | 0.087427 | 0.090213 | 0.090768 |

### Means across the five runs

These are arithmetic means of run-level metrics, not scores of an ensemble made by averaging predictions. Lower RMSE/MAE is better; higher R²/F1/accuracy is better. Accuracy is a fraction: 0.7775 means 77.75%.

| Split | Model | Mean RMSE | Mean MAE | Mean R-squared | Mean macro F1 | Mean accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| validation | transformer | 0.087812 | 0.070837 | 0.055845 | 0.302160 | 0.804620 |
| validation | linear | 0.088023 | 0.071960 | 0.051330 | 0.297568 | 0.806200 |
| validation | persistence | 0.122718 | 0.095389 | -0.843900 | 0.357819 | 0.693450 |
| test | transformer | 0.090802 | 0.073590 | 0.091959 | 0.296793 | 0.777500 |
| test | linear | 0.090768 | 0.074245 | 0.092718 | 0.292051 | 0.779600 |
| test | persistence | 0.128357 | 0.100304 | -0.814347 | 0.364971 | 0.671800 |

Test-RMSE sample SD is 0.000877748 for the transformer and 0 for OLS. Persistence also has no fitted randomness. Both fitted models have lower regression errors than persistence on this fixed split. Persistence has higher macro F1, so do not claim the fitted models beat it on every metric.

The single seed-42 result initially favored the transformer: RMSE 0.089750 versus 0.090768. The repeated-run average does not: transformer 0.090802 versus OLS 0.090768. This illustrates why the five-run result should lead the presentation.

### Paired t-test

One pair is the two RMSEs from one training seed. The difference is transformer RMSE minus OLS RMSE. Negative values favor the transformer. The t-test uses five run-level differences, **not 20,000 correlated pixel errors**.

| Quantity | Result |
| --- | --- |
| Pairs / degrees of freedom | 5 / 4 |
| Null hypothesis | Mean paired RMSE difference is zero |
| Alternative | Two-sided: mean difference is not zero |
| Alpha | 0.05 |
| Mean difference | +0.0000345448 NDVI RMSE units |
| t statistic | 0.0880030 |
| p-value | 0.9341040 |
| 95% CI for mean difference | [-0.001055323, +0.001124413] |

**Interpretation:** the test did not detect a mean RMSE difference. It provides no evidence of transformer superiority and does not establish equivalence. A p-value of 0.934 is not a 93.4% probability that the models are equal.

Inference is conditional on this fixed dataset and concerns training-seed variability. OLS is constant; variation comes from transformer fitting. These are not five independent fields or test datasets. The test assumes independent seed realizations and approximately normal run-level differences, which five observations cannot establish. The test set had already been examined, so this is exploratory rather than a pristine confirmatory experiment.

## 7. Health-category diagnostics

The categories are derived from NDVI, not independent field-health measurements:

- Unfavorable: NDVI < 0.4.
- Marginal: 0.4 <= NDVI < 0.6.
- Healthy: NDVI >= 0.6.

Targets were already filtered to NDVI > 0.2; evaluation does not cover all bare-soil/background conditions.

| Actual test category | Count | Percentage |
| --- | ---: | ---: |
| Unfavorable | 15,592 | 77.960% |
| Marginal | 4,301 | 21.505% |
| Healthy | 107 | 0.535% |

**OLS predicts all 20,000 samples as unfavorable.** Its 77.96% accuracy equals the majority-category baseline. Macro F1 is only 0.292051; marginal and healthy recalls are zero.

The transformer predictions are:

| Transformer seed | Predicted unfavorable | Predicted marginal | Predicted healthy |
| --- | --- | --- | --- |
| 42 | 19847 | 153 | 0 |
| 43 | 19969 | 31 | 0 |
| 44 | 19756 | 244 | 0 |
| 45 | 19920 | 80 | 0 |
| 46 | 19937 | 63 | 0 |

No transformer run predicts a healthy sample. Marginal recall ranges from approximately 0.2% to 1.8%. Mean accuracy of 77.75% must not be presented as strong three-class performance; mean macro F1 is 0.296793.

Predictions are strongly underdispersed: actual test NDVI SD is approximately 0.095293, OLS prediction SD is 0.024601, and transformer prediction SD averages about 0.031608. This is consistent with compressed predictions toward central values. It does not establish a unique cause such as architecture, scaling or inadequate optimization.

## 8. Quantile diagnostics

The 0.1-to-0.9 band has nominal 80% coverage. Test diagnostics:

| Seed | Empirical coverage | Mean interval width (NDVI) | Crossing rate |
| --- | --- | --- | --- |
| 42 | 0.838850 | 0.239463 | 0.000000 |
| 43 | 0.840600 | 0.245745 | 0.000000 |
| 44 | 0.832700 | 0.244732 | 0.000000 |
| 45 | 0.814700 | 0.233536 | 0.000000 |
| 46 | 0.863500 | 0.251358 | 0.000000 |

Empirical coverage ranges from 81.47% to 86.35%; mean widths are about 0.234–0.251 NDVI units. No quantile crossings occurred in these saved evaluations. This is fixed-test empirical coverage, not a calibration guarantee or evidence of accurate intervals at every horizon/date. Good aggregate coverage can coexist with weak median/category predictions.

## 9. Historical real-data results

These preserved results are background, not extra independent pairs in the current test:

| Experiment | Model | RMSE | MAE | R-squared |
| --- | --- | --- | --- | --- |
| Earlier adapted pipeline: strawberry | linear | 0.086466 | 0.071092 | 0.079660 |
| Earlier adapted pipeline: strawberry | transformer | 0.085682 | 0.069779 | 0.096270 |
| Earlier adapted pipeline: reconstructed persistence | persistence | 0.121254 | 0.094761 | -0.809882 |
| First season-aware spatial run: v1 | persistence | 0.128357 | 0.100304 | -0.814347 |
| First season-aware spatial run: v1 | linear | 0.090768 | 0.074245 | 0.092718 |
| First season-aware spatial run: v1 | transformer | 0.089623 | 0.072626 | 0.115455 |

The older strawberry run used an earlier reference-footprint cache, calendar-year assumptions and a non-residual transformer interpretation. Its scores are not directly interchangeable with the newer union-grid/provisional-season scores.

Spatial v1 and the later seed-42 retraining are separate saved runs with the same declared seed; both are not counted as independent seed repetitions. evaluation_manual_01 reproduced v1's test metrics and is a repeated evaluation of one checkpoint, not a new fit.

## 10. Forecast products and historical coverage audit

Forecast export can cover all eligible locations, including training/validation regions. Full-field map coverage is different from held-out sample accuracy.

Outputs include predicted NDVI, latest history, history counts, coverage, per-pixel horizon, coverage reasons, optional NDVI categories and transformer quantiles. With observed target truth, outputs also include actual NDVI, signed/absolute error and validity masks. GeoTIFFs retain CRS, transform, extent/spacing and NoData. Bounded PNGs support inspection.

An audit of the retained June 30, 2025 transformer map found:

| Coverage category | Cells |
| --- | ---: |
| Total grid | 49,000,000 |
| Predicted | 2,125,161 |
| NoData | 46,874,839 |
| Missing because no usable history | 28,853,584 |
| Missing because insufficient history | 18,021,255 |
| Unexplained missing under old rules | 0 |

June 30 is not an observed manifest date, so that map cannot be scored against same-date truth. The audit explains coverage under old rules; it does not validate forecast accuracy or current season assumptions. The five-seed exercise generated evaluation products, not a new full-field real-data map for every seed.

## 11. Synthetic experiments and software tests

Synthetic rasters exercise alignment shifts, missing values and controlled input conditions. Their models are tiny, typically d_model=8, one layer, two heads, two epochs and 30 samples per split. **Their scores are not strawberry forecasting results.**

Recorded synthetic test scores, for completeness:

| Synthetic experiment ONLY | Linear RMSE | Transformer RMSE | Persistence RMSE |
| --- | --- | --- | --- |
| Original synthetic run | 0.014436 | 0.952901 | Not saved in original comparison |
| Spatial smoke (earlier settings) | 0.014436 | 0.456032 | 0.041741 |
| Spatial CLI validation | 0.014436 | 0.038207 | 0.041741 |
| Cross-year smoke (earlier settings) | 0.162643 | 0.495238 | 0.037099 |
| Cross-year CLI validation | 0.162643 | 0.033605 | 0.037099 |

Earlier smoke runs used different early output/loss settings. Large errors from those two-epoch synthetic models must not be assigned to the final real-data model. Differences between smoke and later CLI scores are not a controlled ablation showing which one change caused improvement.

Complete synthetic CLI workflows passed in spatial and cross-year modes: prepare, examples, train, evaluate and forecast all three models with classification. Additional small parallel-seed checks verified repeated training, evaluation and summary generation. Temporary artifacts outside the project are machine-local verification evidence, not durable research datasets.

### Automated test outcomes

Latest full suite: **25 passed**, with 126 existing Affine/Matplotlib deprecation warnings. After the last summary-text change, all eight paired-analysis cases passed again. These are software checks, not 25 independent forecasting experiments.

Coverage expanded from seven initial cases to 14, then 16 with import/serialization compatibility, 17 with legacy CLI integration, and 25 with paired-analysis checks.

| Test file | Check | Latest result |
| --- | --- | --- |
| test_pipeline.py | alignment missing values | Passed |
| test_pipeline.py | sequences and no future leakage | Passed |
| test_pipeline.py | spatial isolation | Passed |
| test_pipeline.py | forward masks and time | Passed |
| test_pipeline.py | dataset future mutation | Passed |
| test_pipeline.py | optional features and external mask | Passed |
| test_pipeline.py | end to end and export | Passed |
| test_experiments.py | union retains partial edges | Passed |
| test_experiments.py | explicit season id | Passed |
| test_experiments.py | cross year frozen rolling origin | Passed |
| test_experiments.py | heldout validation never becomes training history | Passed |
| test_experiments.py | no crossing unverified boundaries | Passed |
| test_experiments.py | chronological validation and invalid training years | Passed |
| test_experiments.py | old checkpoint still loads | Passed |
| test_experiments.py | model import and pickle compatibility | Passed |
| test_experiments.py | pipeline import and pickle compatibility | Passed |
| test_experiments.py | legacy launcher train and predict | Passed |
| test_repeated_comparison.py | paired statistic uses run differences | Passed |
| test_repeated_comparison.py | zero variance does not fabricate a p value | Passed |
| test_repeated_comparison.py | invalid pairs rejected: one pair | Passed |
| test_repeated_comparison.py | invalid pairs rejected: nonfinite score | Passed |
| test_repeated_comparison.py | invalid pairs rejected: unequal lengths | Passed |
| test_repeated_comparison.py | duplicate seeds are not independent repeats | Passed |
| test_repeated_comparison.py | mismatched test targets rejected | Passed |
| test_repeated_comparison.py | summary outputs pairs not pooled pixels | Passed |

Other checks confirmed old checkpoint loadability; exactly preserved 24 quantile outputs on eight samples during source reorganization; both manifests resolving source rasters after relocation; matching checkpoint/dataset metadata; valid documentation links; unchanged sample-file hashes during repeated fits; and the paired statistic/p-value reproduced by a one-sample test on the difference vector.

Cross-year chronology/leakage logic passed synthetic tests. **A full real-data cross-year comparison is not complete.** Do not present its synthetic scores as real 2025 results.

## 12. Organization and reproducibility work

The Research directory contains source, data, configurations, caches, checkpoints, reports, maps and upstream reference code. The existing .venv remains in the parent directory to preserve installed launchers.

The software stack includes Python 3.13, PyTorch, scikit-learn, NumPy, pandas, rasterio, Matplotlib, joblib, PyYAML and SciPy. The source layout and CLI packaging are declared in pyproject.toml. This is a local research pipeline, not a deployed production service.

- Models: src/uav_ndvi/models/.
- Preprocessing: preprocessing.py; histories/splits: dataset.py.
- Fitting/checkpoint loading: train.py; evaluation: evaluate.py; maps: forecast.py.
- CLI: python -m uav_ndvi, with prepare/preprocess, examples, train, evaluate and forecast.
- Older module names remain compatibility imports, not duplicate implementations.
- The standalone legacy launcher remains usable.
- Upstream source/provenance and license text are retained; documentation points to the packaged license.
- Absolute paths in 21 saved JSON files were migrated. .relocation.json resolves historical checkpoint metadata without rewriting weight binaries.
- Saved sample indices, resolved settings, histories, checkpoints, paired scores and pre-run sample hashes support traceability.
- Verified disposable caches and duplicate historical-document/report/license copies were removed. Cleanup is an engineering change, not an accuracy improvement.

## 13. Gaps to consider before the presentation

1. **Resolve split wording with the professor.** PDF page 8 specifies the implemented spatial split but also calls ground truth a withheld acquisition date. Clarify whether the intended claim is unseen locations, unseen dates or both. Current evidence supports unseen locations only.
2. **Address poor health-category detection.** Neither fitted model predicts healthy samples. OLS reproduces the majority category; marginal recall is also very low for the transformer. Include class counts, macro F1 and confusion matrices, not accuracy alone.
3. **Discuss absolute fit.** Mean R² is approximately 0.09 for both models. Similar scores do not establish that either is useful for operational decisions. Inspect scatter, distribution and date/horizon plots.
4. **Complete or explicitly defer attention/temporal analysis.** The proposal promises attention weights and understanding influential flights. The architecture's ability to attend does not demonstrate which temporal dependencies were learned. Extraction or a properly designed temporal sensitivity/ablation study remains work.
5. **AUC-ROC is absent.** Define suitable scoring for the derived categories or agree to defer/revise that metric. Current regression outputs/quantiles are not a trained three-class probability distribution. Do not invent AUC scores.
6. **Explain proposal-method changes.** Dataset years expanded beyond 2024–2025. The baseline uses engineered statistics/trends rather than the specified flattening/sums. Current outputs are unclipped. Describe actual-day encoding as the implemented adaptation rather than claiming unchanged inheritance.
7. **Scope the statistical inference.** Only five seeds, one fixed split and deterministic OLS were used. Normality of seed-level differences is not established. These are not independent farms/datasets. p=0.934 does not prove equivalence, and prior test-set inspection makes the analysis exploratory.
8. **Account for dependence and representativeness.** Neighboring pixels and repeated dates can be correlated. No spatial buffer is used. Unequal date/class coverage and the 20,000-example caps limit representativeness; geographic percentages are different from sample-count percentages.
9. **Confirm season labels and data quality.** Observation blocks are provisional, 2025 has only four acquisitions, and the 2023 singleton contributes no supervised sequence. Confirm crop cycles, physical spacing and registration before stronger claims.
10. **Clarify the meaning of health.** Threshold-derived NDVI categories are not independent diagnoses of disease, stress, biomass or yield. Filtering NDVI <= 0.2 also excludes low-NDVI conditions from current evaluation.
11. **Temporal generalization remains untested on real data.** If future unseen dates/years are a required main claim, add a properly frozen chronological evaluation. Shared target dates across spatial regions do not establish that claim.
12. **No controlled component ablations are complete.** Benefit from actual-day encoding, the future-token branch, residual learning, history length or loss choice has not been isolated. Hyperparameters were not exhaustively optimized. Do not claim transformers generally fail on NDVI.
13. **Uncertainty and maps need careful wording.** Aggregate quantile coverage is not date-specific calibration. Full-field maps include training regions; an unobserved forecast date cannot have an observed same-date error map.
14. **Audit literature claims before repeating them.** Prior results on other crops, biomass/yield, sensors or architectures motivate the study; they do not guarantee superior transformer forecasts on this dataset.

## 14. Presentation-ready conclusion

An end-to-end UAV NDVI forecasting pipeline was implemented with explicit observation blocks, irregular-time histories, a transformer adaptation, OLS, persistence, georeferenced outputs and reproducible evaluation. Across five seeds on a fixed geographic split, transformer mean RMSE was 0.090802 versus OLS 0.090768; the paired test did not detect a mean difference (p=0.934). Both fitted models reduced regression error relative to persistence, but category discrimination was weak, especially for marginal and healthy targets. These findings support investigation of data assumptions, temporal evaluation, class imbalance and model behavior. They do not demonstrate transformer superiority or deployment-ready crop-health forecasting.

## 15. Suggested slide sequence

1. Title, motivation and research question.
2. Proposal objectives and implemented scope.
3. Data, acquisition gaps and provisional blocks.
4. NDVI preparation, alignment and missing values.
5. Geographic split, sampling and fair comparison.
6. Persistence and engineered OLS.
7. Transformer architecture and elapsed-day encoding.
8. Training and validation-based selection.
9. Five-run scores, averages and paired comparison plot.
10. Paired t-test and its scope.
11. Scatter plots, class imbalance and confusion matrices.
12. Gaps, limitations and next experiments.
13. Conclusion and contribution.

## 16. Evidence locations

All paths below are relative to Research:

- Five-pair report/plot: runs/model_comparison/five_seed_2026-11-18/comparison.md and paired_comparison.png.
- Individual scores/statistics: paired_scores.csv and paired_test.json in that directory.
- Settings/sample hashes: protocol.json in that directory.
- Seed 42: runs/spatial_generalization/retrain_2026-11-18/.
- Seeds 43–46: their individual directories under the five-seed comparison directory.
- Each evaluation_test/ contains predictions, confusion/class reports, overall/date/horizon metrics, diagnostic plots and interval diagnostics.
- Useful illustrative single-run figures are evaluation_test/test_transformer_scatter.png, test_transformer_diagnostics.png, test_linear_diagnostics.png, test_rmse_by_date.png and test_rmse_by_horizon.png under the seed-42 run. Its training.png shows fitting history. Label these as seed-42 figures rather than five-run averages.
- Historical audit: docs/existing_audit.json.
- Adaptation/source review: docs/upstream-review.md.
- Execution history: docs/validation.md. Some opening historical statements about unexecuted real-data runs predate later completed spatial runs and are superseded by the later entries.
- Commands: docs/workflows.md; repeated-study method: docs/repeated-comparison.md.
- The supplied proposal was reviewed from the user's Downloads folder, outside this repository.

Method references: [NIST paired observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) and [SciPy paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html). These describe methods; the numerical results above come from saved local experiments.
