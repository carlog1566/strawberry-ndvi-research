# Spatial model retraining - November 18, 2026

Retrained the current transformer and supervised regression using the exact previously saved 20,000 examples per partition. Geographic row strips are 70% training, 15% validation and 15% testing; dates are shared across regions. This measures geographic generalization, not unseen-date forecasting.

Training stopped after 48 epochs. The transformer checkpoint is epoch 36, selected by validation RMSE 0.08741611. The linear model fits the training partition only; persistence has no fitted parameters. CPU execution used four threads, seed 42 and the existing spatial configuration. Original run v1 is preserved.

| Model | Test RMSE | Test MAE | Test R2 |
| --- | --- | --- | --- |
| persistence | 0.128357 | 0.100304 | -0.814347 |
| linear | 0.090768 | 0.074245 | 0.092718 |
| transformer | 0.089750 | 0.072664 | 0.112942 |

The 20,000 sampled pixel-date pairs are correlated; these metrics alone do not establish statistical significance or unseen-date performance. Season labels remain provisional. Both saved fitted models were reloaded, and their predictions on eight test examples matched the evaluation CSV.

From the Research directory, evaluate again into a fresh output directory:

```powershell
python -m uav_ndvi evaluate --run runs/spatial_generalization/retrain_2026-11-18 --out runs/spatial_generalization/retrain_2026-11-18/evaluation_repeat_01
```
