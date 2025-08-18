# Results Summary

## Cross-Validation (mean ± std)
- PLS: RMSE 1.399 ± 0.108, MAE 1.099 ± 0.077, R² 0.365 ± 0.150
- SVR: RMSE 1.564 ± 0.048, MAE 1.210 ± 0.037, R² 0.217 ± 0.085

## Leave-One-Batch-Out (mean ± std)
- PLS: RMSE 1.393 ± 0.060, MAE 1.089 ± 0.051, R² 0.383 ± 0.054
- SVR: RMSE 1.544 ± 0.023, MAE 1.212 ± 0.016, R² 0.243 ± 0.023

## LOBO Best Model Per Held-Out Batch
| held_out   | model   |    rmse |     mae |       r2 |
|:-----------|:--------|--------:|--------:|---------:|
| kiwi-1.csv | PLS     | 1.34981 | 1.05657 | 0.421486 |
| kiwi-2.csv | PLS     | 1.46124 | 1.14699 | 0.322035 |
| kiwi-3.csv | PLS     | 1.36669 | 1.06254 | 0.406932 |