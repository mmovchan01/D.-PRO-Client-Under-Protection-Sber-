# Numeric customer_id vs protection_score

Script: `id_analysis.py` (run: `python id_analysis.py --train-csv hard_train.csv --output-dir reports/id_analysis`).

- Numeric ID = digits of `customer_id`: train 100000..109999 (10 000 unique, contiguous); open test 110000..111999 (lies above all training IDs).
- Pearson r = -0.0154 (p = 0.125); Spearman rho = -0.0153 (p = 0.127); lag-1 autocorrelation in ID order = -0.009.
- ANOVA of target over 20 blocks of 500 consecutive IDs: F = 1.54, p = 0.063; Kruskal-Wallis p = 0.147.
- Models on the numeric ID only (RMSE, 0-100 scale):

| model | 5-fold CV RMSE | temporal tail RMSE (train first 80%, validate last 20%) |
| --- | ---: | ---: |
| constant mean | 23.827 | 24.300 |
| linear regression | 23.827 | 24.298 |
| kNN (k=200) | 23.858 | 24.305 |
| spline + linear | 23.836 | 24.509 |
| gradient boosting | 23.927 | 26.337 |

Conclusion: the numeric ID carries no signal about protection_score. The best ID-only model is no better than a constant, and boosting is worse on the tail, where the test IDs lie. The ID must not be used as a feature.
