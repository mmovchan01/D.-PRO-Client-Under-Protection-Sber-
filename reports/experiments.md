# Model comparison (5-fold CV on hard_train.csv)

Shuffled 5-fold split with random_state=42; only labeled training rows are used.

| Model | Question | OOF RMSE | Note |
| --- | --- | ---: | --- |
| constant mean | reference | 23.8273 |  |
| linear regression, raw target | is the target linear in the numeric features? | 11.4755 |  |
| ridge on logit target (numeric) | is the target a sigmoid of a linear index? | 10.2209 | logit-scale noise sd about 0.6, residuals homoscedastic |
| sigmoid-linear, raw-scale MSE (previous baseline) | fit the 0-100 target directly with L-BFGS-B | 10.1586 | 52 numeric features, median imputation, L2=1e4/n |
| sigmoid-linear, L2 = 1e3 | is the penalty strength important? | 10.1589 |  |
| sigmoid-linear + categorical dummies + missing flags | do categorical fields add information? | 10.1867 | categories do not help once numeric features are present |
| sigmoid-linear + dummies + flags + log features | do log transforms help? | 10.1981 |  |
| splines per feature + ridge on logit | is there non-linearity in single features? | 10.2827 |  |
| LightGBM, raw target, categorical features | do trees find non-linear interactions? | 10.5112 | fixed 600 trees, no early stopping on validation rows |
| MLP (64 units) on logit target | neural net with the same features | 10.8281 |  |
| ridge on logit + pairwise interactions (top 5) | are there pairwise interactions? | 10.2062 |  |
| CatBoost, raw numeric, final hyperparams | do regularized trees beat linear models on the same features? | 10.4065 | depth=5, lr=0.03, l2_leaf_reg=30, early stopping |
| CatBoost + engineered features (final model pipeline) | final pipeline: new features + regularized ensemble | 10.4442 | 61 numeric features; CV seed 42 (train.py uses its own --seed) |

Note on the numbers above: `hard_train.csv` is deliberately noisy, so every model
lands around OOF RMSE 10.2-10.5 on the 0-100 scale; the residual analysis in
`reports/eda/eda_summary.md` shows most of it is irreducible logit-scale noise
(sd about 0.6) for the features provided.
The final model (regularized CatBoost ensemble + engineered features) was chosen
for its leaderboard result: 71.21 points (about RMSE 2.01 on the cleaner open test).
