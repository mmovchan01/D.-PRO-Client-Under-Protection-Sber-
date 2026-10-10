# Hypotheses about hidden structure (hard_train.csv, 5-fold CV)

Yardstick: out-of-fold residual sd on the logit scale; baseline sigmoid-linear index.

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| H0 baseline sigmoid-linear index | reference | logit resid sd 0.6072 (0.6106, 0.6039, 0.6072) |
| H1 a numeric feature is an exact function of the others (derived feature) | rejected | max OOF R^2 of one feature from the others: active_policies 0.72, mobile_app_usage 0.71, digital_behavior_score 0.67 |
| H2 residuals are locally correlated (kNN in feature space) | rejected | corr(own residual, mean of 20 neighbours) = -0.0007 |
| H3a category x numeric interactions improve the fit | rejected | all 1300 interactions: 0.6220 (worse than 0.6072); overfitting |
| H3b the gain comes from wealth_segment=private x income | explains the small gain | adding only private x income: 0.6005; this term affects 32 rows (income extrapolation) |
| H4 explicit money ratios (loan, balance, house, car, phone to income) | small logit gain (not tested on 0-100 scale) | logit resid sd 0.6031 vs 0.6072 |
| H5 threshold effects missed by the linear index (boosting on residuals) | rejected on 0-100 scale | logit resid sd 0.6073 -> 0.6007; but 0-100 RMSE 10.1696 (reference) vs 10.2598 (boosted) |
| H6 noise level differs between groups | rejected | largest within-variable range of group residual sd: 0.026 (all within 0.60-0.66) |
| H7 test set comes from a different distribution than train | rejected | adversarial validation AUC 0.499 |
| H8 another link function (probit) fits clearly better than logit | rejected | logit: raw RMSE 10.417; probit: raw RMSE 10.414; logit residuals are homoscedastic, probit residuals are not |
| H9 the remaining error is irreducible logit noise | supported | Bayes RMSE floor by sigma: 0.400 -> 6.93, 0.500 -> 8.60, 0.600 -> 10.22, 0.611 -> 10.39 |
| H10 income tail (income > 300k) is a large source of error | rejected | 46 rows (0.46%); perfect prediction there moves RMSE 10.1586 -> 10.1425 |
| H11 fractional logit (binomial GLM) beats MSE fit on 0-100 scale | rejected | 0-100 RMSE 10.1609 vs 10.1586 for the reference model (difference within CV noise) |
| H12 drop the aggregate insurance_products (collinear with the 8 flags) | no effect | 0-100 RMSE without aggregate 10.1588 vs 10.1586; dropping the flags instead is much worse (logit sd 0.705) |

Conclusion: no tested hypothesis lowers the logit noise below roughly 0.60, and the 0-100 RMSE was checked for every candidate that lowered it on the logit scale (H5).
A logit-scale gain (H4, H5) does not transfer to the 0-100 scale, so decisions are made on the 0-100 RMSE.
At that noise level the Bayes RMSE floor is about 10.2-10.3, so RMSE < 7 would require a logit noise near 0.4.
