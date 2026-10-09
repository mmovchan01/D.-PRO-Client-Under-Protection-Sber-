# EDA: protection_score

## Shapes
- train: 10000 rows x 61 columns
- open test: 2000 rows x 60 columns

## Target
|       |   protection_score |
|:------|-------------------:|
| count |          10000     |
| mean  |             69.37  |
| std   |             23.826 |
| min   |              3.02  |
| 25%   |             53.398 |
| 50%   |             75.757 |
| 75%   |             89.447 |
| max   |             99.806 |

## Missing values (share of rows)
|                    |   train_missing_share |   test_missing_share |
|:-------------------|----------------------:|---------------------:|
| house_value        |                0.4671 |               0.461  |
| car_value          |                0.4103 |               0.4155 |
| average_claim_cost |                0.3959 |               0.4015 |

## Categorical features: target mean by level
### gender
| gender   |   mean |   std |   count |
|:---------|-------:|------:|--------:|
| F        |  69.12 | 23.85 |    4969 |
| M        |  69.62 | 23.81 |    5031 |

### region
| region    |   mean |   std |   count |
|:----------|-------:|------:|--------:|
| FarEast   |  65.78 | 24.69 |     806 |
| Moscow    |  71.51 | 23.16 |    1828 |
| NorthWest |  68.94 | 24.19 |    1215 |
| SPb       |  71.26 | 23.18 |    1014 |
| Siberia   |  68.98 | 23.74 |    1212 |
| South     |  69.39 | 23.39 |    1049 |
| Ural      |  68.77 | 24.23 |    1156 |
| Volga     |  68.62 | 24.03 |    1720 |

### city_type
| city_type   |   mean |   std |   count |
|:------------|-------:|------:|--------:|
| large_city  |  69.59 | 23.85 |    3495 |
| metro       |  71.34 | 23.07 |    2367 |
| rural       |  67.43 | 23.83 |    1540 |
| town        |  68.42 | 24.33 |    2598 |

### education
| education   |   mean |   std |   count |
|:------------|-------:|------:|--------:|
| bachelor    |  68.34 | 24.22 |    4580 |
| master      |  71.71 | 23.25 |    2390 |
| phd         |  74.55 | 22.2  |     669 |
| secondary   |  67.54 | 23.7  |    2361 |

### family_status
| family_status   |   mean |   std |   count |
|:----------------|-------:|------:|--------:|
| divorced        |  69.47 | 23.49 |    1694 |
| married         |  69.98 | 23.61 |    5259 |
| single          |  67.73 | 24.53 |    2277 |
| widowed         |  69.84 | 23.71 |     770 |

### employment
| employment     |   mean |   std |   count |
|:---------------|-------:|------:|--------:|
| business_owner |  78.03 | 20.57 |     780 |
| employed       |  70.62 | 23.32 |    4723 |
| retired        |  64.5  | 25.28 |    1375 |
| self_employed  |  72.98 | 21.82 |    1274 |
| student        |  64.66 | 24.35 |    1047 |
| unemployed     |  62.32 | 24.88 |     801 |

### wealth_segment
| wealth_segment   |   mean |   std |   count |
|:-----------------|-------:|------:|--------:|
| affluent         |  75.92 | 21.22 |    2765 |
| mass             |  65.43 | 24.32 |    6643 |
| premium          |  82.83 | 17.14 |     560 |
| private          |  86.88 | 12.46 |      32 |

## Consistency checks
- `insurance_products` equals the sum of the 8 insurance flags: 100.00% of rows
- `claim_frequency` equals `number_of_claims / years_with_company` (4-decimal rounding): 100.00% of rows
- `average_claim_cost` is missing exactly when `number_of_claims == 0`: 100.00% of rows
- `house_value` is missing for 46.7% of rows (dataset card: ~62%); `car_value` for 41.0% (~58%)
- `wealth_segment` vs `income` ranges:

| wealth_segment   |    min |    max |
|:-----------------|-------:|-------:|
| affluent         |  45001 | 119976 |
| mass             |  12000 |  44999 |
| premium          | 120001 | 334577 |
| private          | 345139 | 346531 |
- Duplicated feature rows in train: 0

## Top correlations with the target (Pearson)
|                        |   pearson_r |
|:-----------------------|------------:|
| insurance_products     |       0.583 |
| active_policies        |       0.492 |
| digital_behavior_score |       0.479 |
| cyber_protection       |       0.461 |
| mobile_app_usage       |       0.436 |
| internet_activity      |       0.425 |
| identity_protection    |       0.296 |
| mobile_sessions        |       0.262 |
| two_factor_auth        |       0.259 |
| password_manager       |       0.246 |
| income                 |       0.244 |
| website_visits         |       0.239 |
| marketing_response     |       0.199 |
| number_of_claims       |       0.188 |
| life_insurance         |       0.185 |
| gadget_insurance       |       0.18  |
| late_payments          |      -0.174 |
| property_insurance     |       0.172 |
| car_insurance          |       0.171 |
| travel_insurance       |       0.165 |

## Target noise structure
- Logit target `z = logit(protection_score/100)`: std 1.457, range [-3.47, 6.24]
- Out-of-fold ridge R^2 on the logit scale: 0.826
- Logit residual: std 0.607, skew -0.032, excess kurtosis 0.072
- Residual std by decile of the fitted index (logit scale is roughly constant, raw scale is not):

|   decile |   logit_resid_std |   raw_resid_std |
|---------:|------------------:|----------------:|
|        0 |             0.614 |          10.948 |
|        1 |             0.615 |          13.836 |
|        2 |             0.601 |          13.806 |
|        3 |             0.599 |          12.939 |
|        4 |             0.596 |          11.686 |
|        5 |             0.604 |          10.178 |
|        6 |             0.623 |           8.804 |
|        7 |             0.591 |           6.623 |
|        8 |             0.62  |           4.627 |
|        9 |             0.608 |           2.647 |

- Out-of-fold RMSE of the linear logit model on the 0-100 scale: 10.358
- Expected RMSE from the logit-scale noise alone (sigma=0.607): 10.41
- Expected RMSE for other logit-noise levels (a model that recovers the true index would still be at this level):

|   logit_noise_sd |   expected_rmse |
|-----------------:|----------------:|
|            0.2   |           3.516 |
|            0.3   |           5.238 |
|            0.4   |           6.934 |
|            0.5   |           8.646 |
|            0.607 |          10.383 |

Reaching RMSE < 7 on this data would require the logit-scale noise to be well below the observed ~0.6,
i.e. the remaining signal must be explained by information that is not in the given features.

## Outliers
Non-binary numeric features were checked with the 1.5*IQR rule; the counts are informational only, no rows were removed because the residuals on the logit scale are close to Gaussian.
|                        |   outliers |
|:-----------------------|-----------:|
| expired_policies       |       2107 |
| late_payments          |       1259 |
| claim_frequency        |       1165 |
| loan_amount            |       1146 |
| average_balance        |        992 |
| active_policies        |        864 |
| income                 |        646 |
| average_claim_cost     |        559 |
| days_since_last_login  |        504 |
| years_with_company     |        454 |
| payment_discipline     |        433 |
| days_since_last_policy |        298 |
| occupation_risk        |        280 |
| smartphone_price       |        272 |
| smartphone_age         |        212 |
| bank_products          |        212 |
| house_value            |        206 |
| mobile_sessions        |        152 |
| customer_loyalty       |        122 |
| website_visits         |         77 |
| children               |         76 |
| crime_rate             |         73 |
| regional_risk          |         66 |
| credit_score           |         45 |
| risk_tolerance         |         36 |
| number_of_claims       |         28 |
| health_index           |         27 |
| cyber_risk             |         23 |
| call_center_contacts   |         21 |
| insurance_products     |          3 |
| online_payments_ratio  |          3 |

