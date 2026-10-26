# Results - dataset `synthetic`

Test split = last 15% of days (chronological, 1-day gap). All numbers in minutes.
CRPS is approximated from 19 quantiles. Coverage should be close to 80% / 90%.

## Departure delay

| model | h | MAE | CRPS | cov80 | cov90 | width90 |
|---|---|---|---|---|---|---|
| LightGBM (baseline) | 1h | 9.12 | 7.27 | 77.9% | 88.2% | 38 |
| LightGBM (baseline) | 3h | 11.00 | 8.81 | 76.6% | 87.1% | 45 |
| LightGBM (baseline) | 6h | 12.07 | 9.68 | 75.7% | 86.3% | 46 |

## Arrival delay

| model | h | MAE | CRPS | cov80 | cov90 | width90 |
|---|---|---|---|---|---|---|
| LightGBM (baseline) | 1h | 10.59 | 8.28 | 77.1% | 87.7% | 43 |
| LightGBM (baseline) | 3h | 12.37 | 9.78 | 76.6% | 87.0% | 49 |
| LightGBM (baseline) | 6h | 13.39 | 10.64 | 75.2% | 86.2% | 51 |

## Missed connections at WAW (MCT 35 min, slack <= 150 min)

| model | h | Brier | Brier skill vs base rate | AUC | miss rate |
|---|---|---|---|---|---|
| LightGBM (baseline) | 1h | 0.0217 | 0.728 | 0.985 | 8.7% |
| LightGBM (baseline) | 3h | 0.0400 | 0.499 | 0.946 | 8.7% |
| LightGBM (baseline) | 6h | 0.0580 | 0.274 | 0.866 | 8.7% |

![CRPS](figures/crps_by_horizon.png)

![calibration](figures/calibration.png)
