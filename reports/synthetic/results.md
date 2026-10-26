# Results - dataset `synthetic`

Test split = last 15% of days (chronological, 1-day gap). All numbers in minutes.
CRPS is approximated from 19 quantiles. Coverage should be close to 80% / 90%.
Both models are recalibrated post hoc on the validation split (per-quantile offsets);
`raw` columns show the uncalibrated model.

## Departure delay

| model | h | MAE | CRPS | cov80 | cov90 | width90 | raw CRPS | raw cov90 |
|---|---|---|---|---|---|---|---|---|
| LightGBM (baseline) | 1h | 9.01 | 7.19 | 78.3% | 88.7% | 37 | 7.19 | 88.0% |
| LightGBM (baseline) | 3h | 10.86 | 8.71 | 77.2% | 87.9% | 44 | 8.71 | 87.2% |
| LightGBM (baseline) | 6h | 11.92 | 9.57 | 76.1% | 87.3% | 46 | 9.57 | 86.6% |
| GNN (HeteroGATv2 + GRU) | 1h | 8.95 | 7.16 | 77.2% | 87.6% | 35 | 7.17 | 86.2% |
| GNN (HeteroGATv2 + GRU) | 3h | 10.53 | 8.47 | 76.3% | 87.1% | 40 | 8.48 | 85.7% |
| GNN (HeteroGATv2 + GRU) | 6h | 11.22 | 9.05 | 76.4% | 86.8% | 44 | 9.05 | 85.3% |

## Arrival delay

| model | h | MAE | CRPS | cov80 | cov90 | width90 | raw CRPS | raw cov90 |
|---|---|---|---|---|---|---|---|---|
| LightGBM (baseline) | 1h | 10.40 | 8.18 | 78.2% | 88.8% | 43 | 8.18 | 87.8% |
| LightGBM (baseline) | 3h | 12.15 | 9.64 | 77.5% | 88.3% | 49 | 9.64 | 87.3% |
| LightGBM (baseline) | 6h | 13.22 | 10.49 | 76.6% | 87.8% | 51 | 10.49 | 87.0% |
| GNN (HeteroGATv2 + GRU) | 1h | 10.31 | 8.13 | 77.6% | 87.8% | 41 | 8.13 | 86.7% |
| GNN (HeteroGATv2 + GRU) | 3h | 11.77 | 9.35 | 77.3% | 87.4% | 45 | 9.36 | 86.3% |
| GNN (HeteroGATv2 + GRU) | 6h | 12.45 | 9.92 | 77.1% | 87.3% | 49 | 9.93 | 86.2% |

## Missed connections at WAW (MCT 35 min, slack <= 150 min)

| model | h | Brier | Brier skill vs base rate | AUC | miss rate |
|---|---|---|---|---|---|
| LightGBM (baseline) | 1h | 0.0215 | 0.731 | 0.984 | 8.7% |
| LightGBM (baseline) | 3h | 0.0400 | 0.500 | 0.944 | 8.7% |
| LightGBM (baseline) | 6h | 0.0578 | 0.277 | 0.869 | 8.7% |
| GNN (HeteroGATv2 + GRU) | 1h | 0.0218 | 0.727 | 0.982 | 8.7% |
| GNN (HeteroGATv2 + GRU) | 3h | 0.0389 | 0.513 | 0.947 | 8.7% |
| GNN (HeteroGATv2 + GRU) | 6h | 0.0548 | 0.314 | 0.880 | 8.7% |

![CRPS](figures/crps_by_horizon.png)

![calibration](figures/calibration.png)
