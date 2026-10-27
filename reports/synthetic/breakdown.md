# CRPS (departure delay, minutes) by inbound-aircraft state - lgbm vs gnn

| horizon | subset | n | lgbm | gnn | gnn vs lgbm |
|---|---|---|---|---|---|
| 1h | all flights | 17849 | 7.19 | 7.16 | -0.4% |
| 1h | inbound aircraft not departed yet | 514 | 9.88 | 9.84 | -0.4% |
| 1h | inbound departed or landed | 16500 | 6.56 | 6.61 | +0.7% |
| 1h | WAW departures | 8920 | 7.10 | 7.13 | +0.4% |
| 3h | all flights | 17849 | 8.71 | 8.47 | -2.8% |
| 3h | inbound aircraft not departed yet | 8449 | 9.86 | 9.49 | -3.8% |
| 3h | inbound departed or landed | 8563 | 7.19 | 7.23 | +0.6% |
| 3h | WAW departures | 8920 | 7.72 | 7.53 | -2.5% |
| 6h | all flights | 17849 | 9.57 | 9.05 | -5.5% |
| 6h | inbound aircraft not departed yet | 13468 | 10.04 | 9.27 | -7.7% |
| 6h | inbound departed or landed | 4363 | 8.13 | 8.37 | +3.0% |
| 6h | WAW departures | 8920 | 8.08 | 7.94 | -1.7% |

Day-block bootstrap, 95% CI of mean CRPS difference (gnn - lgbm, minutes; < 0 favours gnn):

- 1h: -0.029 [-0.151, +0.094] over 55 days
- 3h: -0.241 [-0.452, -0.053] over 55 days
- 6h: -0.522 [-0.857, -0.236] over 55 days
