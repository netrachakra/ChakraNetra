# ChakraNetra -- Evaluation Results

## Model: HistGradientBoostingRegressor (scikit-learn)

Evaluated on **5 held-out test storms** (real IBTrACS NI basin data, 2018-2023).

| Lead Time | Track Error (km) | Wind MAE (kt) | Pressure MAE (hPa) | Samples |
|-----------|-----------------|---------------|--------------------|---------| 
| +24h | 279.72 | 13.56 | 7.09 | 205 |
| +48h | 523.08 | 21.56 | 11.86 | 160 |
| +72h | 736.14 | 24.09 | 13.12 | 121 |

## Honest Summary

This is a gradient-boosted baseline (HistGradientBoostingRegressor) trained on 15 real North Indian Ocean cyclones from the IBTrACS best-track archive (2018-2023). Track errors grow from ~280 km at +24h to ~736 km at +72h, and intensity MAE from ~14 kt at +24h to ~24 kt at +72h. These numbers are in the expected range for a simple statistical model with no ensemble averaging, no dynamical core, and only 20 storms total. The model demonstrates the full pipeline works end-to-end (IBTrACS -> features -> train -> predict -> evaluate) and provides a usable floor for the hackathon demo, but is NOT competitive with operational NWP-based forecasting.

> **Limitation**: With only 20 storms and simple features (position, motion vector, intensity change), the model struggles with recurvature and rapid intensification. More data and richer features (SST, shear, moisture) would help significantly.
