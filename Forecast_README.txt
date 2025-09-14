
Forecast add-on for your PBM notebook

Files:
- manifold_learning_v3_with_forecast.ipynb — your original notebook + STEP 6 appended at the end.
- forecast_addon.py — helper functions used by STEP 6.
- (created on run) ./forecast_exports/ — folder with metrics, per-well predictions, and a simple HTML report.

How to use:
1) Open 'manifold_learning_v3_with_forecast.ipynb' and run steps 1–5 as usual to build 'out', 'Z_dtw', 'res', etc.
2) Run STEP 6. It:
   - builds a leakage-free prefix-normalized channel (r_oil_pref_norm),
   - creates prefix/target matrices,
   - fits two baselines (KNN completion and ElasticNet multioutput),
   - evaluates RMSE and sMAPE on months 21–100,
   - saves forecasts and report into './forecast_exports'.
3) Inspect 'forecast_exports/forecast_report.html' and 'metrics.csv'.

Notes:
- The prefix window is set to T_pref = 20 in STEP 6 (change if needed).
- KNN uses neighbor amplitude alignment (least squares) per well.
- ElasticNet features are computed only on the prefix window to avoid leakage.
- Both methods evaluate only wells with full horizon available.

