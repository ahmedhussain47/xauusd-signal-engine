# Modern Equity Forecasting Benchmark

This project benchmarks modern time-series forecasting models and foundation models for equity return prediction under a leakage-safe walk-forward evaluation setup.

The original project idea focused on embedding-augmented forecasting using Chronos/Kronos representations. After professor feedback, the project was revised into a direct benchmark of stronger forecasting models and zero-shot foundation models.

## Final research question

Do modern time-series models and foundation models improve equity return forecasting compared to a simple Naive / Random Walk baseline under a common walk-forward evaluation setup?

Sub-questions:

1. Which model performs best for 1-day and 5-day horizons?
2. Are results consistent across assets?
3. Do model predictions produce useful directional or ranking signals?
4. Are zero-shot foundation models competitive with task-specific deep learning models?

## Models

The final benchmark includes:

- Naive / Random Walk baseline
- N-BEATS
- PatchTST
- Chronos zero-shot
- Kronos zero-shot, optional if the local Kronos repository and dependencies are available

## Evaluation metrics

The project reports:

- MAE
- RMSE
- Directional Accuracy
- RankIC
- Long-short portfolio summary
- Final model comparison table

## Important note on RankIC

RankIC is a cross-sectional ranking metric. With a small asset universe, RankIC should be interpreted cautiously. In the current 10-asset setup, RankIC is useful as an exploratory diagnostic, not as strong statistical evidence.

For stronger RankIC conclusions, increase the asset universe to 50+ stocks and rerun the data preparation and forecasting notebooks.

## Project workflow

Run the notebooks in this order:

1. `notebooks/01_data_collection.ipynb`
2. `notebooks/02_prepare_forecasting_data.ipynb`
3. `notebooks/03_run_new_models_complete.ipynb`
4. `notebooks/04_final_analysis.ipynb`

If the prediction CSV files already exist and you only need to regenerate the final merged output, keep the rerun flags in `03_run_new_models_complete.ipynb` as:

```python
FORCE_RERUN_NEURAL = False
FORCE_RERUN_CHRONOS = False
FORCE_RERUN_KRONOS = False
```

This reuses existing model predictions, regenerates the cheap Naive baseline, and recreates `all_predictions.csv`.

## Main output files

After running the final notebooks, check these files:

```text
results/new_models/all_predictions.csv
results/new_models/summary.csv
results/new_models/rank_ic_by_date.csv
results/new_models/rank_ic_summary.csv
results/new_models/portfolio_summary.csv
results/new_models/final_model_comparison.csv
```

Expected model names in the final comparison:

```text
Naive-RandomWalk
NBEATS
PatchTST
Chronos-ZeroShot
Kronos-ZeroShot
```

## Result format

All model prediction files use the same columns:

```text
Date, Ticker, Model, Horizon, y_true, y_pred, cutoff_date, last_close, pred_close, actual_close
```

Where:

- `y_true` is the realized future return
- `y_pred` is the predicted future return
- `last_close` is the close price at the cutoff date
- `pred_close` is the forecasted close price
- `actual_close` is the realized close price at the forecast horizon

## Defendable interpretation

This project should be presented as a benchmark study, not as proof that stock prices can be reliably predicted.

A safe final conclusion is:

> We implemented a leakage-safe benchmark comparing Naive / Random Walk, N-BEATS, PatchTST, Chronos zero-shot, and Kronos zero-shot across multiple equities and forecast horizons. Results show that advanced models can be evaluated consistently in this setup, but directional and ranking signals remain limited under the current asset universe. A larger universe is needed for stronger RankIC and portfolio-level conclusions.

## Notes for final run

For the lowest-risk final run, first reuse existing predictions:

```python
RUN_NEURAL_MODELS = True
RUN_CHRONOS = True
RUN_KRONOS = True

FORCE_RERUN_NEURAL = False
FORCE_RERUN_CHRONOS = False
FORCE_RERUN_KRONOS = False
```

If you want to rerun every model from scratch, change all `FORCE_RERUN_*` values to `True`. This will take much longer.
