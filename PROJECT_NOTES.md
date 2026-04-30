# Project Notes

## What changed after professor feedback

The original presentation proposed an embedding-augmented forecasting setup using Chronos/Kronos representations. The professor suggested improving the comparison by using stronger and more relevant forecasting models.

The project was therefore revised into a direct benchmark of modern time-series models and foundation models.

Main changes:

1. Added a Naive / Random Walk baseline.
2. Added N-BEATS as a strong deep learning time-series baseline.
3. Added PatchTST as a transformer-based forecasting baseline.
4. Added Chronos zero-shot as a general time-series foundation model baseline.
5. Added Kronos zero-shot as a finance-focused foundation model baseline.
6. Kept walk-forward evaluation to avoid look-ahead bias.
7. Added MAE, RMSE, Directional Accuracy, RankIC, and portfolio-style long-short evaluation.
8. Added a final comparison table combining forecasting, ranking, and portfolio-style metrics.

## Why the Naive baseline matters

The Naive / Random Walk baseline predicts no future return:

```text
y_pred = 0
pred_close = last_close
```

This is essential because financial returns are difficult to predict. Any advanced model should be compared against this simple benchmark before claiming improvement.

## Final model list

The final benchmark includes:

```text
Naive-RandomWalk
NBEATS
PatchTST
Chronos-ZeroShot
Kronos-ZeroShot
```

## Evaluation setup

The project uses walk-forward evaluation:

1. Train or condition only on historical data.
2. Predict the next forecast horizon.
3. Move the cutoff date forward.
4. Repeat for all selected cutoff dates and horizons.

This avoids using future data during training or prediction.

## Metrics used

### Forecast-level metrics

- MAE
- RMSE

These measure prediction error magnitude.

### Signal-level metrics

- Directional Accuracy
- RankIC

These evaluate whether the model predicts return direction or cross-sectional ranking.

### Portfolio-style metric

- Long-short portfolio summary
- Approximate Sharpe ratio

This checks whether predicted rankings can be translated into a simple finance signal.

## Important RankIC limitation

RankIC needs a reasonably large cross-section of assets. With only 10 stocks, RankIC should be described as exploratory.

Defendable wording:

> RankIC is reported as a diagnostic metric, but conclusions are limited by the small asset universe. A larger universe, such as 50+ stocks, would provide a more reliable cross-sectional evaluation.

## Recommended final run strategy

To avoid errors and unnecessary computation, first reuse existing predictions:

```python
RUN_NEURAL_MODELS = True
RUN_CHRONOS = True
RUN_KRONOS = True

FORCE_RERUN_NEURAL = False
FORCE_RERUN_CHRONOS = False
FORCE_RERUN_KRONOS = False
```

Then run:

```text
notebooks/03_run_new_models_complete.ipynb
notebooks/04_final_analysis.ipynb
```

This should regenerate:

```text
results/new_models/naive_predictions.csv
results/new_models/naive_summary.csv
results/new_models/all_predictions.csv
results/new_models/summary.csv
results/new_models/rank_ic_by_date.csv
results/new_models/rank_ic_summary.csv
results/new_models/portfolio_summary.csv
results/new_models/final_model_comparison.csv
```

## When to do a full rerun

Only use a full rerun if you changed:

- asset list
- forecast horizons
- model configuration
- input window size
- training steps
- raw or processed data

For a full rerun, set:

```python
FORCE_RERUN_NEURAL = True
FORCE_RERUN_CHRONOS = True
FORCE_RERUN_KRONOS = True
```

Expected runtime for the current 10-asset setup is roughly 45 minutes to 2.5 hours, depending on hardware.

## If scaling to 50 assets

To make RankIC more defendable, update `src/config.py` with a larger ticker list, then rerun everything from the beginning:

```text
01_data_collection.ipynb
02_prepare_forecasting_data.ipynb
03_run_new_models_complete.ipynb
04_final_analysis.ipynb
```

For 50 assets, use a smaller check configuration first:

```python
config = BenchmarkConfig(
    horizons=[1],
    input_size=64,
    test_step=20,
    max_test_dates=1,
    max_steps=10,
    freq="B",
    random_seed=42,
)
```

After confirming the pipeline works, run the stronger final configuration.

## Defendable final interpretation

This project should not claim that the models strongly predict the stock market.

The safest interpretation is:

> The project demonstrates a leakage-safe benchmark pipeline for comparing modern time-series models and foundation models in equity return forecasting. N-BEATS, PatchTST, Chronos, and Kronos can be compared consistently against a Naive / Random Walk baseline. However, directional and ranking signals remain limited, especially under a small asset universe. Future work should scale the evaluation to more assets and add stronger statistical testing.
