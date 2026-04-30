from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def summarize_predictions(predictions):
    rows = []

    for (model, horizon), g in predictions.groupby(["Model", "Horizon"]):
        g = g.dropna(subset=["y_true", "y_pred"])

        if g.empty:
            continue

        mse = mean_squared_error(g["y_true"], g["y_pred"])
        rmse = mse ** 0.5

        rows.append({
            "Model": model,
            "Horizon": horizon,
            "N": len(g),
            "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
            "RMSE": rmse,
            "DirectionalAccuracy": directional_accuracy(g["y_true"], g["y_pred"]),
            "MeanActualReturn": g["y_true"].mean(),
            "MeanPredReturn": g["y_pred"].mean(),
        })

    return pd.DataFrame(rows).sort_values(["Horizon", "MAE", "Model"]).reset_index(drop=True)


def cross_sectional_rank_ic(predictions: pd.DataFrame, min_assets: int = 5) -> pd.DataFrame:
    rows = []
    required = ["Date", "Ticker", "Model", "Horizon", "y_true", "y_pred"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing columns for RankIC: {missing}")

    for (model, horizon, date), g in predictions.groupby(["Model", "Horizon", "Date"]):
        g = g.dropna(subset=["y_true", "y_pred"])
        if g["Ticker"].nunique() < min_assets:
            continue
        if g["y_true"].nunique() < 2 or g["y_pred"].nunique() < 2:
            continue
        ic, _ = spearmanr(g["y_pred"], g["y_true"])
        if np.isfinite(ic):
            rows.append({
                "Model": model,
                "Horizon": horizon,
                "Date": date,
                "RankIC": float(ic),
                "N_Assets": int(g["Ticker"].nunique()),
            })
    return pd.DataFrame(rows)


def summarize_rank_ic(rank_ic_df: pd.DataFrame) -> pd.DataFrame:
    if rank_ic_df.empty:
        return pd.DataFrame(columns=["Model", "Horizon", "RankIC_Mean", "RankIC_Std", "RankIC_IR", "N_Dates"])
    rows = []
    for (model, horizon), g in rank_ic_df.groupby(["Model", "Horizon"]):
        mean = g["RankIC"].mean()
        std = g["RankIC"].std(ddof=1)
        rows.append({
            "Model": model,
            "Horizon": horizon,
            "RankIC_Mean": mean,
            "RankIC_Std": std,
            "RankIC_IR": mean / std if std and np.isfinite(std) else np.nan,
            "N_Dates": len(g),
        })
    return pd.DataFrame(rows).sort_values(["Horizon", "RankIC_Mean"], ascending=[True, False]).reset_index(drop=True)
def long_short_portfolio_summary(
    predictions: pd.DataFrame,
    top_n: int = 2,
    bottom_n: int = 2,
) -> pd.DataFrame:
    """
    Simple cross-sectional long-short portfolio test.

    For each model, horizon, and date:
    - rank assets by predicted return
    - long top_n assets
    - short bottom_n assets
    - portfolio return = average true return of longs - average true return of shorts
    """
    rows = []

    required = ["Date", "Ticker", "Model", "Horizon", "y_true", "y_pred"]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"Missing columns for portfolio summary: {missing}")

    for (model, horizon, date), g in predictions.groupby(["Model", "Horizon", "Date"]):
        g = g.dropna(subset=["y_true", "y_pred"]).copy()

        if len(g) < top_n + bottom_n:
            continue

        g = g.sort_values("y_pred", ascending=False)

        long_return = g.head(top_n)["y_true"].mean()
        short_return = g.tail(bottom_n)["y_true"].mean()
        portfolio_return = long_return - short_return

        rows.append({
            "Model": model,
            "Horizon": horizon,
            "Date": date,
            "LongReturn": long_return,
            "ShortReturn": short_return,
            "LongShortReturn": portfolio_return,
            "N_Assets": g["Ticker"].nunique(),
        })

    daily = pd.DataFrame(rows)

    if daily.empty:
        return pd.DataFrame(columns=[
            "Model", "Horizon", "MeanLongShortReturn",
            "VolLongShortReturn", "SharpeApprox", "N_Dates"
        ])

    summary_rows = []

    for (model, horizon), g in daily.groupby(["Model", "Horizon"]):
        mean_ret = g["LongShortReturn"].mean()
        vol_ret = g["LongShortReturn"].std(ddof=1)

        summary_rows.append({
            "Model": model,
            "Horizon": horizon,
            "MeanLongShortReturn": mean_ret,
            "VolLongShortReturn": vol_ret,
            "SharpeApprox": mean_ret / vol_ret if vol_ret and np.isfinite(vol_ret) else np.nan,
            "N_Dates": len(g),
        })

    return pd.DataFrame(summary_rows).sort_values(
        ["Horizon", "SharpeApprox"],
        ascending=[True, False],
    ).reset_index(drop=True)