from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def download_ohlcv(tickers: Iterable[str], raw_dir: Path, start: str, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV data with yfinance and save one CSV per ticker."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df.empty:
                print(f"  WARNING: no data for {ticker}")
                summary_rows.append({"Ticker": ticker, "rows": 0, "status": "empty"})
                continue

            # Handle yfinance MultiIndex columns if present.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            df = df.rename(columns={"date": "Date"})
            df["Ticker"] = ticker

            keep = ["Date", "Ticker", "open", "high", "low", "close", "adj_close", "volume"]
            keep = [c for c in keep if c in df.columns]
            df = df[keep].copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

            out = raw_dir / f"{ticker}.csv"
            df.to_csv(out, index=False)
            summary_rows.append({"Ticker": ticker, "rows": len(df), "status": "ok"})
            print(f"  saved {out} ({len(df)} rows)")
        except Exception as exc:
            print(f"  ERROR for {ticker}: {exc}")
            summary_rows.append({"Ticker": ticker, "rows": 0, "status": f"error: {exc}"})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(raw_dir / "download_summary.csv", index=False)
    return summary


def load_raw_ticker(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date")
    return df


def prepare_single_ticker(raw_path: Path, processed_dir: Path, freq: str = "B") -> pd.DataFrame:
    """Prepare one ticker file for forecasting.

    We reindex to business days and forward-fill OHLCV so NeuralForecast sees a regular series.
    Returns are kept for diagnostics, but models forecast close prices and convert forecasts to returns.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_ticker(raw_path)
    ticker = df["Ticker"].iloc[0] if "Ticker" in df.columns else raw_path.stem

    df = df.set_index("Date").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    df.index.name = "Date"
    df["Ticker"] = ticker

    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    volume_cols = [c for c in ["volume"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    df[volume_cols] = df[volume_cols].fillna(0)

    df = df.reset_index()
    df["return_1d"] = df["close"].pct_change()
    df["log_return_1d"] = np.log(df["close"]).diff()
    df["volume"] = df.get("volume", 0)
    df["amount"] = df["close"] * df["volume"].fillna(0)

    # Keep rows with usable close.
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    out = processed_dir / f"{ticker}_prepared.csv"
    df.to_csv(out, index=False)
    return df


def prepare_all_raw_files(raw_dir: Path, processed_dir: Path, freq: str = "B") -> pd.DataFrame:
    summaries = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.name == "download_summary.csv":
            continue
        try:
            df = prepare_single_ticker(path, processed_dir, freq=freq)
            summaries.append({"Ticker": df["Ticker"].iloc[0], "rows": len(df), "start": df["Date"].min(), "end": df["Date"].max()})
            print(f"Prepared {path.stem}: {len(df)} rows")
        except Exception as exc:
            summaries.append({"Ticker": path.stem, "rows": 0, "start": None, "end": None, "error": str(exc)})
            print(f"ERROR preparing {path.stem}: {exc}")
    summary = pd.DataFrame(summaries)
    summary.to_csv(processed_dir / "prepare_summary.csv", index=False)
    return summary


def load_prepared_panel(processed_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(processed_dir.glob("*_prepared.csv")):
        df = pd.read_csv(path, parse_dates=["Date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_prepared.csv files found in {processed_dir}")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return panel


def to_neuralforecast_df(panel: pd.DataFrame) -> pd.DataFrame:
    nf_df = panel[["Ticker", "Date", "close"]].copy()
    nf_df = nf_df.rename(columns={"Ticker": "unique_id", "Date": "ds", "close": "y"})
    nf_df = nf_df.dropna(subset=["y"])
    return nf_df


def make_cutoff_dates(panel: pd.DataFrame, input_size: int, horizon: int, test_step: int, max_test_dates: int) -> list[pd.Timestamp]:
    """Make shared cutoff dates where all tickers have enough history and future observations."""
    counts = panel.groupby("Date")["Ticker"].nunique()
    n_tickers = panel["Ticker"].nunique()
    common_dates = counts[counts == n_tickers].index.sort_values()

    if len(common_dates) < input_size + horizon + 1:
        raise ValueError("Not enough common dates for the chosen input_size and horizon.")

    candidates = common_dates[input_size : len(common_dates) - horizon : test_step]
    return list(candidates[-max_test_dates:])
