"""
Chronos Embedding + N-BEATS Hybrid Pipeline
============================================
Extracts 256-dim embeddings from Chronos encoder, then trains a custom
N-BEATS-inspired neural network that uses BOTH the price series AND the
embedding vector as input.

Comparison:
  - NBEATS-NoEmb  : standard N-BEATS (price only)   [already done]
  - NBEATS-Emb-8  : N-BEATS + Chronos emb (PCA 8)
  - NBEATS-Emb-16 : N-BEATS + Chronos emb (PCA 16)
  - NBEATS-Emb-32 : N-BEATS + Chronos emb (PCA 32)

This directly answers RQ1 (do embeddings help?) and satisfies the
professor's request to vary embedding dimensions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_utils import make_cutoff_dates


# ---------------------------------------------------------------------------
# 1. Chronos Embedding Extractor
# ---------------------------------------------------------------------------

class ChronosEmbeddingExtractor:
    """
    Extracts a single embedding vector per price window using the
    Chronos T5 encoder (frozen weights).

    The encoder hidden states are mean-pooled to get a fixed-size vector.
    No future information is used — only the historical context window.
    """

    def __init__(self, model_id: str = "amazon/chronos-t5-tiny", device: str = "cpu"):
        from chronos import ChronosPipeline

        self.device = device
        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.float32,
        )
        # Freeze all encoder weights — we only use it as a feature extractor
        for param in self.pipeline.model.parameters():
            param.requires_grad = False

        self.pipeline.model.eval()
        print(f"Chronos encoder loaded from {model_id} on {device}")

    def extract(self, price_series: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a 1D price series.

        Parameters
        ----------
        price_series : np.ndarray
            Historical close prices, shape (T,)

        Returns
        -------
        np.ndarray
            Embedding vector, shape (hidden_dim,)
        """
        with torch.no_grad():
            tensor = torch.tensor(price_series, dtype=torch.float32).unsqueeze(0)

            # Tokenize using Chronos tokenizer
            context = self.pipeline.tokenizer.context_input_transform(tensor)
            input_ids = context[0] if isinstance(context, tuple) else context

            # Pass through encoder and mean-pool hidden states
            encoder_out = self.pipeline.model.model.encoder(input_ids=input_ids)
            hidden = encoder_out.last_hidden_state  # [1, seq_len, hidden_dim]
            embedding = hidden.mean(dim=1).squeeze(0)  # [hidden_dim]

        return embedding.cpu().numpy()


# ---------------------------------------------------------------------------
# 2. Embedding Cache (avoids re-extracting for same ticker/date)
# ---------------------------------------------------------------------------

def extract_all_embeddings(
    panel: pd.DataFrame,
    extractor: ChronosEmbeddingExtractor,
    cutoff_dates: list,
    input_size: int,
) -> dict[tuple[str, pd.Timestamp], np.ndarray]:
    """
    Pre-extract embeddings for all (ticker, cutoff_date) pairs.
    Returns a dict: {(ticker, cutoff_date): embedding_vector}
    """
    cache = {}
    tickers = sorted(panel["Ticker"].unique())
    total = len(tickers) * len(cutoff_dates)
    done = 0

    print(f"Extracting embeddings for {len(tickers)} tickers x {len(cutoff_dates)} dates...")

    for ticker in tickers:
        g = panel[panel["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

        for cutoff_date in cutoff_dates:
            idx_list = g.index[g["Date"] == cutoff_date].tolist()
            if not idx_list:
                continue
            idx = idx_list[0]
            if idx + 1 < input_size:
                continue

            context = g.loc[idx - input_size + 1: idx, "close"].astype(float).values
            emb = extractor.extract(context)
            cache[(ticker, cutoff_date)] = emb

            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} embeddings extracted")

    print(f"Done. {len(cache)} embeddings cached.")
    return cache


# ---------------------------------------------------------------------------
# 3. N-BEATS Inspired Neural Network with Embedding Support
# ---------------------------------------------------------------------------

class NBEATSBlock(nn.Module):
    """Single N-BEATS block with fully connected layers."""

    def __init__(self, input_size: int, hidden_size: int, horizon: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.backcast_fc = nn.Linear(hidden_size, input_size)
        self.forecast_fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast_fc(h)
        forecast = self.forecast_fc(h)
        return backcast, forecast


class NBEATSWithEmbedding(nn.Module):
    """
    N-BEATS inspired model that accepts:
    - price_series : (batch, input_size)  — historical close prices
    - embedding    : (batch, emb_dim)     — Chronos embedding (optional)

    When embedding is provided, it is projected and added to the input
    before the N-BEATS blocks. This keeps the architecture clean and
    comparable to the no-embedding version.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        hidden_size: int = 128,
        n_blocks: int = 2,
        emb_dim: int = 0,  # 0 means no embedding
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.emb_dim = emb_dim

        # Project embedding to input_size so it can be added to price series
        if emb_dim > 0:
            self.emb_proj = nn.Sequential(
                nn.Linear(emb_dim, input_size),
                nn.ReLU(),
            )
        else:
            self.emb_proj = None

        self.blocks = nn.ModuleList([
            NBEATSBlock(input_size, hidden_size, horizon)
            for _ in range(n_blocks)
        ])

    def forward(self, price_series: torch.Tensor, embedding: torch.Tensor | None = None):
        # Normalize price series
        mean = price_series.mean(dim=1, keepdim=True)
        std = price_series.std(dim=1, keepdim=True) + 1e-8
        x = (price_series - mean) / std

        # Add embedding if provided
        if embedding is not None and self.emb_proj is not None:
            emb_projected = self.emb_proj(embedding)
            x = x + emb_projected

        # N-BEATS stack
        forecast = torch.zeros(x.shape[0], self.horizon, device=x.device)
        residual = x

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


# ---------------------------------------------------------------------------
# 4. Dataset for Training
# ---------------------------------------------------------------------------

class PriceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[dict],
        input_size: int,
        horizon: int,
        emb_dim: int = 0,
    ):
        self.samples = samples
        self.input_size = input_size
        self.horizon = horizon
        self.emb_dim = emb_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prices = np.array(s["prices"], dtype=np.float32)
        future = np.array(s["future_prices"], dtype=np.float32)

        x = torch.tensor(prices, dtype=torch.float32)

        # Target: future returns relative to last price in context
        last_price = prices[-1] + 1e-8
        y = torch.tensor(future / last_price - 1.0, dtype=torch.float32)

        if self.emb_dim > 0 and "embedding" in s:
            emb = torch.tensor(s["embedding"], dtype=torch.float32)
        else:
            emb = None

        return x, y, emb


def collate_fn(batch):
    xs, ys, embs = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    if embs[0] is not None:
        embs = torch.stack(embs)
    else:
        embs = None
    return xs, ys, embs


# ---------------------------------------------------------------------------
# 5. Training Function
# ---------------------------------------------------------------------------

def train_nbeats_model(
    model: NBEATSWithEmbedding,
    train_samples: list[dict],
    input_size: int,
    horizon: int,
    emb_dim: int,
    max_steps: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> NBEATSWithEmbedding:
    """Train the N-BEATS model on the given samples."""
    dataset = PriceDataset(train_samples, input_size, horizon, emb_dim)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()  # MAE loss — same as NeuralForecast default

    model.train()
    step = 0

    while step < max_steps:
        for xs, ys, embs in loader:
            if step >= max_steps:
                break
            optimizer.zero_grad()
            preds = model(xs, embs)
            loss = loss_fn(preds, ys)
            loss.backward()
            optimizer.step()
            step += 1

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 6. Main Run Function
# ---------------------------------------------------------------------------

def run_nbeats_with_embeddings(
    panel: pd.DataFrame,
    config,
    chronos_model_id: str = "amazon/chronos-t5-tiny",
    emb_dims: list[int] = [8, 16, 32],
) -> pd.DataFrame:
    """
    Walk-forward evaluation of N-BEATS with Chronos embeddings.

    For each horizon and cutoff date:
    1. Extract Chronos embeddings for all training windows
    2. Apply PCA to reduce to emb_dim dimensions (vary: 8, 16, 32)
    3. Train N-BEATS WITH embeddings
    4. Compare predictions to N-BEATS WITHOUT embeddings

    Parameters
    ----------
    panel : pd.DataFrame
        Prepared panel with columns: Ticker, Date, close
    config : BenchmarkConfig
        Same config used for other models
    chronos_model_id : str
        Chronos model to use for embedding extraction
    emb_dims : list[int]
        PCA dimensions to try (professor asked to vary this)

    Returns
    -------
    pd.DataFrame
        Predictions in same format as other models
    """
    extractor = ChronosEmbeddingExtractor(
        model_id=chronos_model_id,
        device="cpu",
    )

    all_predictions = []
    tickers = sorted(panel["Ticker"].unique())

    for horizon in config.horizons:
        cutoff_dates = make_cutoff_dates(
            panel,
            config.input_size,
            horizon,
            config.test_step,
            config.max_test_dates,
        )
        print(f"\nHorizon {horizon}: {len(cutoff_dates)} cutoff dates")

        # Pre-extract ALL embeddings for this horizon's cutoff dates
        emb_cache = extract_all_embeddings(
            panel, extractor, cutoff_dates, config.input_size
        )

        for cutoff_idx, cutoff_date in enumerate(cutoff_dates):
            print(f"  Cutoff {cutoff_idx+1}/{len(cutoff_dates)}: {cutoff_date.date()}")

            # Build training samples (all data BEFORE cutoff)
            train_samples_base = []  # without embeddings
            train_samples_emb = []   # with raw embeddings

            for ticker in tickers:
                g = panel[panel["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
                cutoff_matches = g.index[g["Date"] == cutoff_date].tolist()
                if not cutoff_matches:
                    continue
                cutoff_idx_in_g = cutoff_matches[0]

                # Use all windows BEFORE cutoff for training
                for start in range(config.input_size, cutoff_idx_in_g - horizon + 1):
                    prices = g.loc[start - config.input_size: start - 1, "close"].astype(float).values
                    future = g.loc[start: start + horizon - 1, "close"].astype(float).values

                    if len(prices) != config.input_size or len(future) != horizon:
                        continue

                    sample = {"prices": prices, "future_prices": future}
                    train_samples_base.append(sample)

                    # Add embedding if available
                    emb_key = (ticker, cutoff_date)
                    if emb_key in emb_cache:
                        sample_emb = dict(sample)
                        sample_emb["embedding"] = emb_cache[emb_key]
                        train_samples_emb.append(sample_emb)

            if len(train_samples_base) < 10:
                print(f"    Skipping: not enough training samples ({len(train_samples_base)})")
                continue

            # Raw embedding dimension from Chronos
            raw_emb_dim = list(emb_cache.values())[0].shape[0] if emb_cache else 256

            # ---------------------------------------------------------------
            # Train and predict for each PCA dimension
            # ---------------------------------------------------------------
            for emb_dim in emb_dims:
                if emb_dim >= raw_emb_dim:
                    emb_dim = raw_emb_dim

                # Apply PCA to reduce embedding dimension
                raw_embeddings = np.array([s["embedding"] for s in train_samples_emb])
                pca = PCA(n_components=emb_dim, random_state=config.random_seed)
                reduced_embeddings = pca.fit_transform(raw_embeddings)

                # Update samples with PCA-reduced embeddings
                train_samples_pca = []
                for i, s in enumerate(train_samples_emb):
                    s_copy = dict(s)
                    s_copy["embedding"] = reduced_embeddings[i]
                    train_samples_pca.append(s_copy)

                # Build and train model WITH embeddings
                torch.manual_seed(config.random_seed)
                model_emb = NBEATSWithEmbedding(
                    input_size=config.input_size,
                    horizon=horizon,
                    hidden_size=128,
                    n_blocks=2,
                    emb_dim=emb_dim,
                )
                model_emb = train_nbeats_model(
                    model_emb,
                    train_samples_pca,
                    config.input_size,
                    horizon,
                    emb_dim=emb_dim,
                    max_steps=config.max_steps,
                )

                # Also train model WITHOUT embeddings for direct comparison
                if emb_dim == emb_dims[0]:  # only once per cutoff date
                    torch.manual_seed(config.random_seed)
                    model_no_emb = NBEATSWithEmbedding(
                        input_size=config.input_size,
                        horizon=horizon,
                        hidden_size=128,
                        n_blocks=2,
                        emb_dim=0,
                    )
                    model_no_emb = train_nbeats_model(
                        model_no_emb,
                        train_samples_base,
                        config.input_size,
                        horizon,
                        emb_dim=0,
                        max_steps=config.max_steps,
                    )

                # ---------------------------------------------------------------
                # Generate predictions for each ticker at this cutoff date
                # ---------------------------------------------------------------
                for ticker in tickers:
                    g = panel[panel["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
                    cutoff_matches = g.index[g["Date"] == cutoff_date].tolist()
                    if not cutoff_matches:
                        continue
                    idx = cutoff_matches[0]

                    if idx + horizon >= len(g) or idx + 1 < config.input_size:
                        continue

                    context = g.loc[idx - config.input_size + 1: idx, "close"].astype(float).values
                    last_close = float(g.loc[idx, "close"])
                    actual_close = float(g.loc[idx + horizon, "close"])
                    target_date = pd.Timestamp(g.loc[idx + horizon, "Date"])
                    y_true = actual_close / last_close - 1.0

                    # Predict WITHOUT embedding (only once per cutoff)
                    if emb_dim == emb_dims[0]:
                        with torch.no_grad():
                            x = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                            pred_no_emb = model_no_emb(x, None)
                            # Model directly outputs return
                            y_pred_no_emb = float(pred_no_emb[0, horizon - 1].item())

                        all_predictions.append({
                            "Date": target_date,
                            "Ticker": ticker,
                            "Model": "NBEATS-NoEmb",
                            "Horizon": horizon,
                            "y_true": y_true,
                            "y_pred": y_pred_no_emb,
                            "cutoff_date": cutoff_date,
                            "last_close": last_close,
                            "pred_close": last_close * (1 + y_pred_no_emb),
                            "actual_close": actual_close,
                        })

                    # Predict WITH embedding
                    emb_key = (ticker, cutoff_date)
                    if emb_key not in emb_cache:
                        continue

                    raw_emb = emb_cache[emb_key].reshape(1, -1)
                    reduced_emb = pca.transform(raw_emb)

                    with torch.no_grad():
                        x = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                        emb_tensor = torch.tensor(reduced_emb, dtype=torch.float32)
                        pred_emb = model_emb(x, emb_tensor)
                        # Model directly outputs return
                        y_pred_emb = float(pred_emb[0, horizon - 1].item())

                    all_predictions.append({
                        "Date": target_date,
                        "Ticker": ticker,
                        "Model": f"NBEATS-Emb-{emb_dim}",
                        "Horizon": horizon,
                        "y_true": y_true,
                        "y_pred": y_pred_emb,
                        "cutoff_date": cutoff_date,
                        "last_close": last_close,
                        "pred_close": last_close * (1 + y_pred_emb),
                        "actual_close": actual_close,
                    })

    return pd.DataFrame(all_predictions)