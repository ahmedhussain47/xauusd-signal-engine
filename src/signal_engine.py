"""
signal_engine.py — Multi-timeframe trading signal generator.

Architecture
────────────
                 ┌───────────┐   macro bias (40 % weight)
  Higher TF ────►│ Bias score│
  (4H / 1D)     └─────┬─────┘
                       │
                 ┌─────▼─────┐   trend confirmation (30 % weight)
  Mid TF   ─────►│ Conf score│
  (1H)           └─────┬─────┘
                       │
                 ┌─────▼─────┐   entry precision (20 % weight)
  Lower TF ─────►│Entry score│
  (15min)        └─────┬─────┘
                       │
                 ┌─────▼─────┐   optional extra signal (≤15 pts)
  ML model ─────►│ Model vote│
                 └─────┬─────┘
                       │
                 ┌─────▼─────┐
                 │ Confidence │  0–100
                 └─────┬─────┘
                       │   ≥ threshold?
                 ┌─────▼─────┐
                 │   Signal   │  entry / TP / SL / direction
                 └───────────┘

Timeframe veto rule
───────────────────
The HIGHEST available timeframe acts as macro filter.  If its direction is
bearish, only SELL signals are allowed.  If it is neutral, both directions
are allowed but confidence is capped at 70.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .feature_engineering import atr, build_multi_tf_snapshot, pivot_levels, rsi as _rsi_fn

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

# Contribution weight of each timeframe to the confidence score.
# Must sum to 1.0 for the weights actually present.
TF_WEIGHTS: Dict[str, float] = {
    "1D":    0.40,
    "4H":    0.30,
    "1H":    0.20,
    "30min": 0.15,
    "15min": 0.10,
    "5min":  0.05,
    "1min":  0.03,
}

# How long a signal remains valid after generation.
SIGNAL_TTL: Dict[str, timedelta] = {
    "1min":  timedelta(hours=4),
    "5min":  timedelta(minutes=45),
    "15min": timedelta(hours=2),
    "30min": timedelta(hours=4),
    "1H":    timedelta(hours=8),
    "4H":    timedelta(hours=24),
    "1D":    timedelta(days=3),
}

# Canonical timeframe order (low → high resolution).
TF_ORDER = ["1min", "5min", "15min", "30min", "1H", "4H", "1D"]

# Forex assets identified by suffix (yfinance convention).
_FOREX_SUFFIXES = ("=X",)

# UTC hour ranges [start, end) for each major trading session.
_SESSION_HOURS = {
    "asian":    (0,  7),
    "london":   (7,  16),
    "new_york": (12, 21),
}


def _session_liquidity_multiplier(asset: str, now: Optional[datetime] = None) -> float:
    """
    Returns a confidence multiplier (0.7–1.0) for forex assets during
    low-liquidity sessions.  Non-forex assets always return 1.0.
    """
    if not any(asset.endswith(s) for s in _FOREX_SUFFIXES):
        return 1.0
    if now is None:
        now = datetime.now(timezone.utc)
    hour = now.hour
    # Only London / New York (or overlap) are high-liquidity for major pairs.
    in_london   = _SESSION_HOURS["london"][0]   <= hour < _SESSION_HOURS["london"][1]
    in_new_york = _SESSION_HOURS["new_york"][0] <= hour < _SESSION_HOURS["new_york"][1]
    if not (in_london or in_new_york):
        return 0.70   # Asian session: dampen forex signals
    return 1.0


def _detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> int:
    """
    Detect RSI divergence over the most-recent `lookback` bars.

    Returns:
        +1  bullish divergence (price lower low, RSI higher low)
        -1  bearish divergence (price higher high, RSI lower high)
         0  no clear divergence
    """
    if len(df) < lookback + 14:
        return 0
    try:
        recent    = df.tail(lookback)
        close     = recent["close"]
        rsi_vals  = _rsi_fn(df["close"], 14).reindex(recent.index)
        half      = lookback // 2

        first_close, second_close = close.iloc[:half], close.iloc[half:]
        first_rsi,   second_rsi   = rsi_vals.iloc[:half], rsi_vals.iloc[half:]

        # Bullish divergence: lower price low, higher RSI low
        if (second_close.min() < first_close.min() and
                second_rsi.min() > first_rsi.min()):
            return +1

        # Bearish divergence: higher price high, lower RSI high
        if (second_close.max() > first_close.max() and
                second_rsi.max() < first_rsi.max()):
            return -1
    except Exception:
        pass
    return 0


# ── Signal dataclass ───────────────────────────────────────────────────────────

@dataclass
class Signal:
    asset:        str
    timeframe:    str            # Entry timeframe
    signal:       str            # "BUY" | "SELL"
    entry:        float
    take_profit:  float
    stop_loss:    float
    confidence:   int            # 0 – 100
    timestamp:    str            # UTC ISO-format
    valid_until:  str            # UTC ISO-format
    atr_value:    float  = 0.0
    rr_ratio:     float  = 0.0
    sl_distance:  float  = 0.0
    tf_alignment: Dict   = field(default_factory=dict)
    model_pred:   Optional[float] = None

    # Convenience ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        return (
            f"[{self.timestamp}] {self.signal:4s} {self.asset:8s} | "
            f"Entry={self.entry:.5f}  TP={self.take_profit:.5f}  SL={self.stop_loss:.5f} | "
            f"Conf={self.confidence}%  R/R={self.rr_ratio:.1f}x  "
            f"Valid until {self.valid_until}"
        )


# ── Internal: timeframe scorer ─────────────────────────────────────────────────

def _score_timeframe(snap: pd.Series) -> Dict:
    """
    Evaluate the bullish/bearish bias of a single timeframe snapshot.

    Returns:
        direction: +1 (bullish) | -1 (bearish) | 0 (neutral)
        strength:  0.0 – 1.0  (how convincing the bias is)
        factors:   list of strings describing active signals
    """
    raw   = 0.0
    total = 0.0
    factors: List[str] = []
    close = float(snap.get("close", 0.0))

    def _vote(score: float, weight: float, label: str):
        nonlocal raw, total
        raw   += score * weight
        total += weight
        if abs(score) > 0:
            factors.append(label)

    # ── EMA alignment (price vs. stack) ───────────────────────────────────────
    try:
        e20, e50, e200 = snap["ema_20"], snap["ema_50"], snap["ema_200"]
        _vote(+1.0 if close > e20  else -1.0, 1.0, f"price>ema20"  if close > e20  else "price<ema20")
        _vote(+1.0 if e20  > e50   else -1.0, 0.8, "ema20>ema50"   if e20  > e50   else "ema20<ema50")
        _vote(+1.0 if e50  > e200  else -1.0, 0.8, "ema50>ema200"  if e50  > e200  else "ema50<ema200")
    except KeyError:
        pass

    # ── RSI momentum ──────────────────────────────────────────────────────────
    try:
        rv = float(snap["rsi_14"])
        if rv > 60:
            _vote(+1.0, 0.7, f"rsi_bull({rv:.0f})")
        elif rv < 40:
            _vote(-1.0, 0.7, f"rsi_bear({rv:.0f})")
        else:
            _vote(+(rv - 50) / 50, 0.3, f"rsi_neutral({rv:.0f})")
    except KeyError:
        pass

    # ── MACD ──────────────────────────────────────────────────────────────────
    try:
        hist = float(snap["macd_hist"])
        line = float(snap["macd_line"])
        sig  = float(snap["macd_signal"])
        if hist > 0 and line > sig:
            _vote(+1.0, 0.8, "macd_bull")
        elif hist < 0 and line < sig:
            _vote(-1.0, 0.8, "macd_bear")
        else:
            _vote(0.0,  0.2, "macd_mixed")
    except KeyError:
        pass

    # ── ADX / Directional movement ────────────────────────────────────────────
    try:
        adx_v    = float(snap["adx_14"])
        plus_di  = float(snap["plus_di"])
        minus_di = float(snap["minus_di"])
        if adx_v > 25:   # Trending — DI spread is meaningful
            w = min(adx_v / 50.0, 1.0)   # Stronger trend → more weight
            if plus_di > minus_di:
                _vote(+1.0, w, f"adx_up({adx_v:.0f})")
            else:
                _vote(-1.0, w, f"adx_dn({adx_v:.0f})")
    except KeyError:
        pass

    # ── Bollinger %B ──────────────────────────────────────────────────────────
    try:
        bb_pct = float(snap["bb_pct"])
        if bb_pct > 0.8:
            _vote(+0.5, 0.4, f"bb_upper({bb_pct:.2f})")   # Momentum up
        elif bb_pct < 0.2:
            _vote(-0.5, 0.4, f"bb_lower({bb_pct:.2f})")   # Momentum down
    except KeyError:
        pass

    # ── Stochastic ────────────────────────────────────────────────────────────
    try:
        k = float(snap["stoch_k"])
        d = float(snap["stoch_d"])
        if k > 50 and k > d:
            _vote(+0.5, 0.4, f"stoch_bull({k:.0f})")
        elif k < 50 and k < d:
            _vote(-0.5, 0.4, f"stoch_bear({k:.0f})")
    except KeyError:
        pass

    # ── Volume confirmation ────────────────────────────────────────────────
    # A volume spike in the direction of the bar's last-bar return confirms
    # momentum; low volume suggests weak conviction.
    try:
        vol_ratio = float(snap.get("volume_ratio", 1.0))
        ret_1     = float(snap.get("ret_1", 0.0))
        if vol_ratio > 1.5 and ret_1 != 0.0:
            # Spike confirms the bar's direction
            _vote(+0.6 if ret_1 > 0 else -0.6, 0.5, f"vol_confirm({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            _vote(0.0, 0.25, f"vol_weak({vol_ratio:.1f}x)")
    except (KeyError, TypeError):
        pass

    if total == 0.0:
        return {"direction": 0, "strength": 0.0, "factors": factors}

    normalised = raw / total          # −1.0 … +1.0
    direction  = (
        +1 if normalised >  0.15 else
        -1 if normalised < -0.15 else
         0
    )
    strength = min(abs(normalised), 1.0)
    return {"direction": direction, "strength": strength, "factors": factors}


# ── Internal: TP / SL calculator ──────────────────────────────────────────────

def _calc_tp_sl(
    direction:    str,
    entry:        float,
    atr_val:      float,
    support:      float,
    resistance:   float,
    rr_ratio:     float,
    atr_sl_mult:  float,
    max_sl_atr:   float = 0.0,   # >0 caps SL distance to this many ATRs
) -> Dict[str, float]:
    """
    Derive TP and SL using ATR-scaled distance, anchored to S/R when tighter.

    For BUY:
        SL = min( entry − ATR*mult,  support − 0.3*ATR )
        TP = entry + |entry − SL| * rr_ratio

    For SELL the logic is mirrored.

    max_sl_atr: when >0, SL distance is capped to atr_val * max_sl_atr.
    Use for 1min/5min entries so distant S/R pivots don't inflate the window.
    """
    if direction == "BUY":
        sl_atr  = entry - atr_val * atr_sl_mult
        sl_sr   = support - atr_val * 0.25        # Slightly below support
        sl      = min(sl_atr, sl_sr)              # Widest (most conservative)
        sl_dist = max(entry - sl, atr_val * 0.5)  # Never tighter than 0.5 ATR
        if max_sl_atr > 0:
            sl_dist = min(sl_dist, atr_val * max_sl_atr)
        sl = entry - sl_dist
        tp = entry + sl_dist * rr_ratio
    else:  # SELL
        sl_atr  = entry + atr_val * atr_sl_mult
        sl_sr   = resistance + atr_val * 0.25
        sl      = max(sl_atr, sl_sr)
        sl_dist = max(sl - entry, atr_val * 0.5)
        if max_sl_atr > 0:
            sl_dist = min(sl_dist, atr_val * max_sl_atr)
        sl = entry + sl_dist
        tp = entry - sl_dist * rr_ratio

    actual_rr = abs(tp - entry) / sl_dist if sl_dist > 0 else 0.0
    return {
        "take_profit": round(tp, 5),
        "stop_loss":   round(sl, 5),
        "rr_ratio":    round(actual_rr, 2),
        "sl_distance": round(sl_dist, 5),
    }


# ── Internal: confidence scorer ────────────────────────────────────────────────

def _compute_confidence(
    tf_scores:        Dict[str, Dict],
    final_direction:  int,
    available_tfs:    List[str],
    model_pred:       Optional[float],
    model_threshold:  float,
) -> int:
    """
    Combine per-timeframe scores into a 0–100 integer confidence score.

    Timeframes that agree with final_direction add to the score;
    opposing ones subtract (weighted by TF_WEIGHTS).
    Model prediction adds up to 15 bonus points.
    """
    # Normalise weights to only the timeframes we actually have
    present_weights = {tf: TF_WEIGHTS.get(tf, 0.05) for tf in available_tfs if tf in tf_scores}
    weight_sum = sum(present_weights.values())
    if weight_sum == 0.0:
        return 0

    weighted_score = 0.0
    for tf, w_raw in present_weights.items():
        w    = w_raw / weight_sum     # Re-normalise
        info = tf_scores[tf]
        if info["direction"] == final_direction:
            weighted_score += w * info["strength"]
        elif info["direction"] == 0:
            weighted_score += w * 0.2     # Neutral: small positive contribution
        else:
            weighted_score -= w * info["strength"]

    # Map [−1, +1] → [0, 85]
    base_conf = int(((weighted_score + 1.0) / 2.0) * 85.0)
    base_conf = max(0, min(85, base_conf))

    # Model bonus (up to 15 pts)
    model_bonus = 0
    if model_pred is not None and model_threshold > 0:
        ratio = abs(model_pred) / model_threshold
        # Only grant bonus if model agrees with direction
        agrees = (final_direction == +1 and model_pred > model_threshold) or \
                 (final_direction == -1 and model_pred < -model_threshold)
        if agrees:
            model_bonus = min(15, int(ratio * 10))

    return min(99, base_conf + model_bonus)


# ── Main signal engine ─────────────────────────────────────────────────────────

class SignalEngine:
    """
    Orchestrates multi-timeframe analysis and outputs tradeable signals.

    Parameters
    ──────────
    asset:               Symbol string, e.g. "XAUUSD", "EURUSD", "AAPL".
    timeframes:          Ordered list of TFs to use (lowest → highest).
                         Default: ["15min", "1H", "4H", "1D"].
    confidence_threshold Minimum confidence (0–100) required to emit a signal.
    rr_ratio:            Target risk-reward ratio for TP placement.
    atr_sl_mult:         ATR multiplier for SL placement.
    model_fn:            Optional callable: (df: pd.DataFrame) → float
                         Should return the predicted 1-period return.
                         Positive = bullish, negative = bearish.
    model_threshold:     Absolute return threshold above which the model
                         prediction is considered directional.
    """

    DEFAULT_TIMEFRAMES = ["15min", "1H", "4H", "1D"]

    def __init__(
        self,
        asset:                str,
        timeframes:           Optional[List[str]] = None,
        confidence_threshold: int   = 60,
        rr_ratio:             float = 2.0,
        atr_sl_mult:          float = 1.5,
        model_fn:             Optional[Callable[[pd.DataFrame], float]] = None,
        model_threshold:      float = 0.003,
    ):
        self.asset                = asset
        self.rr_ratio             = rr_ratio
        self.atr_sl_mult          = atr_sl_mult
        self.confidence_threshold = confidence_threshold
        self.model_fn             = model_fn
        self.model_threshold      = model_threshold

        raw_tfs = timeframes or self.DEFAULT_TIMEFRAMES
        self.timeframes = [tf for tf in TF_ORDER if tf in raw_tfs]
        if not self.timeframes:
            self.timeframes = raw_tfs   # Keep as-is if none matched

        # Cooldown: suppress repeated signals within one TTL window.
        self._last_signal_ts: Optional[datetime] = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def entry_tf(self) -> str:
        """Lowest (fastest) timeframe — used for entry price and timing."""
        return self.timeframes[0]

    @property
    def bias_tf(self) -> str:
        """Highest (slowest) timeframe — sets the macro direction veto."""
        return self.timeframes[-1]

    # ── Main method ────────────────────────────────────────────────────────────

    def generate(self, tf_data: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        """
        Generate a trading signal from multi-timeframe OHLCV data.

        Args:
            tf_data: Dict mapping timeframe string → raw OHLCV DataFrame.
                     Use fetch_multi_timeframe() from data_feeds to build this.

        Returns:
            Signal if all conditions are met, None otherwise.
        """
        now = datetime.now(timezone.utc)

        # 1. Compute feature snapshots for each available timeframe
        snapshots = build_multi_tf_snapshot(
            {tf: df for tf, df in tf_data.items() if tf in self.timeframes}
        )
        if not snapshots:
            logger.debug("%s: no usable snapshots", self.asset)
            return None

        available_tfs = [tf for tf in self.timeframes if tf in snapshots]

        # Volatility regime filter — skip dormant or panic markets
        entry_snap_early = snapshots.get(self.entry_tf, snapshots[available_tfs[0]])
        vol_regime = float(entry_snap_early.get("vol_regime", 1.0))
        if vol_regime > 3.0:
            logger.debug("%s: extreme volatility (vol_regime=%.1f) — no signal", self.asset, vol_regime)
            return None
        if vol_regime < 0.2:
            logger.debug("%s: market dormant (vol_regime=%.1f) — no signal", self.asset, vol_regime)
            return None

        # 2. Score every timeframe
        tf_scores = {tf: _score_timeframe(snap) for tf, snap in snapshots.items()}

        # 3. Determine macro bias from the highest available timeframe
        highest_tf = next(
            (tf for tf in reversed(available_tfs) if tf in tf_scores), None
        )
        if highest_tf is None:
            return None

        bias_direction = tf_scores[highest_tf]["direction"]

        # Veto rule: if macro is neutral, relax but cap confidence later
        macro_neutral = (bias_direction == 0)
        if macro_neutral:
            # Determine direction from mid-timeframe if available
            mid_candidates = available_tfs[:-1]
            for tf in reversed(mid_candidates):
                d = tf_scores[tf]["direction"]
                if d != 0:
                    bias_direction = d
                    break
            if bias_direction == 0:
                logger.debug("%s: all timeframes neutral — no signal", self.asset)
                return None

        final_direction = bias_direction
        direction_str   = "BUY" if final_direction == 1 else "SELL"

        # 4. Entry timeframe must not oppose the bias
        entry_dir = tf_scores.get(self.entry_tf, {}).get("direction", 0)
        if entry_dir != 0 and entry_dir != final_direction:
            logger.debug(
                "%s: entry TF (%s) contradicts macro bias — skipped",
                self.asset, self.entry_tf,
            )
            return None

        # 5. Optional model prediction
        model_pred = None
        if self.model_fn is not None:
            entry_df = tf_data.get(self.entry_tf, next(iter(tf_data.values()), None))
            if entry_df is not None and not entry_df.empty:
                try:
                    model_pred = float(self.model_fn(entry_df))
                except Exception as exc:
                    logger.warning("%s: model inference failed: %s", self.asset, exc)

        # 6. Confidence
        confidence = _compute_confidence(
            tf_scores       = tf_scores,
            final_direction = final_direction,
            available_tfs   = available_tfs,
            model_pred      = model_pred,
            model_threshold = self.model_threshold,
        )
        if macro_neutral:
            confidence = min(confidence, 70)     # Cap when macro is unclear

        # RSI divergence bonus / penalty
        entry_df_raw = tf_data.get(self.entry_tf, next(iter(tf_data.values())))
        divergence = _detect_rsi_divergence(entry_df_raw)
        if divergence == final_direction:
            confidence = min(99, confidence + 8)
        elif divergence != 0:
            confidence = max(0, confidence - 5)

        # Market session multiplier (dampens forex during Asian session)
        session_mult = _session_liquidity_multiplier(self.asset, now)
        if session_mult < 1.0:
            confidence = int(confidence * session_mult)

        if confidence < self.confidence_threshold:
            logger.debug(
                "%s: confidence %d < threshold %d",
                self.asset, confidence, self.confidence_threshold,
            )
            return None

        # Cooldown: suppress signal if we emitted one within the last TTL window
        cooldown = SIGNAL_TTL.get(self.entry_tf, timedelta(hours=2))
        if self._last_signal_ts is not None and (now - self._last_signal_ts) < cooldown:
            logger.debug(
                "%s: cooldown active (%s remaining)",
                self.asset, cooldown - (now - self._last_signal_ts),
            )
            return None

        # 7. Entry price and ATR from the entry timeframe
        entry_snap  = snapshots.get(self.entry_tf, snapshots[available_tfs[0]])
        entry_price = float(entry_snap["close"])
        atr_val     = float(entry_snap.get("atr_14", entry_price * 0.005))

        # 8. Support / resistance + Fibonacci levels from entry timeframe data
        entry_df = tf_data.get(self.entry_tf, next(iter(tf_data.values())))
        pivots   = pivot_levels(entry_df, lookback=20)

        # Tighten S/R using nearest Fibonacci level when it lies between
        # the price and the classic pivot (avoids over-wide SL/TP).
        fib_vals = [v for k, v in pivots.items() if k.startswith("fib_")]
        if fib_vals:
            if direction_str == "BUY":
                fib_sup = max((f for f in fib_vals if f < entry_price), default=None)
                if fib_sup is not None and fib_sup > pivots["support"]:
                    pivots["support"] = fib_sup
            else:
                fib_res = min((f for f in fib_vals if f > entry_price), default=None)
                if fib_res is not None and fib_res < pivots["resistance"]:
                    pivots["resistance"] = fib_res

        # 9. TP / SL
        # Hard-cap SL distance by TF so distant S/R pivots never inflate risk.
        _sl_caps = {"1min": 1.2, "5min": 1.5, "15min": 2.0}
        max_sl_atr = _sl_caps.get(self.entry_tf, 0.0)
        tp_sl = _calc_tp_sl(
            direction   = direction_str,
            entry       = entry_price,
            atr_val     = atr_val,
            support     = pivots["support"],
            resistance  = pivots["resistance"],
            rr_ratio    = self.rr_ratio,
            atr_sl_mult = self.atr_sl_mult,
            max_sl_atr  = max_sl_atr,
        )

        # 10. Validity window
        valid_until = now + SIGNAL_TTL.get(self.entry_tf, timedelta(hours=2))

        signal = Signal(
            asset        = self.asset,
            timeframe    = self.entry_tf,
            signal       = direction_str,
            entry        = round(entry_price, 5),
            take_profit  = tp_sl["take_profit"],
            stop_loss    = tp_sl["stop_loss"],
            confidence   = confidence,
            timestamp    = now.strftime("%Y-%m-%d %H:%M"),
            valid_until  = valid_until.strftime("%Y-%m-%d %H:%M"),
            atr_value    = round(atr_val, 5),
            rr_ratio     = tp_sl["rr_ratio"],
            sl_distance  = tp_sl["sl_distance"],
            tf_alignment = {
                tf: {
                    "direction": s["direction"],
                    "strength":  round(s["strength"], 2),
                    "factors":   s["factors"],
                }
                for tf, s in tf_scores.items()
            },
            model_pred   = (
                round(model_pred, 6) if model_pred is not None else None
            ),
        )

        self._last_signal_ts = now
        logger.info("Signal generated: %s", signal)
        return signal

    # ── Batch generation ───────────────────────────────────────────────────────

    def scan_history(
        self,
        tf_data:       Dict[str, pd.DataFrame],
        step_bars:     int = 10,
        min_conf:      Optional[int] = None,
    ) -> List[Signal]:
        """
        Generate signals across historical data by rolling the window forward.

        Useful for building a dataset of historical signals for backtesting.

        Args:
            tf_data:   Full historical data per timeframe.
            step_bars: How many entry-TF bars to advance per iteration.
            min_conf:  Override confidence threshold for this scan.
        """
        if min_conf is not None:
            orig = self.confidence_threshold
            self.confidence_threshold = min_conf

        entry_df_full = tf_data.get(self.entry_tf)
        if entry_df_full is None or entry_df_full.empty:
            return []

        signals: List[Signal] = []
        n = len(entry_df_full)
        # Warm-up: EMA-200 needs 200 bars. If we have fewer, start at 60% of n.
        warmup = min(200, int(n * 0.6))
        start  = max(warmup, step_bars * 2)   # always leave room for at least 2 steps
        if start >= n:
            return signals

        for i in range(start, n, step_bars):
            sliced = {
                tf: df[df.index <= entry_df_full.index[i]]
                for tf, df in tf_data.items()
            }
            sig = self.generate(sliced)
            if sig is not None:
                # Stamp signal with the bar's actual timestamp so the
                # backtester can find future price bars after this point.
                bar_ts = entry_df_full.index[i]
                if hasattr(bar_ts, "strftime"):
                    sig.timestamp = bar_ts.strftime("%Y-%m-%d %H:%M")
                signals.append(sig)

        if min_conf is not None:
            self.confidence_threshold = orig

        return signals
