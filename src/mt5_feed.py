"""
mt5_feed.py — Real-time OHLCV data from MetaTrader 5.

Replaces yfinance with MT5 for live trading (no data delay).
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("MetaTrader5 not installed. Run: pip install MetaTrader5")

logger = logging.getLogger(__name__)

# MT5 symbol names (match trading platforms)
MT5_SYMBOL_MAP: Dict[str, str] = {
    "XAUUSD": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDCHF": "USDCHF",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "NZDUSD": "NZDUSD",
    "GBPJPY": "GBPJPY",
    "EURJPY": "EURJPY",
    "EURGBP": "EURGBP",
}

# MT5 timeframe constants
_MT5_TF: Dict[str, int] = {
    "1min":  mt5.TIMEFRAME_M1,
    "5min":  mt5.TIMEFRAME_M5,
    "15min": mt5.TIMEFRAME_M15,
    "30min": mt5.TIMEFRAME_M30,
    "1H":    mt5.TIMEFRAME_H1,
    "4H":    mt5.TIMEFRAME_H4,
    "1D":    mt5.TIMEFRAME_D1,
}

# Minimum bars for each timeframe
_MIN_BARS: Dict[str, int] = {
    "1min":  100,
    "5min":  50,
    "15min": 50,
    "30min": 50,
    "1H":    50,
    "4H":    30,
    "1D":    30,
}


def _load_config() -> Dict:
    """Load MT5 credentials from .mt5_config.json"""
    config_path = Path(__file__).parent.parent / ".mt5_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"MT5 config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def initialize() -> bool:
    """Connect to MT5 terminal. Returns True if successful."""
    config = _load_config()
    # Try without path (auto-detect), then with common paths
    paths_to_try = [
        None,  # Auto-detect
        "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
    ]

    for path in paths_to_try:
        try:
            if mt5.initialize(login=config["login"], password=config["password"], server=config["server"], path=path):
                logger.info(f"MT5 connected: {config['server']}")
                return True
        except:
            continue

    error = mt5.last_error()
    logger.error(f"MT5 init failed: {error}")
    return False


def shutdown():
    """Disconnect from MT5."""
    mt5.shutdown()


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    bars: int = 300,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from MT5.

    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        timeframe: '1min', '5min', '15min', '30min', '1H', '4H', '1D'
        bars: Number of bars to fetch

    Returns:
        DataFrame with OHLCV + datetime index (CET timezone).
    """
    if not mt5.terminal_info():
        if not initialize():
            return pd.DataFrame()

    mt5_symbol = MT5_SYMBOL_MAP.get(symbol.upper(), symbol)
    mt5_tf = _MT5_TF.get(timeframe)

    if mt5_tf is None:
        logger.error(f"Unknown timeframe: {timeframe}")
        return pd.DataFrame()

    # Ensure symbol is selected (available for quotes)
    if not mt5.symbol_select(mt5_symbol, True):
        logger.error(f"Symbol not found: {mt5_symbol}")
        return pd.DataFrame()

    # Fetch bars
    try:
        rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, bars)
        if rates is None or len(rates) < _MIN_BARS.get(timeframe, 10):
            logger.warning(f"Insufficient bars for {mt5_symbol} @ {timeframe}: got {len(rates) if rates else 0}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.index = df.index.tz_convert("Europe/Berlin")  # Convert to CET (DST-aware)

        # Rename columns to match expected format
        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
        })

        return df[["open", "high", "low", "close", "volume"]].copy()

    except Exception as e:
        logger.error(f"MT5 fetch failed for {mt5_symbol} @ {timeframe}: {e}")
        return pd.DataFrame()


def fetch_multi_timeframe(
    symbol: str,
    timeframes: List[str],
    bars: int = 300,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for several timeframes.

    Returns:
        Dict mapping timeframe → OHLCV DataFrame.
    """
    if not mt5.terminal_info():
        if not initialize():
            return {}

    result: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf, bars=bars)
        if not df.empty:
            result[tf] = df
        else:
            logger.warning(f"Skipping {symbol} @ {tf} — no usable data")

    return result


def get_tick_price(symbol: str) -> Optional[float]:
    """Get current bid/ask spread (tick) for a symbol. Returns mid-price."""
    if not mt5.terminal_info():
        if not initialize():
            return None

    mt5_symbol = MT5_SYMBOL_MAP.get(symbol.upper(), symbol)
    tick = mt5.symbol_info_tick(mt5_symbol)

    if tick is None:
        logger.error(f"Could not fetch tick for {mt5_symbol}")
        return None

    return (tick.bid + tick.ask) / 2.0
