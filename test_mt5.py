#!/usr/bin/env python
"""Test MT5 connection and data fetching."""

from src.mt5_feed import initialize, fetch_ohlcv, shutdown

try:
    print("Connecting to MT5 ...")
    if not initialize():
        print("Failed to connect to MT5. Make sure terminal is running.")
        exit(1)

    print("Fetching XAUUSD 15min data ...")
    df = fetch_ohlcv('XAUUSD', '15min', 100)

    if df.empty:
        print("No data received. Check if XAUUSD is enabled in MT5.")
    else:
        last_price = df['close'].iloc[-1]
        print(f"✅ Success! Last price: {last_price:.2f}")
        print(f"   Bars: {len(df)}")
        print(f"   Time: {df.index[-1]}")

finally:
    shutdown()
