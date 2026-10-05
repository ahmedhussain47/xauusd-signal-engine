#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# ENGINE 7 — GOLD PRO LIVE TRADING DAEMON
# Continuous live signal generation with MT5 auto-trading
# ════════════════════════════════════════════════════════════════════════════════

import sys
import os
import time
import pickle
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Engine7Live - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('engine7_live.log')
    ]
)
logger = logging.getLogger()

print('════════════════════════════════════════════════════════════════════════════════')
print('ENGINE 7 — GOLD PRO LIVE TRADING DAEMON')
print('════════════════════════════════════════════════════════════════════════════════')

# ─── CONFIGURATION ───────────────────────────────────────────────────────────────
SYMBOL = 'XAUUSD'
TIMEFRAME = '1H'
CHECK_INTERVAL = 300  # Check every 5 minutes (300 seconds)
MODEL_CACHE_FILE = Path('models/engine7_models.pkl')

# MT5 Credentials
MT5_ACCOUNT = int(os.getenv('MT5_ACCOUNT', '0'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', 'Exness-MT5Trial15')

# Discord Webhook
DISCORD_WEBHOOK = os.getenv('DISCORD_WEBHOOK_GOLD', '')

# Trading Parameters
LIVE_MODE = True  # Set to False for demo
ENTRY_CONFIDENCE_THRESHOLD = 50
MIN_CONFLUENCE = 2
MAX_POSITION_SIZE = 0.05

print(f'\nConfiguration:')
print(f'  Symbol: {SYMBOL}')
print(f'  Timeframe: {TIMEFRAME}')
print(f'  Check Interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL//60} min)')
print(f'  Account: {MT5_ACCOUNT}')
print(f'  Server: {MT5_SERVER}')
print(f'  Live Mode: {"ENABLED" if LIVE_MODE else "DEMO"}')
print(f'  Discord: {"ENABLED" if DISCORD_WEBHOOK else "DISABLED"}')

# ─── MT5 CONNECTION ──────────────────────────────────────────────────────────────
def connect_mt5():
    """Connect to MT5 terminal."""
    try:
        if not mt5.initialize(
            path='C:\\Program Files\\MetaTrader 5\\terminal64.exe',
            login=MT5_ACCOUNT,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
            timeout=5000
        ):
            logger.error(f'MT5 init failed: {mt5.last_error()}')
            return False

        logger.info(f'✓ Connected to MT5: {MT5_ACCOUNT}@{MT5_SERVER}')
        return True
    except Exception as e:
        logger.error(f'MT5 connection error: {e}')
        return False

def disconnect_mt5():
    """Disconnect from MT5."""
    try:
        mt5.shutdown()
        logger.info('MT5 disconnected')
    except:
        pass

# ─── DATA LOADING ────────────────────────────────────────────────────────────────
def get_latest_data(symbol: str, timeframe: str = 'H1', bars: int = 100) -> pd.DataFrame:
    """Get latest OHLCV data from MT5."""
    try:
        if not mt5.initialize():
            return None

        tf_map = {
            '15min': mt5.TIMEFRAME_M15,
            '1H': mt5.TIMEFRAME_H1,
            '4H': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1,
        }

        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)

        if rates is None or len(rates) == 0:
            logger.warning(f'No data from MT5 for {symbol}')
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'tick_volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        return df
    except Exception as e:
        logger.error(f'Data loading error: {e}')
        return None

# ─── SIGNAL GENERATION (SIMPLIFIED) ──────────────────────────────────────────────
def generate_signal_simple(df: pd.DataFrame) -> dict:
    """
    Generate simple signal based on RSI + MACD + Price action.
    Returns: {'signal': 'BUY'/'SELL'/None, 'confidence': 0-100, 'entry': price, 'sl': price, 'tp': price}
    """
    if df is None or len(df) < 50:
        logger.warning(f'❌ BLOCKED: Insufficient data (need 50 bars, got {len(df) if df is not None else 0})')
        return None

    try:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # RSI (14)
        deltas = np.diff(close)
        seed = deltas[:14]
        up = seed[seed >= 0].sum() / 14
        down = -seed[seed < 0].sum() / 14
        rs = up / (down + 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        for i in range(14, len(close)):
            delta = close[i] - close[i-1]
            if delta > 0:
                up = (up * 13 + delta) / 14
                down = (down * 13) / 14
            else:
                up = (up * 13) / 14
                down = (down * 13 - delta) / 14
            rs = up / (down + 1e-6)
            rsi = np.append(rsi, 100.0 - (100.0 / (1.0 + rs)))

        rsi_current = rsi[-1]

        # MACD (12, 26, 9)
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26
        macd_signal = pd.Series(macd).ewm(span=9).mean().values
        macd_hist = macd - macd_signal

        macd_direction = macd_hist[-1] > 0

        # ATR (14) for SL/TP
        tr = np.maximum(high[1:] - low[1:],
                        np.maximum(high[1:] - close[:-1], close[:-1] - low[1:]))
        atr = np.mean(tr[-14:])

        current_price = close[-1]

        # Detailed logging
        logger.info(f'📊 Market Analysis:')
        logger.info(f'   Current Price: ${current_price:.2f}')
        logger.info(f'   RSI(14): {rsi_current:.2f} (Oversold:<30, Neutral:30-70, Overbought:>70)')
        logger.info(f'   MACD Hist: {macd_hist[-1]:.4f} (Bullish:>0, Bearish:<0)')
        logger.info(f'   ATR(14): {atr:.2f}')

        # Signal generation
        signal = None
        confidence = 0

        # ─── BUY SIGNAL ───────────────────────────────────────────────────────
        if rsi_current < 30:
            logger.info(f'✓ RSI Condition: {rsi_current:.2f} < 30 (OVERSOLD)')
            if macd_direction:
                logger.info(f'✓ MACD Condition: Bullish (Hist > 0)')
                signal = 'BUY'
                confidence = min(int((1 - rsi_current/30) * 100), 100)
            else:
                logger.warning(f'❌ BLOCKED: RSI oversold but MACD bearish')
        else:
            logger.info(f'❌ RSI Condition FAILED: {rsi_current:.2f} not < 30')

        # ─── SELL SIGNAL ──────────────────────────────────────────────────────
        if rsi_current > 70:
            logger.info(f'✓ RSI Condition: {rsi_current:.2f} > 70 (OVERBOUGHT)')
            if not macd_direction:
                logger.info(f'✓ MACD Condition: Bearish (Hist < 0)')
                signal = 'SELL'
                confidence = min(int((rsi_current/70 - 1) * 100), 100)
            else:
                logger.warning(f'❌ BLOCKED: RSI overbought but MACD bullish')
        else:
            logger.info(f'❌ RSI Condition FAILED: {rsi_current:.2f} not > 70')

        # ─── CONFIDENCE FILTER ────────────────────────────────────────────────
        if signal:
            logger.info(f'📈 Signal Generated: {signal}')
            logger.info(f'   Confidence: {confidence}%')
            logger.info(f'   Threshold: {ENTRY_CONFIDENCE_THRESHOLD}%')

            if confidence < ENTRY_CONFIDENCE_THRESHOLD:
                logger.warning(f'❌ BLOCKED: Confidence {confidence}% < {ENTRY_CONFIDENCE_THRESHOLD}%')
                return None

            sl_distance = atr * 1.5
            tp_distance = sl_distance * 2.0

            result = None
            if signal == 'BUY':
                result = {
                    'signal': 'BUY',
                    'entry': current_price,
                    'stop_loss': current_price - sl_distance,
                    'take_profit': current_price + tp_distance,
                    'confidence': confidence,
                    'atr': atr,
                    'rsi': rsi_current,
                    'macd_hist': macd_hist[-1]
                }
            else:
                result = {
                    'signal': 'SELL',
                    'entry': current_price,
                    'stop_loss': current_price + sl_distance,
                    'take_profit': current_price - tp_distance,
                    'confidence': confidence,
                    'atr': atr,
                    'rsi': rsi_current,
                    'macd_hist': macd_hist[-1]
                }

            logger.info(f'✅ SIGNAL APPROVED: Entry=${result["entry"]:.2f}, SL=${result["stop_loss"]:.2f}, TP=${result["take_profit"]:.2f}')
            return result
        else:
            logger.info(f'❌ NO SIGNAL: Market in neutral zone (RSI: {rsi_current:.2f}, MACD Dir: {macd_direction})')

        return None

    except Exception as e:
        logger.error(f'Signal generation error: {e}')
        return None

# ─── DISCORD ALERT ──────────────────────────────────────────────────────────────
def send_discord_alert(signal: dict, title: str = 'ENTRY'):
    """Send signal to Discord."""
    if not DISCORD_WEBHOOK or signal is None:
        return

    try:
        embed = {
            'title': f'🏆 Engine 7 Gold Pro — {title}',
            'color': 16711680 if signal['signal'] == 'BUY' else 255,
            'fields': [
                {'name': 'Direction', 'value': signal['signal'], 'inline': True},
                {'name': 'Entry', 'value': f"${signal['entry']:.2f}", 'inline': True},
                {'name': 'Confidence', 'value': f"{signal['confidence']}%", 'inline': True},
                {'name': 'SL', 'value': f"${signal['stop_loss']:.2f}", 'inline': True},
                {'name': 'TP', 'value': f"${signal['take_profit']:.2f}", 'inline': True},
                {'name': 'ATR', 'value': f"{signal['atr']:.2f}", 'inline': True},
                {'name': 'RSI', 'value': f"{signal['rsi']:.2f}", 'inline': True},
                {'name': 'MACD Hist', 'value': f"{signal['macd_hist']:.4f}", 'inline': True},
            ],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        payload = {'embeds': [embed]}
        response = requests.post(DISCORD_WEBHOOK, json=payload, timeout=5)

        if response.status_code == 204:
            logger.info(f'✓ Discord alert sent: {signal["signal"]}')
        else:
            logger.warning(f'Discord alert failed: {response.status_code}')

    except Exception as e:
        logger.error(f'Discord error: {e}')

# ─── AUTO TRADE ──────────────────────────────────────────────────────────────────
def place_trade(signal: dict) -> bool:
    """Place trade via MT5."""
    if not LIVE_MODE:
        logger.info(f'[DEMO] Signal: {signal["signal"]} @ ${signal["entry"]:.2f} | SL: ${signal["stop_loss"]:.2f} | TP: ${signal["take_profit"]:.2f}')
        return True

    try:
        if not mt5.initialize():
            logger.error('MT5 not initialized')
            return False

        position_size = 0.1  # Lot size

        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': SYMBOL,
            'volume': position_size,
            'type': mt5.ORDER_TYPE_BUY if signal['signal'] == 'BUY' else mt5.ORDER_TYPE_SELL,
            'price': signal['entry'],
            'sl': signal['stop_loss'],
            'tp': signal['take_profit'],
            'comment': f'Engine7-{signal["signal"]}'
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f'Trade failed: {result.comment if result else "Unknown error"}')
            return False

        logger.info(f'✓ Trade placed: {signal["signal"]} @ ${signal["entry"]:.2f}')
        return True

    except Exception as e:
        logger.error(f'Trade execution error: {e}')
        return False

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────────
def main_loop():
    """Continuous live trading loop."""
    logger.info('Starting live trading loop...')
    logger.info(f'Checking for signals every {CHECK_INTERVAL} seconds')

    last_signal_time = None
    last_signal = None
    cooldown_period = 300  # 5 minute cooldown between trades

    while True:
        try:
            # Get latest data
            df = get_latest_data(SYMBOL, TIMEFRAME, 100)

            if df is None:
                logger.warning('Failed to fetch data, retrying...')
                time.sleep(CHECK_INTERVAL)
                continue

            current_time = datetime.now(timezone.utc)

            # Generate signal
            logger.info(f'\n{"="*80}')
            logger.info(f'SIGNAL CHECK: {current_time.strftime("%Y-%m-%d %H:%M:%S UTC")}')
            logger.info(f'{"="*80}')

            signal = generate_signal_simple(df)

            # ─── COOLDOWN FILTER ──────────────────────────────────────────────
            if signal:
                if last_signal_time:
                    time_since_last = (current_time - last_signal_time).total_seconds()
                    if time_since_last < cooldown_period:
                        logger.warning(f'❌ BLOCKED: Cooldown active ({cooldown_period - time_since_last:.0f}s remaining)')
                        time.sleep(CHECK_INTERVAL)
                        continue

                logger.info(f'\n✅✅✅ SIGNAL APPROVED ✅✅✅')
                logger.info(f'Direction: {signal["signal"]}')
                logger.info(f'Entry: ${signal["entry"]:.2f}')
                logger.info(f'SL: ${signal["stop_loss"]:.2f}')
                logger.info(f'TP: ${signal["take_profit"]:.2f}')
                logger.info(f'Confidence: {signal["confidence"]}%')

                # Send Discord alert
                logger.info(f'Sending Discord alert...')
                send_discord_alert(signal, 'ENTRY')

                # Place trade
                logger.info(f'Placing trade...')
                if place_trade(signal):
                    last_signal_time = current_time
                    last_signal = signal
                    logger.info(f'✅ Trade executed successfully!')
                else:
                    logger.error(f'❌ Trade execution failed')
            else:
                logger.info(f'❌ NO SIGNAL GENERATED - Market conditions not met\n')

            logger.info(f'Next check in {CHECK_INTERVAL}s...\n')
            # Wait for next check
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info('Shutting down...')
            disconnect_mt5()
            break

        except Exception as e:
            logger.error(f'Main loop error: {e}')
            time.sleep(CHECK_INTERVAL)

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '═' * 80)
    print('LIVE TRADING DAEMON STARTING')
    print('═' * 80)

    try:
        # Connect to MT5
        if connect_mt5():
            print('✓ MT5 connected successfully')
            print('✓ Starting continuous monitoring...')
            print('\nPress Ctrl+C to stop\n')
            main_loop()
        else:
            print('✗ Failed to connect to MT5')
            print('✗ Make sure MetaTrader 5 terminal is running')
            sys.exit(1)

    except Exception as e:
        logger.error(f'Fatal error: {e}')
        sys.exit(1)
