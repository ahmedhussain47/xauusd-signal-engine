"""
Example: Send trading signals to Telegram

This script demonstrates how to integrate Telegram notifications
with your signal engine for real-time trade alerts.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your Telegram credentials here or via environment variables
# os.environ["TELEGRAM_BOT_TOKEN"] = "your_bot_token"
# os.environ["TELEGRAM_CHAT_ID"] = "your_chat_id"


def example_1_auto_notify():
    """Auto-notify signals using SignalEngine."""
    from src.signal_engine import SignalEngine
    from src.data_feeds import fetch_multi_timeframe

    logger.info("Example 1: Auto-notify signals")

    # Create engine with Telegram notifications enabled
    engine = SignalEngine(
        asset="EURUSD=X",
        timeframes=["15min", "1H", "4H", "1D"],
        confidence_threshold=60,
        notify_telegram=True,  # <- Enable Telegram
    )

    # Fetch data and generate signal
    # (You'll need real data from your data feed)
    # tf_data = fetch_multi_timeframe("EURUSD=X", ["15min", "1H", "4H", "1D"])
    # signal = engine.generate(tf_data)
    # If signal is generated, it will automatically send to Telegram!

    print("✓ Engine configured with Telegram notifications")


def example_2_manual_notify():
    """Manually send signals to Telegram when ready."""
    from src.signal_engine import SignalEngine

    logger.info("Example 2: Manual notification")

    # Create engine without auto-notify
    engine = SignalEngine(
        asset="AAPL",
        timeframes=["15min", "1H"],
        notify_telegram=False,  # <- Disable auto-notify
    )

    # You can still send signals manually
    # signal = engine.generate(tf_data)
    # if signal and signal.confidence > 70:
    #     signal.send_to_telegram()  # Manual send

    print("✓ Ready to send signals manually with signal.send_to_telegram()")


def example_3_direct_notifier():
    """Use the notifier directly without signal engine."""
    from src.telegram_notifier import get_telegram_notifier

    logger.info("Example 3: Direct notifier usage")

    notifier = get_telegram_notifier()

    # Check if configured
    if not notifier.is_configured():
        print("⚠️  Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return

    # Send a signal
    success = notifier.send_signal(
        asset="GBPUSD=X",
        direction="BUY",
        entry=1.27500,
        take_profit=1.28000,
        stop_loss=1.27000,
        confidence=72,
        timeframe="1H",
        atr_value=0.00120,
        rr_ratio=2.50,
    )

    if success:
        print("✓ Signal sent to Telegram")
    else:
        print("✗ Failed to send signal")

    # Send an error notification
    notifier.send_error("Connection lost to data feed for 5 minutes")

    # Send a status update
    notifier.send_status("Signal engine started. Monitoring 15 forex pairs.")


def example_4_batch_processing():
    """Process multiple signals and send them to Telegram."""
    from src.telegram_notifier import get_telegram_notifier

    logger.info("Example 4: Batch processing with notifications")

    notifier = get_telegram_notifier()

    # Example signals
    signals_data = [
        {
            "asset": "EURUSD=X",
            "direction": "BUY",
            "entry": 1.08500,
            "take_profit": 1.09000,
            "stop_loss": 1.08000,
            "confidence": 78,
            "timeframe": "1H",
        },
        {
            "asset": "GBPUSD=X",
            "direction": "SELL",
            "entry": 1.27500,
            "take_profit": 1.27000,
            "stop_loss": 1.28000,
            "confidence": 65,
            "timeframe": "4H",
        },
    ]

    sent_count = 0
    for sig_data in signals_data:
        success = notifier.send_signal(**sig_data)
        if success:
            sent_count += 1

    print(f"✓ Sent {sent_count}/{len(signals_data)} signals to Telegram")


def example_5_conditional_notification():
    """Only notify high-confidence signals."""
    from src.signal_engine import SignalEngine

    logger.info("Example 5: Conditional notifications")

    engine = SignalEngine(
        asset="XAUUSD",
        timeframes=["15min", "1H"],
        notify_telegram=False,  # Manual control
    )

    # Simulate signal generation
    # signal = engine.generate(tf_data)

    # Only send high-confidence signals
    # if signal and signal.confidence >= 80:
    #     signal.send_to_telegram()
    #     print(f"High-confidence signal sent: {signal.asset} {signal.signal}")
    # elif signal:
    #     print(f"Signal generated but below confidence threshold: {signal.confidence}%")

    print("✓ Ready to conditionally notify based on confidence levels")


def example_6_error_handling():
    """Handle Telegram errors gracefully."""
    from src.telegram_notifier import get_telegram_notifier

    logger.info("Example 6: Error handling")

    notifier = get_telegram_notifier()

    if not notifier.is_configured():
        logger.warning("Telegram not configured - continuing without notifications")
        return

    try:
        # Try to send a signal
        success = notifier.send_signal(
            asset="TEST",
            direction="BUY",
            entry=1.0,
            take_profit=1.1,
            stop_loss=0.9,
            confidence=50,
            timeframe="1H",
        )

        if not success:
            logger.warning("Telegram notification failed but trading continues")

    except Exception as e:
        logger.error(f"Unexpected error with Telegram: {e}")
        # Don't let notification errors crash your trading system


if __name__ == "__main__":
    print("Telegram Signal Notifications - Examples\n")
    print("=" * 50)

    # Uncomment the examples you want to run:

    example_1_auto_notify()
    print()

    example_2_manual_notify()
    print()

    # Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to be set:
    # example_3_direct_notifier()
    # example_4_batch_processing()
    # example_5_conditional_notification()
    # example_6_error_handling()

    print("=" * 50)
    print("\nTo enable Telegram notifications:")
    print("1. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    print("2. See TELEGRAM_SETUP.md for detailed setup instructions")
