# Telegram Signal Notifications Setup

Send your trading signals directly to Telegram!

## 1. Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Send `/start` and then `/newbot`
3. Follow the prompts to create your bot
4. Copy the **Bot Token** (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

## 2. Get Your Chat ID

1. Send any message to `@userinfobot` on Telegram
2. Copy your **Chat ID** (a numeric ID like `123456789`)

## 3. Set Environment Variables

Add these to your `.env` file or set them in your system:

```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Or set them in Python before using:

```python
import os
os.environ["TELEGRAM_BOT_TOKEN"] = "your_bot_token"
os.environ["TELEGRAM_CHAT_ID"] = "your_chat_id"
```

## 4. Usage Examples

### Option A: Auto-notify with SignalEngine

Enable automatic notifications when creating a `SignalEngine`:

```python
from src.signal_engine import SignalEngine

engine = SignalEngine(
    asset="EURUSD=X",
    timeframes=["15min", "1H", "4H", "1D"],
    confidence_threshold=60,
    notify_telegram=True  # Enable Telegram notifications
)

# Signals will automatically be sent to Telegram
signal = engine.generate(tf_data)
```

### Option B: Manual notification

Send signals manually to Telegram:

```python
from src.signal_engine import SignalEngine
from src.telegram_notifier import get_telegram_notifier

# Generate signal normally
engine = SignalEngine(asset="EURUSD=X")
signal = engine.generate(tf_data)

# Send to Telegram when ready
if signal:
    signal.send_to_telegram()
```

### Option C: Direct notifier usage

```python
from src.telegram_notifier import get_telegram_notifier

notifier = get_telegram_notifier()

# Send a signal
notifier.send_signal(
    asset="EURUSD=X",
    direction="BUY",
    entry=1.08500,
    take_profit=1.09000,
    stop_loss=1.08000,
    confidence=75,
    timeframe="1H",
    atr_value=0.00150,
    rr_ratio=3.33,
)

# Send errors
notifier.send_error("Failed to fetch data for EURUSD=X")

# Send status updates
notifier.send_status("Signal engine started for 10 forex pairs")
```

## Signal Message Format

Each signal includes:

```
🟢 BUY EURUSD=X
━━━━━━━━━━━━━━━━━━
Time: 2026-08-11 14:30
Timeframe: 1H
Confidence: 78%

📊 Levels:
  Entry:  1.08500
  TP:     1.09000
  SL:     1.08000

📈 Risk/Reward:
  R/R Ratio: 3.33x
  ATR (14):  0.00150

⏱️ Timeframe Alignment:
  15min    ↑ 0.65
  1H       ↑ 0.82
  4H       ↑ 0.71
  1D       ↑ 0.55
```

## Troubleshooting

**"Telegram credentials not set"**
- Make sure both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set
- Check the environment variables are accessible: `os.getenv("TELEGRAM_BOT_TOKEN")`

**"Failed to send Telegram message"**
- Check your internet connection
- Verify the bot token is correct
- Make sure the bot has been added to the chat

**Bot doesn't respond to messages**
- Ensure you've added the bot to your chat
- Start a message with the bot to activate it

## Notes

- Messages are sent with HTML formatting
- Notifications are logged (check your logger)
- Failed notifications don't block signal generation
- The `Signal.send_to_telegram()` method returns `True`/`False` for success/failure
