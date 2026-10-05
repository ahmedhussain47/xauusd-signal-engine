"""
telegram_notifier.py — Send trading signals to Telegram via Bot API.
"""

import logging
import os
from typing import Optional
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends trading signals to a Telegram chat using Bot API.

    Setup:
        1. Create a bot via @BotFather on Telegram
        2. Get your chat ID (use @userinfobot)
        3. Set environment variables:
           - TELEGRAM_BOT_TOKEN: Your bot token
           - TELEGRAM_CHAT_ID: Your chat ID
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = 10,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token (falls back to env var TELEGRAM_BOT_TOKEN)
            chat_id: Telegram chat ID (falls back to env var TELEGRAM_CHAT_ID)
            timeout: Request timeout in seconds
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout = timeout
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        if not self.bot_token or not self.chat_id:
            logger.warning(
                "Telegram credentials not set. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            )

    def is_configured(self) -> bool:
        """Check if Telegram credentials are set."""
        return bool(self.bot_token and self.chat_id)

    def send_signal(
        self,
        asset: str,
        direction: str,
        entry: float,
        take_profit: float,
        stop_loss: float,
        confidence: int,
        timeframe: str,
        atr_value: float = 0.0,
        rr_ratio: float = 0.0,
        model_pred: Optional[float] = None,
        tf_alignment: Optional[dict] = None,
    ) -> bool:
        """
        Send a trading signal to Telegram.

        Args:
            asset: Trading asset symbol
            direction: "BUY" or "SELL"
            entry: Entry price
            take_profit: Take profit price
            stop_loss: Stop loss price
            confidence: Confidence level 0-100
            timeframe: Entry timeframe
            atr_value: ATR value
            rr_ratio: Risk-reward ratio
            model_pred: Optional model prediction
            tf_alignment: Dict of timeframe alignments

        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.is_configured():
            logger.error("Telegram notifier not configured")
            return False

        try:
            emoji = "🟢" if direction == "BUY" else "🔴"
            message = self._format_signal_message(
                asset=asset,
                direction=direction,
                entry=entry,
                take_profit=take_profit,
                stop_loss=stop_loss,
                confidence=confidence,
                timeframe=timeframe,
                atr_value=atr_value,
                rr_ratio=rr_ratio,
                model_pred=model_pred,
                tf_alignment=tf_alignment,
                emoji=emoji,
            )

            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }

            response = requests.post(
                self.api_url, json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            logger.info(f"Signal sent to Telegram: {asset} {direction}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to Telegram: {e}")
            return False

    def send_error(self, error_message: str) -> bool:
        """Send an error notification to Telegram."""
        if not self.is_configured():
            return False

        try:
            message = f"⚠️ <b>Trading System Error</b>\n\n{error_message}"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }

            response = requests.post(
                self.api_url, json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            logger.info("Error notification sent to Telegram")
            return True

        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
            return False

    def send_status(self, status_message: str) -> bool:
        """Send a status update to Telegram."""
        if not self.is_configured():
            return False

        try:
            message = f"ℹ️ <b>Status Update</b>\n\n{status_message}"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }

            response = requests.post(
                self.api_url, json=payload, timeout=self.timeout
            )
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Failed to send status notification: {e}")
            return False

    @staticmethod
    def _format_signal_message(
        asset: str,
        direction: str,
        entry: float,
        take_profit: float,
        stop_loss: float,
        confidence: int,
        timeframe: str,
        atr_value: float,
        rr_ratio: float,
        model_pred: Optional[float],
        tf_alignment: Optional[dict],
        emoji: str,
    ) -> str:
        """Format a trading signal as HTML message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        message = f"{emoji} <b>{direction} {asset}</b>\n"
        message += f"<b>━━━━━━━━━━━━━━━━━━</b>\n"
        message += f"<b>Time:</b> {now}\n"
        message += f"<b>Timeframe:</b> {timeframe}\n"
        message += f"<b>Confidence:</b> {confidence}%\n\n"

        message += f"<b>📊 Levels:</b>\n"
        message += f"  Entry:  <code>{entry:.5f}</code>\n"
        message += f"  TP:     <code>{take_profit:.5f}</code>\n"
        message += f"  SL:     <code>{stop_loss:.5f}</code>\n\n"

        message += f"<b>📈 Risk/Reward:</b>\n"
        message += f"  R/R Ratio: {rr_ratio:.2f}x\n"
        message += f"  ATR (14):  {atr_value:.5f}\n"

        if model_pred is not None:
            message += f"\n<b>🤖 Model Signal:</b>\n"
            message += f"  Prediction: {model_pred:.6f}\n"

        if tf_alignment:
            message += f"\n<b>⏱️ Timeframe Alignment:</b>\n"
            for tf, info in tf_alignment.items():
                direction_symbol = "↑" if info.get("direction") == 1 else "↓" if info.get("direction") == -1 else "→"
                strength = info.get("strength", 0)
                message += f"  {tf:8s} {direction_symbol} {strength:.2f}\n"

        return message


# Singleton instance
_notifier: Optional[TelegramNotifier] = None


def get_telegram_notifier(
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> TelegramNotifier:
    """Get or create the Telegram notifier singleton."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
    return _notifier
