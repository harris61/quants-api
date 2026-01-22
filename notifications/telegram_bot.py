"""
Telegram Bot - Send predictions and alerts via Telegram
"""

import asyncio
from datetime import datetime
from typing import List, Optional
import pandas as pd
from telegram import Bot
from telegram.error import TelegramError

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TOP_PICKS_COUNT


class TelegramNotifier:
    """Send notifications via Telegram"""

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.bot = None

        if self.bot_token:
            self.bot = Bot(token=self.bot_token)

    def _check_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        if not self.bot_token or not self.chat_id:
            print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            return False
        return True

    async def _send_message_async(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message asynchronously"""
        if not self._check_configured():
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            print(f"Telegram error: {e}")
            return False

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram

        Args:
            message: Message text (supports HTML formatting)
            parse_mode: Parse mode (HTML or Markdown)

        Returns:
            True if successful
        """
        return asyncio.run(self._send_message_async(message, parse_mode))

    def send_predictions(
        self,
        predictions: pd.DataFrame,
        title: str = None
    ) -> bool:
        """
        Send prediction results via Telegram

        Args:
            predictions: DataFrame with symbol, probability, rank columns
            title: Optional title for the message

        Returns:
            True if successful
        """
        if predictions.empty:
            return self.send_message("No predictions available today.")

        # Build message
        date_str = datetime.now().strftime("%Y-%m-%d")
        title = title or f"Top Gainer Predictions for {date_str}"

        message = f"<b>{title}</b>\n"
        message += "=" * 30 + "\n\n"

        for _, row in predictions.iterrows():
            symbol = row.get('symbol', 'N/A')
            prob = row.get('probability', 0)
            rank = row.get('rank', 0)

            # Emoji based on probability
            if prob >= 0.7:
                emoji = "🔥"
            elif prob >= 0.5:
                emoji = "📈"
            else:
                emoji = "📊"

            message += f"{emoji} <b>{rank}. {symbol}</b>\n"
            message += f"   Probability: {prob:.2%}\n\n"

        message += "-" * 30 + "\n"
        message += f"<i>Generated at {datetime.now().strftime('%H:%M:%S')}</i>"

        return self.send_message(message)

    def send_daily_summary(
        self,
        predictions: pd.DataFrame,
        previous_results: pd.DataFrame = None
    ) -> bool:
        """
        Send daily summary with predictions and yesterday's results

        Args:
            predictions: Today's predictions
            previous_results: Yesterday's prediction results

        Returns:
            True if successful
        """
        date_str = datetime.now().strftime("%Y-%m-%d")

        message = f"<b>📊 Daily Report - {date_str}</b>\n"
        message += "=" * 30 + "\n\n"

        # Yesterday's results
        if previous_results is not None and not previous_results.empty:
            correct = previous_results['is_correct'].sum() if 'is_correct' in previous_results.columns else 0
            total = len(previous_results)
            precision = correct / total if total > 0 else 0

            message += "<b>Yesterday's Results:</b>\n"
            message += f"Precision: {precision:.2%} ({correct}/{total})\n\n"

            # Top performers
            if 'actual_return' in previous_results.columns:
                previous_results = previous_results.copy()
                previous_results['actual_return'] = pd.to_numeric(
                    previous_results['actual_return'], errors='coerce'
                )
                top_return = previous_results.nlargest(3, 'actual_return')
                if not top_return.empty:
                    message += "Top performers:\n"
                    for _, row in top_return.iterrows():
                        ret = row.get('actual_return', 0) * 100
                        message += f"  • {row['symbol']}: {ret:+.2f}%\n"
                    message += "\n"

        # Today's predictions
        message += "<b>Today's Top Picks:</b>\n"
        if predictions.empty:
            message += "No predictions available.\n"
        else:
            for _, row in predictions.head(TOP_PICKS_COUNT).iterrows():
                symbol = row.get('symbol', 'N/A')
                prob = row.get('probability', 0)
                rank = row.get('rank', 0)
                message += f"  {rank}. {symbol} ({prob:.2%})\n"

        message += "\n" + "-" * 30 + "\n"
        message += "<i>Quants-API Automated Report</i>"

        return self.send_message(message)

    def send_alert(
        self,
        alert_type: str,
        message_text: str,
        symbol: str = None
    ) -> bool:
        """
        Send an alert message

        Args:
            alert_type: Type of alert (info, warning, error)
            message_text: Alert message
            symbol: Related stock symbol (optional)

        Returns:
            True if successful
        """
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
        }

        emoji = emoji_map.get(alert_type.lower(), "📢")

        message = f"{emoji} <b>{alert_type.upper()}</b>\n\n"
        if symbol:
            message += f"Symbol: <b>{symbol}</b>\n"
        message += f"{message_text}\n\n"
        message += f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"

        return self.send_message(message)

    def send_weekly_report(
        self,
        weekly_stats: dict
    ) -> bool:
        """
        Send weekly performance report

        Args:
            weekly_stats: Dictionary with weekly statistics

        Returns:
            True if successful
        """
        message = "<b>📈 Weekly Performance Report</b>\n"
        message += "=" * 30 + "\n\n"

        # Overall stats
        message += f"Total Predictions: {weekly_stats.get('total_predictions', 0)}\n"
        message += f"Correct Predictions: {weekly_stats.get('correct_predictions', 0)}\n"
        message += f"Precision: {weekly_stats.get('precision', 0):.2%}\n"
        message += f"Hit Rate: {weekly_stats.get('hit_rate', 0):.2%}\n\n"

        # Best predictions
        if 'best_picks' in weekly_stats:
            message += "<b>Best Picks:</b>\n"
            for pick in weekly_stats['best_picks'][:5]:
                message += f"  • {pick['symbol']}: {pick['return']:+.2f}%\n"
            message += "\n"

        # Model performance trend
        if 'precision_trend' in weekly_stats:
            trend = weekly_stats['precision_trend']
            if trend > 0:
                message += f"📈 Precision improving: +{trend:.2%}\n"
            else:
                message += f"📉 Precision declining: {trend:.2%}\n"

        message += "\n" + "-" * 30 + "\n"
        message += "<i>Quants-API Weekly Report</i>"

        return self.send_message(message)


def test_telegram():
    """Test Telegram connection"""
    notifier = TelegramNotifier()

    if not notifier._check_configured():
        print("\nTo configure Telegram:")
        print("1. Create a bot via @BotFather on Telegram")
        print("2. Get your chat ID (send a message to your bot and check via API)")
        print("3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        return

    # Send test message
    success = notifier.send_message(
        "<b>Test Message</b>\n\n"
        "Quants-API Telegram integration is working!\n\n"
        f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
    )

    if success:
        print("Test message sent successfully!")
    else:
        print("Failed to send test message.")


if __name__ == "__main__":
    test_telegram()
