import logging
import httpx
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_CONFIDENCE_THRESHOLD

logger = logging.getLogger("notifier")

_last_alert_time = 0

async def send_telegram_alert(prediction: dict):
    global _last_alert_time
    import time
    now = time.time()

    # Don't send more than 1 alert per 5 minutes
    if now - _last_alert_time < 300:
        return

    confidence = prediction.get("confidence", 0)
    if confidence < TELEGRAM_CONFIDENCE_THRESHOLD:
        return

    direction = prediction.get("direction", "neutral")
    if direction == "neutral":
        return

    price = prediction.get("price", 0)
    method = prediction.get("method", "rules")

    arrow = "🟢 صعود ▲" if direction == "up" else "🔴 هبوط ▼"
    method_text = "🧠 ML" if method == "ml" else "📊 قواعد"

    text = (
        f"⚡ تنبيه BTC Trajectory\n\n"
        f"{arrow}\n"
        f"💰 السعر: ${price:,.2f}\n"
        f"📊 الثقة: {confidence*100:.0f}%\n"
        f"🔧 الطريقة: {method_text}\n\n"
        f"🔗 https://trajectory-server.onrender.com"
    )

    try:
        async with httpx.AsyncClient(verify=False, timeout=10) as client:
            await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            )
        _last_alert_time = now
        logger.info(f"Telegram alert sent: {direction} {confidence:.0%}")
    except Exception as e:
        logger.debug(f"Telegram alert failed: {e}")
