"""
BTC Trajectory — Data Collector
يتصل بـ Binance WebSocket ويجمع البيانات ويحسب المؤشرات
"""
import json
import time
import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

import httpx
import websockets

from config import (
    WS_COMBINED, REST_KLINES, SB_URL, SB_HEADERS,
    KLINE_HISTORY_COUNT, MAX_TRADE_BUFFER, SAVE_INTERVAL_SEC,
)
from indicators import IndicatorManager

logger = logging.getLogger("collector")


class DataCollector:
    """
    يجمع بيانات BTC/USDT من Binance ويحسب المؤشرات في الوقت الحقيقي
    """

    def __init__(self):
        self.indicators = IndicatorManager()
        self.current_price: float | None = None
        self.change_24h: float = 0.0
        self.trades_buffer: list = []
        self.latest_depth: dict = {"bids": [], "asks": []}
        self.kline_history: list = []  # آخر الشموع المغلقة
        self.prediction_log: list = []  # سجل التوقعات للتقييم
        self._running = False
        self._last_save = 0
        self._last_kline_time = 0

    # ─── Load Historical Klines on Start ─────────────────────
    async def load_history(self):
        """Indicators will warm up from live WebSocket kline stream"""
        logger.info("Building indicators from live WebSocket data (REST API unavailable).")
        logger.info("Technical indicators will activate after ~15-50 candles (minutes).")

    # ─── Load price history from Supabase ────────────────────
    async def load_supabase_history(self):
        """تحميل آخر 24 ساعة من Supabase"""
        try:
            since = datetime.now(timezone.utc).isoformat()
            # تحميل آخر 24 ساعة
            cutoff = datetime.fromtimestamp(
                time.time() - 86400, tz=timezone.utc
            ).isoformat()
            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                resp = await client.get(
                    f"{SB_URL}/rest/v1/btc_prices",
                    params={
                        "select": "time,price",
                        "time": f"gte.{cutoff}",
                        "order": "time.asc",
                    },
                    headers=SB_HEADERS,
                )
                rows = resp.json()

            if isinstance(rows, list) and rows:
                logger.info(f"Loaded {len(rows)} price points from Supabase")
                return rows
        except Exception as e:
            logger.error(f"Supabase history load failed: {e}")
        return []

    # ─── Save price to Supabase ──────────────────────────────
    async def _save_price(self):
        """حفظ السعر الحالي + المؤشرات في Supabase"""
        if not self.current_price:
            return
        try:
            features = self.indicators.get_feature_vector()
            payload = {
                "price": self.current_price,
                "obi": features.get("obi", 0),
                "rsi": features.get("rsi_14", 0),
                "volume_delta": features.get("volume_delta", 0),
            }
            async with httpx.AsyncClient(verify=False, timeout=10) as client:
                await client.post(
                    f"{SB_URL}/rest/v1/btc_prices",
                    headers={**SB_HEADERS, "Prefer": "return=minimal"},
                    json=payload,
                )
        except Exception as e:
            logger.debug(f"Save failed: {e}")

    # ─── WebSocket Message Handler ───────────────────────────
    def _handle_message(self, raw: str):
        """معالجة رسائل Binance Combined Stream"""
        try:
            msg = json.loads(raw)
            stream = msg.get("stream", "")
            data = msg.get("data", {})

            if "ticker" in stream:
                self._handle_ticker(data)
            elif "depth" in stream:
                self._handle_depth(data)
            elif "kline" in stream:
                self._handle_kline(data)
            elif "aggTrade" in stream:
                self._handle_trade(data)
        except Exception as e:
            logger.debug(f"Message parse error: {e}")

    def _handle_ticker(self, data: dict):
        """تحديث السعر من ticker"""
        price = float(data.get("c", 0))
        if price <= 0:
            return
        self.current_price = price
        self.change_24h = float(data.get("P", 0))
        self.indicators.update_price_tick(price, time.time() * 1000)

    def _handle_depth(self, data: dict):
        """تحديث الأوردر بوك"""
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return
        self.latest_depth = {"bids": bids, "asks": asks}
        self.indicators.update_orderbook(bids, asks)

    def _handle_kline(self, data: dict):
        """تحديث الشموع — فقط عند إغلاق شمعة"""
        k = data.get("k", {})
        is_closed = k.get("x", False)

        if is_closed:
            kline_time = int(k["t"])
            if kline_time <= self._last_kline_time:
                return
            self._last_kline_time = kline_time

            o = float(k["o"])
            h = float(k["h"])
            l = float(k["l"])
            c = float(k["c"])
            v = float(k["v"])

            self.kline_history.append({
                "time": kline_time, "open": o, "high": h,
                "low": l, "close": c, "volume": v,
                "taker_buy_volume": float(k.get("V", 0)),
                "trades": int(k.get("n", 0)),
            })
            # إبقاء آخر 500 شمعة فقط
            if len(self.kline_history) > 600:
                self.kline_history = self.kline_history[-500:]

            self.indicators.update_candle(o, h, l, c, v)
            logger.debug(f"Kline closed: {c}")

    def _handle_trade(self, data: dict):
        """تجميع الصفقات الفردية"""
        trade = {
            "price": float(data.get("p", 0)),
            "qty": float(data.get("q", 0)),
            "time": int(data.get("T", 0)),
            "is_buyer_maker": data.get("m", False),
        }
        self.trades_buffer.append(trade)

        # حذف الصفقات القديمة (أكثر من 5 دقائق)
        cutoff = time.time() * 1000 - 300_000
        if len(self.trades_buffer) > MAX_TRADE_BUFFER:
            self.trades_buffer = [
                t for t in self.trades_buffer if t["time"] > cutoff
            ]

        # تحديث مؤشرات الصفقات كل 50 صفقة
        if len(self.trades_buffer) % 50 == 0:
            self.indicators.update_trades(self.trades_buffer)

    # ─── Main WebSocket Loop ─────────────────────────────────
    async def run(self):
        """حلقة الاتصال الرئيسية — تعيد الاتصال تلقائياً"""
        self._running = True
        await self.load_history()

        while self._running:
            try:
                logger.info("Connecting to Binance Combined Stream...")
                async with websockets.connect(
                    WS_COMBINED,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    logger.info("Connected to Binance!")

                    while self._running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            self._handle_message(raw)

                            # حفظ دوري في Supabase
                            now = time.time()
                            if now - self._last_save >= SAVE_INTERVAL_SEC:
                                self._last_save = now
                                asyncio.create_task(self._save_price())

                        except asyncio.TimeoutError:
                            # Ping/pong timeout — reconnect
                            logger.warning("WS timeout, reconnecting...")
                            break

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WS connection closed: {e}. Reconnecting in 3s...")
            except Exception as e:
                logger.error(f"WS error: {e}. Reconnecting in 5s...")

            if self._running:
                await asyncio.sleep(3)

    def stop(self):
        self._running = False

    # ─── Public Getters ──────────────────────────────────────
    def get_state(self) -> dict:
        """الحالة الكاملة — للـ API"""
        features = self.indicators.get_feature_vector()
        return {
            "price": self.current_price,
            "change_24h": self.change_24h,
            "features": features,
            "depth_snapshot": {
                "bid_total": sum(float(q) for _, q in self.latest_depth["bids"][:10]),
                "ask_total": sum(float(q) for _, q in self.latest_depth["asks"][:10]),
            },
            "klines_count": len(self.kline_history),
            "trades_buffered": len(self.trades_buffer),
            "timestamp": int(time.time() * 1000),
        }

    async def save_prediction(self, prediction: dict):
        """Save prediction to Supabase for accuracy tracking"""
        try:
            payload = {
                "price_at_prediction": prediction.get("price"),
                "direction": prediction.get("direction"),
                "confidence": prediction.get("confidence"),
                "score": prediction.get("score", 0),
                "method": prediction.get("method"),
            }
            async with httpx.AsyncClient(verify=False, timeout=10) as client:
                await client.post(
                    f"{SB_URL}/rest/v1/btc_predictions",
                    headers={**SB_HEADERS, "Prefer": "return=minimal"},
                    json=payload,
                )
        except Exception as e:
            logger.debug(f"Save prediction failed: {e}")

    async def evaluate_old_predictions(self, current_price: float):
        """Evaluate predictions from 10 minutes ago and update their results"""
        try:
            from datetime import timedelta
            ten_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
            ten_min_ago_end = (datetime.now(timezone.utc) - timedelta(minutes=9)).isoformat()

            async with httpx.AsyncClient(verify=False, timeout=10) as client:
                # Get predictions from ~10 minutes ago that haven't been evaluated
                resp = await client.get(
                    f"{SB_URL}/rest/v1/btc_predictions",
                    params={
                        "select": "id,price_at_prediction,direction",
                        "time": f"gte.{ten_min_ago}",
                        "time": f"lte.{ten_min_ago_end}",
                        "actual_price_after": "is.null",
                        "limit": "50",
                    },
                    headers=SB_HEADERS,
                )
                rows = resp.json()

                if not isinstance(rows, list) or not rows:
                    return

                for row in rows:
                    was_correct = False
                    if row["direction"] == "up":
                        was_correct = current_price > row["price_at_prediction"]
                    elif row["direction"] == "down":
                        was_correct = current_price < row["price_at_prediction"]

                    await client.patch(
                        f"{SB_URL}/rest/v1/btc_predictions?id=eq.{row['id']}",
                        headers={**SB_HEADERS, "Prefer": "return=minimal"},
                        json={
                            "actual_price_after": current_price,
                            "was_correct": was_correct,
                        },
                    )
        except Exception as e:
            logger.debug(f"Evaluate predictions failed: {e}")
