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

    # ─── Backfill gaps from Binance klines ───────────────────
    async def backfill_gaps(self):
        """Fill data gaps from Binance klines when server restarts"""
        logger.info("Checking for data gaps to backfill...")
        try:
            # 1. Find the last saved price time in Supabase
            async with httpx.AsyncClient(verify=False, timeout=15) as client:
                resp = await client.get(
                    f"{SB_URL}/rest/v1/btc_prices?select=time&order=time.desc&limit=1",
                    headers=SB_HEADERS,
                )
                rows = resp.json()

            if not isinstance(rows, list) or not rows:
                logger.info("No previous data in Supabase, skipping backfill")
                return

            last_time_str = rows[0]["time"]
            last_time = datetime.fromisoformat(last_time_str.replace("+00:00", "+00:00"))
            last_ts = int(last_time.timestamp() * 1000)
            now_ts = int(time.time() * 1000)
            gap_minutes = (now_ts - last_ts) / 60000

            if gap_minutes < 2:
                logger.info(f"No significant gap ({gap_minutes:.1f} min), skipping backfill")
                return

            logger.info(f"Found gap of {gap_minutes:.1f} minutes. Backfilling from Binance klines...")

            # 2. Fetch 1-minute klines from Binance to cover the gap
            kline_url = REST_KLINES
            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                resp = await client.get(kline_url, params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "startTime": last_ts,
                    "endTime": now_ts,
                    "limit": 1000,
                })
                klines = resp.json()

            if not isinstance(klines, list) or not klines:
                logger.warning("No klines returned from Binance for backfill")
                return

            # 3. Save each kline as a price point in Supabase
            saved = 0
            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                batch = []
                for k in klines:
                    close_price = float(k[4])
                    close_time = int(k[6])  # kline close time
                    ts = datetime.fromtimestamp(close_time / 1000, tz=timezone.utc).isoformat()
                    batch.append({"time": ts, "price": close_price})

                    if len(batch) >= 50:
                        await client.post(
                            f"{SB_URL}/rest/v1/btc_prices",
                            headers={**SB_HEADERS, "Prefer": "return=minimal"},
                            json=batch,
                        )
                        saved += len(batch)
                        batch = []

                if batch:
                    await client.post(
                        f"{SB_URL}/rest/v1/btc_prices",
                        headers={**SB_HEADERS, "Prefer": "return=minimal"},
                        json=batch,
                    )
                    saved += len(batch)

            logger.info(f"Backfilled {saved} price points from Binance klines")

            # 4. Also update indicators with the backfilled klines
            for k in klines:
                o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
                v = float(k[5])
                self.indicators.update_candle(o, h, l, c, v)
                self.current_price = c

            logger.info(f"Indicators warmed up with {len(klines)} backfilled candles")

        except Exception as e:
            logger.error(f"Backfill failed: {e}")

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
        price = float(data.get("c", 0))
        if price <= 0:
            return
        # Outlier filter: reject price if it jumps more than 1% from last known price in one tick
        if self.current_price is not None:
            change_pct = abs(price - self.current_price) / self.current_price
            if change_pct > 0.01:
                logger.debug(f"Rejected outlier price: {price} (current: {self.current_price}, jump: {change_pct:.2%})")
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
        await self.backfill_gaps()

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
        try:
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
            async with httpx.AsyncClient(verify=False, timeout=10) as client:
                resp = await client.get(
                    f"{SB_URL}/rest/v1/btc_predictions?actual_price_after=is.null&time=lte.{cutoff}&limit=50",
                    headers=SB_HEADERS,
                )
                rows = resp.json()
                if not isinstance(rows, list) or not rows:
                    return
                for row in rows:
                    was_correct = (row["direction"] == "up" and current_price > row["price_at_prediction"]) or (row["direction"] == "down" and current_price < row["price_at_prediction"])
                    await client.patch(
                        f"{SB_URL}/rest/v1/btc_predictions?id=eq.{row['id']}",
                        headers={**SB_HEADERS, "Prefer": "return=minimal"},
                        json={"actual_price_after": current_price, "was_correct": was_correct},
                    )
        except Exception as e:
            logger.debug(f"Evaluate predictions failed: {e}")
