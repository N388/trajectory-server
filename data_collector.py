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

    # ─── Build candles from Supabase history ───────────────────
    async def backfill_from_supabase(self):
        """Build 1-minute candles from Supabase price history to warm up indicators"""
        logger.info("Building candles from Supabase history to warm up indicators...")
        try:
            cutoff = datetime.fromtimestamp(
                time.time() - 86400, tz=timezone.utc
            ).isoformat()

            all_rows = []
            page_size = 1000
            offset = 0
            async with httpx.AsyncClient(verify=False, timeout=30) as client:
                while True:
                    resp = await client.get(
                        f"{SB_URL}/rest/v1/btc_prices",
                        params={
                            "select": "time,price",
                            "time": f"gte.{cutoff}",
                            "order": "time.asc",
                            "limit": str(page_size),
                            "offset": str(offset),
                        },
                        headers=SB_HEADERS,
                    )
                    rows = resp.json()
                    if not isinstance(rows, list) or not rows:
                        break
                    all_rows.extend(rows)
                    if len(rows) < page_size:
                        break
                    offset += page_size

            if len(all_rows) < 10:
                logger.info(f"Not enough Supabase data for backfill ({len(all_rows)} rows)")
                return

            logger.info(f"Loaded {len(all_rows)} price points from Supabase")

            # Convert to timestamps and prices
            points = []
            for r in all_rows:
                try:
                    t = r.get("time", "")
                    if isinstance(t, str):
                        ts = int(datetime.fromisoformat(
                            t.replace("+00:00", "+00:00")
                        ).timestamp() * 1000)
                    else:
                        ts = int(t)
                    p = float(r.get("price", 0))
                    if p > 0:
                        points.append({"time": ts, "price": p})
                except Exception:
                    continue

            if len(points) < 10:
                return

            # Build 1-minute OHLCV candles from price points
            candle_interval = 60000  # 1 minute in ms
            candles_built = 0

            first_time = points[0]["time"]
            last_time = points[-1]["time"]
            bucket_start = (first_time // candle_interval) * candle_interval

            i = 0
            while bucket_start < last_time and i < len(points):
                bucket_end = bucket_start + candle_interval

                # Collect all points in this bucket
                bucket_prices = []
                while i < len(points) and points[i]["time"] < bucket_end:
                    bucket_prices.append(points[i]["price"])
                    i += 1

                if bucket_prices:
                    o = bucket_prices[0]
                    h = max(bucket_prices)
                    l = min(bucket_prices)
                    c = bucket_prices[-1]
                    v = len(bucket_prices) * 100  # Approximate volume

                    self.indicators.update_candle(o, h, l, c, v)
                    self.current_price = c
                    candles_built += 1

                bucket_start = bucket_end

            # Populate kline history for frontend
            self.kline_history = [
                {"time": p["time"], "close": p["price"]}
                for p in points[-500:]
            ]

            logger.info(
                f"Built {candles_built} candles from Supabase data. "
                f"Indicators should be warm now."
            )

            # Log indicator status
            features = self.indicators.get_feature_vector()
            non_zero = {k: v for k, v in features.items() if v != 0}
            logger.info(f"Non-zero indicators: {list(non_zero.keys())}")

        except Exception as e:
            logger.error(f"Backfill from Supabase failed: {e}")

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
        await self.backfill_from_supabase()

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
        """Evaluate predictions that are at least 10 minutes old"""
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
            logger.info(f"[EVAL] Checking predictions before {cutoff}, current price: ${current_price:,.2f}")

            url = f"{SB_URL}/rest/v1/btc_predictions"
            params = {
                "select": "id,price_at_prediction,direction,time",
                "actual_price_after": "is.null",
                "time": f"lte.{cutoff}",
                "limit": "100",
                "order": "time.asc",
            }

            async with httpx.AsyncClient(verify=False, timeout=15) as client:
                resp = await client.get(url, params=params, headers=SB_HEADERS)
                logger.info(f"[EVAL] GET status: {resp.status_code}")

                if resp.status_code != 200:
                    logger.error(f"[EVAL] Query failed: {resp.status_code} - {resp.text[:300]}")
                    return

                rows = resp.json()
                logger.info(f"[EVAL] Found {len(rows) if isinstance(rows, list) else 'non-list'} unevaluated predictions")

                if not isinstance(rows, list) or not rows:
                    # Debug: check if there are ANY predictions at all
                    debug_resp = await client.get(
                        f"{SB_URL}/rest/v1/btc_predictions",
                        params={"select": "id,time,actual_price_after", "limit": "5", "order": "time.desc"},
                        headers=SB_HEADERS,
                    )
                    logger.info(f"[EVAL] Debug - latest 5 predictions: {debug_resp.text[:500]}")
                    return

                evaluated_count = 0
                for row in rows:
                    pred_price = float(row.get("price_at_prediction", 0))
                    direction = row.get("direction", "")
                    row_id = row.get("id")
                    row_time = row.get("time", "?")

                    if pred_price <= 0:
                        logger.debug(f"[EVAL] Skipping id={row_id}: invalid price={pred_price}")
                        continue

                    if direction == "up":
                        was_correct = current_price > pred_price
                    elif direction == "down":
                        was_correct = current_price < pred_price
                    else:
                        # Neutral: correct if price didn't move much (<0.1%)
                        change_pct = abs(current_price - pred_price) / pred_price
                        was_correct = change_pct < 0.001

                    patch_resp = await client.patch(
                        f"{SB_URL}/rest/v1/btc_predictions",
                        params={"id": f"eq.{row_id}"},
                        headers={**SB_HEADERS, "Prefer": "return=minimal"},
                        json={
                            "actual_price_after": round(current_price, 2),
                            "was_correct": was_correct,
                        },
                    )

                    logger.info(f"[EVAL] PATCH id={row_id} time={row_time} dir={direction} pred=${pred_price:,.0f} actual=${current_price:,.0f} correct={was_correct} status={patch_resp.status_code}")

                    if patch_resp.status_code in (200, 204):
                        evaluated_count += 1
                    else:
                        logger.error(f"[EVAL] PATCH failed: {patch_resp.status_code} {patch_resp.text[:200]}")
                        continue

                logger.info(f"[EVAL] Done: evaluated {evaluated_count} of {len(rows)}")

        except Exception as e:
            logger.error(f"[EVAL] Exception: {type(e).__name__}: {e}")
