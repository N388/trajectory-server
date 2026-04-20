"""
BTC Trajectory — Technical Indicators
حساب المؤشرات الفنية من بيانات الشموع والأوردر بوك
"""
import numpy as np
from collections import deque


class EMA:
    """Exponential Moving Average — حساب تدريجي بدون إعادة حساب كل مرة"""
    def __init__(self, period: int):
        self.period = period
        self.k = 2.0 / (period + 1)
        self.value = None
        self.count = 0

    def update(self, price: float) -> float | None:
        if self.value is None:
            self.value = price
        else:
            self.value = price * self.k + self.value * (1 - self.k)
        self.count += 1
        return self.value if self.count >= self.period else None

    def reset(self):
        self.value = None
        self.count = 0


class RSI:
    """Relative Strength Index — Wilder's smoothing"""
    def __init__(self, period: int = 14):
        self.period = period
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.prev_price = None
        self.avg_gain = None
        self.avg_loss = None
        self.count = 0

    def update(self, price: float) -> float | None:
        if self.prev_price is not None:
            change = price - self.prev_price
            gain = max(0, change)
            loss = max(0, -change)

            if self.count < self.period:
                self.gains.append(gain)
                self.losses.append(loss)
                self.count += 1
                if self.count == self.period:
                    self.avg_gain = sum(self.gains) / self.period
                    self.avg_loss = sum(self.losses) / self.period
            else:
                self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
                self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period

        self.prev_price = price

        if self.avg_gain is None:
            return None
        if self.avg_loss == 0:
            return 100.0
        rs = self.avg_gain / self.avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


class MACD:
    """Moving Average Convergence Divergence"""
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.ema_fast = EMA(fast)
        self.ema_slow = EMA(slow)
        self.ema_signal = EMA(signal)
        self.slow_period = slow

    def update(self, price: float) -> dict | None:
        fast_val = self.ema_fast.update(price)
        slow_val = self.ema_slow.update(price)

        if fast_val is None or slow_val is None:
            return None
        if self.ema_fast.count < self.slow_period:
            return None

        macd_line = fast_val - slow_val
        signal_val = self.ema_signal.update(macd_line)

        if signal_val is None:
            return {"macd": macd_line, "signal": macd_line, "histogram": 0}

        return {
            "macd": macd_line,
            "signal": signal_val,
            "histogram": macd_line - signal_val,
        }


class BollingerBands:
    """Bollinger Bands — SMA ± k*σ"""
    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std
        self.prices = deque(maxlen=period)

    def update(self, price: float) -> dict | None:
        self.prices.append(price)
        if len(self.prices) < self.period:
            return None

        arr = np.array(self.prices)
        sma = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0

        upper = sma + self.num_std * std
        lower = sma - self.num_std * std
        width = upper - lower

        # موقع السعر داخل البولنجر (0 = عند الحد السفلي، 1 = عند العلوي)
        position = (price - lower) / width if width > 0 else 0.5

        return {
            "upper": upper,
            "middle": sma,
            "lower": lower,
            "width": width,
            "position": np.clip(position, 0, 1),
        }


class ATR:
    """Average True Range — مقياس التقلب"""
    def __init__(self, period: int = 14):
        self.period = period
        self.values = deque(maxlen=period)
        self.prev_close = None
        self.atr_value = None

    def update(self, high: float, low: float, close: float) -> float | None:
        if self.prev_close is not None:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close),
            )
            self.values.append(tr)

            if len(self.values) >= self.period:
                if self.atr_value is None:
                    self.atr_value = sum(self.values) / self.period
                else:
                    self.atr_value = (self.atr_value * (self.period - 1) + tr) / self.period

        self.prev_close = close
        return self.atr_value


class OBV:
    """On-Balance Volume"""
    def __init__(self):
        self.value = 0.0
        self.prev_price = None

    def update(self, price: float, volume: float) -> float:
        if self.prev_price is not None:
            if price > self.prev_price:
                self.value += volume
            elif price < self.prev_price:
                self.value -= volume
        self.prev_price = price
        return self.value


# ═══════════════════════════════════════════════════════════════
# Order Book Microstructure Indicators
# ═══════════════════════════════════════════════════════════════

def calc_order_book_imbalance(bids: list, asks: list, levels: int = 20) -> float:
    """
    Order Book Imbalance (OBI)
    أقوى مؤشر مثبت أكاديمياً للمدى القصير
    Returns: -1 (ضغط بيع كامل) to +1 (ضغط شراء كامل)
    """
    bid_vol = sum(float(q) for _, q in bids[:levels])
    ask_vol = sum(float(q) for _, q in asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total


def calc_micro_price(bids: list, asks: list) -> float | None:
    """
    Micro-Price (Stoikov)
    تقدير أدق للسعر "الحقيقي" بناءً على حجم الأوامر
    """
    if not bids or not asks:
        return None
    best_bid = float(bids[0][0])
    bid_qty = float(bids[0][1])
    best_ask = float(asks[0][0])
    ask_qty = float(asks[0][1])
    total = bid_qty + ask_qty
    if total == 0:
        return (best_bid + best_ask) / 2
    imb = bid_qty / total
    return best_bid * imb + best_ask * (1 - imb)


def calc_vamp(bids: list, asks: list) -> float | None:
    """
    Volume-Adjusted Mid Price (VAMP)
    """
    if not bids or not asks:
        return None
    best_bid = float(bids[0][0])
    bid_qty = float(bids[0][1])
    best_ask = float(asks[0][0])
    ask_qty = float(asks[0][1])
    total = bid_qty + ask_qty
    if total == 0:
        return (best_bid + best_ask) / 2
    return (best_bid * ask_qty + best_ask * bid_qty) / total


def calc_weighted_depth_imbalance(bids: list, asks: list, levels: int = 10) -> float:
    """
    Weighted Depth Imbalance
    يعطي وزن أكبر للمستويات القريبة من السعر
    """
    if not bids or not asks:
        return 0.0
    mid = (float(bids[0][0]) + float(asks[0][0])) / 2
    bid_pressure = 0.0
    ask_pressure = 0.0
    for i in range(min(levels, len(bids))):
        dist = abs(float(bids[i][0]) - mid) / mid
        weight = 1.0 / (1.0 + dist * 1000)  # وزن يتناقص مع البعد
        bid_pressure += float(bids[i][1]) * weight
    for i in range(min(levels, len(asks))):
        dist = abs(float(asks[i][0]) - mid) / mid
        weight = 1.0 / (1.0 + dist * 1000)
        ask_pressure += float(asks[i][1]) * weight
    total = bid_pressure + ask_pressure
    if total == 0:
        return 0.0
    return (bid_pressure - ask_pressure) / total


def calc_spread(bids: list, asks: list) -> float:
    """Bid-Ask Spread as percentage"""
    if not bids or not asks:
        return 0.0
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2
    if mid == 0:
        return 0.0
    return (best_ask - best_bid) / mid * 100


# ═══════════════════════════════════════════════════════════════
# Trade Flow Indicators
# ═══════════════════════════════════════════════════════════════

def calc_volume_delta(trades: list, window_sec: int = 60) -> dict:
    """
    Volume Delta — الفرق بين حجم الشراء والبيع
    trades: list of {"price", "qty", "time", "is_buyer_maker"}
    """
    import time
    now = time.time() * 1000
    cutoff = now - window_sec * 1000

    buy_vol = 0.0
    sell_vol = 0.0
    buy_count = 0
    sell_count = 0

    for t in trades:
        if t["time"] < cutoff:
            continue
        qty = t["qty"]
        if t["is_buyer_maker"]:
            sell_vol += qty
            sell_count += 1
        else:
            buy_vol += qty
            buy_count += 1

    total = buy_vol + sell_vol
    delta = (buy_vol - sell_vol) / total if total > 0 else 0.0
    intensity = buy_count + sell_count  # صفقات في النافذة الزمنية

    return {
        "volume_delta": delta,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "trade_intensity": intensity,
        "taker_buy_ratio": buy_vol / total if total > 0 else 0.5,
    }


# ═══════════════════════════════════════════════════════════════
# Indicator Manager — يجمع كل المؤشرات في كائن واحد
# ═══════════════════════════════════════════════════════════════

class IndicatorManager:
    """يحسب كل المؤشرات بشكل تدريجي مع كل شمعة / تحديث"""

    def __init__(self):
        # مؤشرات فنية كلاسيكية
        self.rsi = RSI(14)
        self.ema_9 = EMA(9)
        self.ema_21 = EMA(21)
        self.ema_50 = EMA(50)
        self.macd = MACD(12, 26, 9)
        self.bb = BollingerBands(20, 2.0)
        self.atr = ATR(14)
        self.obv = OBV()

        # تتبع قيم OBI لحساب معدل التغير
        self.obi_history = deque(maxlen=30)
        self.price_history = deque(maxlen=300)  # آخر 5 دقائق @ 1/ثانية

        # آخر قيم محسوبة
        self.latest = {}

    def update_candle(self, o: float, h: float, l: float, c: float, v: float):
        """تحديث عند إغلاق شمعة جديدة (كل دقيقة)"""
        rsi_val = self.rsi.update(c)
        ema9_val = self.ema_9.update(c)
        ema21_val = self.ema_21.update(c)
        ema50_val = self.ema_50.update(c)
        macd_val = self.macd.update(c)
        bb_val = self.bb.update(c)
        atr_val = self.atr.update(h, l, c)
        obv_val = self.obv.update(c, v)

        result = {"price": c}

        if rsi_val is not None:
            result["rsi_14"] = round(rsi_val, 2)
        if ema9_val is not None:
            result["ema_9_diff"] = round((c - ema9_val) / c * 100, 4)
        if ema21_val is not None:
            result["ema_21_diff"] = round((c - ema21_val) / c * 100, 4)
        if ema50_val is not None:
            result["ema_50_diff"] = round((c - ema50_val) / c * 100, 4)
        if macd_val is not None:
            result["macd_hist"] = round(macd_val["histogram"], 4)
            result["macd_line"] = round(macd_val["macd"], 4)
        if bb_val is not None:
            result["bb_position"] = round(bb_val["position"], 4)
            result["bb_width"] = round(bb_val["width"], 2)
        if atr_val is not None:
            result["atr_14"] = round(atr_val, 2)
            result["atr_pct"] = round(atr_val / c * 100, 4)
        result["obv"] = round(obv_val, 2)

        self.latest.update(result)
        return result

    def update_orderbook(self, bids: list, asks: list):
        """تحديث عند وصول بيانات أوردر بوك جديدة"""
        obi = calc_order_book_imbalance(bids, asks)
        micro = calc_micro_price(bids, asks)
        vamp = calc_vamp(bids, asks)
        w_depth = calc_weighted_depth_imbalance(bids, asks)
        spread = calc_spread(bids, asks)

        self.obi_history.append(obi)

        result = {
            "obi": round(obi, 4),
            "weighted_depth_imb": round(w_depth, 4),
            "spread_pct": round(spread, 6),
        }

        if micro is not None:
            current = self.latest.get("price", micro)
            result["micro_price_diff"] = round((micro - current) / current * 100, 6)

        if vamp is not None:
            current = self.latest.get("price", vamp)
            result["vamp_diff"] = round((vamp - current) / current * 100, 6)

        # معدل تغير OBI
        if len(self.obi_history) >= 5:
            obi_list = list(self.obi_history)
            result["obi_rate"] = round(obi_list[-1] - obi_list[-5], 4)

        self.latest.update(result)
        return result

    def update_trades(self, trades: list):
        """تحديث من بيانات الصفقات"""
        vd = calc_volume_delta(trades, window_sec=60)
        result = {
            "volume_delta": round(vd["volume_delta"], 4),
            "trade_intensity": vd["trade_intensity"],
            "taker_buy_ratio": round(vd["taker_buy_ratio"], 4),
        }
        self.latest.update(result)
        return result

    def update_price_tick(self, price: float, timestamp: float):
        """تحديث عند كل تغير سعر (tick)"""
        self.price_history.append({"price": price, "time": timestamp})
        self.latest["price"] = price

        # حساب الزخم
        hist = list(self.price_history)
        if len(hist) >= 60:
            p_1m = hist[-60]["price"]
            self.latest["momentum_1m"] = round((price - p_1m) / p_1m * 100, 4)
        # 5-minute momentum: find price from ~5 min ago by time, not by index
        five_min_ago = timestamp - 300000  # 5 minutes in ms
        for j in range(len(hist) - 1, -1, -1):
            if hist[j]["time"] <= five_min_ago:
                p_5m = hist[j]["price"]
                self.latest["momentum_5m"] = round((price - p_5m) / p_5m * 100, 4)
                break

        # تقلب 5 دقائق
        if len(hist) >= 60:
            recent = [h["price"] for h in hist[-300:]]
            arr = np.array(recent)
            returns = np.diff(arr) / arr[:-1]
            self.latest["volatility_5m"] = round(float(np.std(returns) * 100), 4)

    def get_feature_vector(self) -> dict:
        """إرجاع كل المؤشرات كـ dict — جاهز لإدخاله للنموذج"""
        feature_keys = [
            "obi", "obi_rate", "weighted_depth_imb",
            "micro_price_diff", "vamp_diff", "spread_pct",
            "volume_delta", "trade_intensity", "taker_buy_ratio",
            "rsi_14", "ema_9_diff", "ema_21_diff", "ema_50_diff",
            "macd_hist", "macd_line",
            "bb_position", "bb_width",
            "atr_14", "atr_pct",
            "momentum_1m", "momentum_5m",
            "volatility_5m",
        ]
        return {k: self.latest.get(k, 0.0) for k in feature_keys}
