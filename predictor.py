"""
BTC Trajectory — Predictor
نظام التوقع: يبدأ بنظام قواعد ثم ينتقل لنموذج ML بعد تدريبه
"""
import os
import time
import math
import logging
import numpy as np

from config import (
    MODEL_PATH, PREDICTION_HORIZON_MIN, TRAJECTORY_POINTS,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger("predictor")

# ═══════════════════════════════════════════════════════════════
# Rule-Based Scorer — يعمل فوراً بدون تدريب
# ═══════════════════════════════════════════════════════════════

# أوزان المؤشرات — مبنية على الأبحاث الأكاديمية
FEATURE_WEIGHTS = {
    "obi":               0.25,   # أقوى مؤشر للمدى القصير
    "weighted_depth_imb": 0.10,
    "micro_price_diff":  0.15,   # ثاني أقوى
    "volume_delta":      0.12,
    "obi_rate":          0.08,   # تسارع ضغط الأوامر
    "rsi_14":            0.07,   # بس كمؤشر انعكاس
    "momentum_1m":       0.06,
    "macd_hist":         0.05,
    "bb_position":       0.05,
    "ema_9_diff":        0.04,
    "taker_buy_ratio":   0.03,
}


def rule_based_predict(features: dict) -> dict:
    """
    نظام قواعد مرجّح — يعطي إشارة اتجاه + ثقة
    Returns: {"direction": "up"|"down"|"neutral", "confidence": 0-1, "score": -1 to 1}
    """
    score = 0.0
    total_weight = 0.0
    signals = {}

    for key, weight in FEATURE_WEIGHTS.items():
        val = features.get(key, None)
        if val is None or val == 0:
            continue

        # تحويل كل مؤشر لإشارة بين -1 و +1
        if key == "obi":
            sig = np.clip(val * 2, -1, 1)  # OBI أصلاً بين -1 و 1
        elif key == "weighted_depth_imb":
            sig = np.clip(val * 2, -1, 1)
        elif key == "micro_price_diff":
            sig = np.clip(val * 500, -1, 1)  # فرق بسيط لكن مهم
        elif key == "volume_delta":
            sig = np.clip(val * 2, -1, 1)
        elif key == "obi_rate":
            sig = np.clip(val * 5, -1, 1)
        elif key == "rsi_14":
            # RSI: نستخدمه كمؤشر انعكاس
            if val > 70:
                sig = -((val - 70) / 30)  # إفراط شراء = احتمال هبوط
            elif val < 30:
                sig = (30 - val) / 30      # إفراط بيع = احتمال صعود
            else:
                sig = (val - 50) / 50 * 0.3  # إشارة ضعيفة في المنطقة الوسطى
        elif key == "momentum_1m":
            sig = np.clip(val / 0.3, -1, 1)
        elif key == "macd_hist":
            sig = np.clip(val / 50, -1, 1)
        elif key == "bb_position":
            # فوق 0.8 = قرب الحد العلوي = احتمال ارتداد
            sig = -(val - 0.5) * 1.5
            sig = np.clip(sig, -1, 1)
        elif key == "ema_9_diff":
            sig = np.clip(val / 0.2, -1, 1)
        elif key == "taker_buy_ratio":
            sig = (val - 0.5) * 4
            sig = np.clip(sig, -1, 1)
        else:
            sig = 0.0

        signals[key] = {"signal": round(float(sig), 3), "weight": weight}
        score += sig * weight
        total_weight += weight

    if total_weight == 0:
        return {"direction": "neutral", "confidence": 0.0, "score": 0.0, "signals": {}}

    # تطبيع
    normalized = score / total_weight
    confidence = min(abs(normalized), 1.0)

    # تحديد الاتجاه
    if confidence < 0.1:
        direction = "neutral"
    elif normalized > 0:
        direction = "up"
    else:
        direction = "down"

    return {
        "direction": direction,
        "confidence": round(float(confidence), 3),
        "score": round(float(normalized), 4),
        "signals": signals,
    }


# ═══════════════════════════════════════════════════════════════
# ML Predictor — يُفعّل بعد تدريب النموذج
# ═══════════════════════════════════════════════════════════════

class MLPredictor:
    """غلاف لنموذج XGBoost/LightGBM"""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self, path: str = MODEL_PATH) -> bool:
        """تحميل نموذج محفوظ"""
        if not os.path.exists(path):
            logger.info(f"No ML model found at {path}. Using rule-based predictor.")
            return False
        try:
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(path)
            # قراءة أسماء المؤشرات من النموذج
            self.feature_names = self.model.feature_names
            self.is_loaded = True
            logger.info(f"ML model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False

    def predict(self, features: dict) -> dict | None:
        """توقع باستخدام النموذج"""
        if not self.is_loaded or self.model is None:
            return None
        try:
            import xgboost as xgb
            # ترتيب المؤشرات حسب ما يتوقعه النموذج
            if self.feature_names:
                vals = [features.get(f, 0.0) for f in self.feature_names]
            else:
                vals = list(features.values())
            dmat = xgb.DMatrix(np.array([vals]), feature_names=self.feature_names)
            prob = float(self.model.predict(dmat)[0])

            direction = "up" if prob > 0.5 else "down"
            confidence = abs(prob - 0.5) * 2  # تحويل 0.5-1.0 → 0-1

            return {
                "direction": direction,
                "confidence": round(confidence, 3),
                "probability": round(prob, 4),
            }
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════
# Trajectory Builder — بناء منحنى التوقع
# ═══════════════════════════════════════════════════════════════

def build_trajectory(
    price: float,
    prediction: dict,
    features: dict,
    start_time: float | None = None,
) -> list:
    """
    بناء منحنى توقع واقعي بناءً على نتيجة التوقع
    بدل smoothstep + sin عشوائي، نبني منحنى يعكس التوقع الفعلي
    """
    if start_time is None:
        start_time = time.time() * 1000

    horizon_ms = PREDICTION_HORIZON_MIN * 60 * 1000
    points = TRAJECTORY_POINTS
    direction = prediction.get("direction", "neutral")
    confidence = prediction.get("confidence", 0.0)
    score = prediction.get("score", 0.0)

    # مقدار التغير المتوقع — مبني على التقلب الحالي
    atr_pct = features.get("atr_pct", 0.05)
    volatility = features.get("volatility_5m", 0.01)

    # الحد الأقصى للتغير المتوقع (واقعي)
    max_change_pct = max(atr_pct * 2, volatility * 3, 0.05)

    if direction == "neutral" or confidence < CONFIDENCE_THRESHOLD:
        # ما عندنا رأي واضح — خط مسطح مع تذبذب طفيف
        trajectory = []
        for i in range(points + 1):
            t = i / points
            noise = math.sin(t * math.pi * 4) * price * 0.00005
            trajectory.append({
                "time": start_time + t * horizon_ms,
                "price": round(price + noise, 2),
            })
        return trajectory

    # اتجاه التوقع
    sign = 1.0 if direction == "up" else -1.0
    target_change = sign * confidence * max_change_pct / 100

    trajectory = []
    for i in range(points + 1):
        t = i / points

        # منحنى واقعي: سريع في البداية ثم يتباطأ
        # (يعكس حقيقة إن التوقع أدق للمدى القريب)
        progress = 1 - (1 - t) ** 2  # quadratic ease-out

        # تذبذب طبيعي يتناقص مع الوقت
        noise_amp = volatility * 0.001 * (1 - t * 0.5)
        noise = math.sin(t * math.pi * 5 + score * 10) * noise_amp

        predicted_price = price * (1 + target_change * progress) + price * noise

        trajectory.append({
            "time": start_time + t * horizon_ms,
            "price": round(predicted_price, 2),
        })

    return trajectory


# ═══════════════════════════════════════════════════════════════
# Prediction Engine — يجمع كل شيء
# ═══════════════════════════════════════════════════════════════

class PredictionEngine:
    """محرك التوقع — يختار بين القواعد و ML تلقائياً"""

    def __init__(self):
        self.ml = MLPredictor()
        self.use_ml = False
        self.prediction_history = []  # تاريخ التوقعات للتقييم
        self.accuracy_cache = {"1h": None, "24h": None, "all": None}
        self._last_accuracy_calc = 0

    def initialize(self):
        """تحميل النموذج لو موجود"""
        if self.ml.load_model():
            self.use_ml = True
            logger.info("Using ML predictor")
        else:
            logger.info("Using rule-based predictor")

    def predict(self, price: float, features: dict) -> dict:
        """التوقع الرئيسي"""
        now = time.time() * 1000

        # محاولة ML أولاً
        ml_result = None
        if self.use_ml:
            ml_result = self.ml.predict(features)

        # استخدام ML لو نجح، وإلا القواعد
        if ml_result:
            prediction = ml_result
            prediction["method"] = "ml"
        else:
            prediction = rule_based_predict(features)
            prediction["method"] = "rules"

        # بناء المنحنى
        trajectory = build_trajectory(price, prediction, features, now)

        # تسجيل التوقع للتقييم لاحقاً
        self.prediction_history.append({
            "time": now,
            "price_at_prediction": price,
            "direction": prediction["direction"],
            "confidence": prediction["confidence"],
            "score": prediction.get("score", 0),
            "method": prediction["method"],
            "features": {k: v for k, v in features.items()},  # نسخة
            "actual_price_after": None,  # يُملأ لاحقاً
        })

        # حذف التوقعات القديمة (أكثر من 48 ساعة)
        cutoff = now - 48 * 3600 * 1000
        self.prediction_history = [
            p for p in self.prediction_history if p["time"] > cutoff
        ]

        return {
            **prediction,
            "trajectory": trajectory,
            "features": features,
            "price": price,
            "timestamp": int(now),
        }

    def evaluate_predictions(self, current_price: float):
        """تقييم التوقعات السابقة بعد مرور الوقت"""
        now = time.time() * 1000
        horizon_ms = PREDICTION_HORIZON_MIN * 60 * 1000

        for pred in self.prediction_history:
            if pred["actual_price_after"] is not None:
                continue
            if now - pred["time"] < horizon_ms:
                continue

            # مر الوقت الكافي — نقيّم
            pred["actual_price_after"] = current_price
            actual_dir = "up" if current_price > pred["price_at_prediction"] else "down"
            pred["was_correct"] = (pred["direction"] == actual_dir)

    def get_accuracy(self) -> dict:
        """حساب دقة التوقعات"""
        now = time.time() * 1000

        def calc_acc(preds):
            evaluated = [p for p in preds if p.get("was_correct") is not None]
            if len(evaluated) < 5:
                return None
            correct = sum(1 for p in evaluated if p["was_correct"])
            return round(correct / len(evaluated) * 100, 1)

        # آخر ساعة
        h1 = [p for p in self.prediction_history if now - p["time"] < 3600_000]
        # آخر 24 ساعة
        h24 = [p for p in self.prediction_history if now - p["time"] < 86400_000]

        return {
            "last_1h": calc_acc(h1),
            "last_24h": calc_acc(h24),
            "all_time": calc_acc(self.prediction_history),
            "total_predictions": len(self.prediction_history),
            "evaluated": len([p for p in self.prediction_history if p.get("was_correct") is not None]),
        }

    def get_feature_importance(self) -> list:
        """أهمية المؤشرات — من ML أو من الأوزان الافتراضية"""
        if self.use_ml and self.ml.model is not None:
            try:
                scores = self.ml.model.get_score(importance_type="gain")
                total = sum(scores.values())
                return sorted(
                    [{"name": k, "importance": round(v / total, 3)} for k, v in scores.items()],
                    key=lambda x: -x["importance"]
                )
            except Exception:
                pass

        # القواعد الافتراضية
        return sorted(
            [{"name": k, "importance": round(v, 3)} for k, v in FEATURE_WEIGHTS.items()],
            key=lambda x: -x["importance"]
        )
