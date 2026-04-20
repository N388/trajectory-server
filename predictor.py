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
        # Require stronger confirmation for "up" signals
        confirmations = 0
        if features.get("obi", 0) > 0.1:          confirmations += 1
        if features.get("volume_delta", 0) > 0.1:  confirmations += 1
        if features.get("rsi_14", 50) > 45:         confirmations += 1
        if features.get("momentum_1m", 0) > 0:      confirmations += 1
        if features.get("macd_hist", 0) > 0:        confirmations += 1
        if features.get("ema_9_diff", 0) > 0:       confirmations += 1

        if confirmations >= 3:
            direction = "up"
        else:
            if confirmations <= 1:
                direction = "down"
                confidence = confidence * 0.3
            else:
                direction = "neutral"
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

def build_trajectory(price, prediction, features, start_time=None):
    if start_time is None:
        start_time = time.time() * 1000

    horizon_ms = PREDICTION_HORIZON_MIN * 60 * 1000
    points = TRAJECTORY_POINTS
    direction = prediction.get("direction", "neutral")
    confidence = prediction.get("confidence", 0.0)
    score = prediction.get("score", 0.0)

    # If truly neutral or extremely low confidence, flat line
    if direction == "neutral" or confidence < CONFIDENCE_THRESHOLD:
        return [{"time": start_time + (i/points) * horizon_ms, "price": round(price, 2)} for i in range(points + 1)]

    # Calculate expected price change based on volatility and confidence
    atr_pct = features.get("atr_pct", 0.05)
    volatility = features.get("volatility_5m", 0.01)
    max_change_pct = max(atr_pct * 1.5, volatility * 2, 0.03)

    # Direction and magnitude
    sign = 1.0 if direction == "up" else -1.0
    target_change = sign * confidence * max_change_pct / 100

    trajectory = []
    for i in range(points + 1):
        t = i / points
        # Smooth ease-out curve - moves more at start, less at end
        progress = 1 - (1 - t) ** 2.5
        # Very subtle organic noise (doesn't reverse the direction)
        noise = math.sin(t * math.pi * 5) * abs(target_change) * 0.03 * (1 - t)
        predicted_price = price * (1 + target_change * progress + noise)
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
        self.score_ema = None  # Will initialize on first prediction

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

        # Smooth the score to prevent flip-flopping
        raw_score = prediction.get("score", 0.0)
        if self.score_ema is None:
            self.score_ema = raw_score  # Initialize to first real score
        else:
            self.score_ema = self.score_ema * 0.3 + raw_score * 0.7

        # Override direction based on smoothed score
        smoothed_confidence = min(abs(self.score_ema), 1.0)
        if smoothed_confidence < 0.05:
            prediction["direction"] = "neutral"
        elif self.score_ema > 0:
            prediction["direction"] = "up"
        else:
            prediction["direction"] = "down"
        prediction["confidence"] = smoothed_confidence
        prediction["score"] = round(self.score_ema, 4)
        prediction["raw_score"] = round(raw_score, 4)

        # بناء المنحنى
        trajectory = build_trajectory(price, prediction, features, now)

        # Only log predictions with meaningful confidence
        if not (prediction["direction"] == "neutral" and prediction["confidence"] < 0.05):
            self.prediction_history.append({
                "time": now,
                "price_at_prediction": price,
                "direction": prediction["direction"],
                "confidence": prediction["confidence"],
                "score": prediction.get("score", 0),
                "method": prediction["method"],
                "features": {k: v for k, v in features.items()},
                "actual_price_after": None,
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
        now = time.time() * 1000
        horizon_ms = PREDICTION_HORIZON_MIN * 60 * 1000
        for pred in self.prediction_history:
            if pred["actual_price_after"] is not None:
                continue
            elapsed = now - pred["time"]
            if elapsed < horizon_ms:
                continue
            pred["actual_price_after"] = current_price
            pred["evaluated_at"] = now
            if pred["direction"] == "neutral":
                pred["was_correct"] = None
            elif pred["direction"] == "up":
                pred["was_correct"] = current_price > pred["price_at_prediction"]
            else:
                pred["was_correct"] = current_price < pred["price_at_prediction"]

    def get_accuracy(self) -> dict:
        now = time.time() * 1000
        evaluated = [p for p in self.prediction_history if p.get("was_correct") is not None]

        def calc_acc(preds):
            if len(preds) < 5:
                return None
            correct = sum(1 for p in preds if p["was_correct"])
            return round(correct / len(preds) * 100, 1)

        # Filter by when the PREDICTION was made, not when it was evaluated
        h1_preds = [p for p in evaluated if now - p["time"] < 3600_000]
        h24_preds = [p for p in evaluated if now - p["time"] < 86400_000]

        acc_1h = calc_acc(h1_preds)
        acc_24h = calc_acc(h24_preds)
        acc_all = calc_acc(evaluated)

        # Breakdown by direction
        up_preds = [p for p in evaluated if p.get("direction") == "up"]
        down_preds = [p for p in evaluated if p.get("direction") == "down"]
        neutral_preds = [p for p in evaluated if p.get("direction") not in ("up", "down")]

        return {
            "last_1h": calc_acc(h1_preds),
            "last_24h": calc_acc(h24_preds) if calc_acc(h24_preds) != calc_acc(h1_preds) else None,
            "all_time": calc_acc(evaluated),
            "total_predictions": len(self.prediction_history),
            "evaluated": len(evaluated),
            "breakdown": {
                "up": {"count": len(up_preds), "accuracy": calc_acc(up_preds)},
                "down": {"count": len(down_preds), "accuracy": calc_acc(down_preds)},
                "neutral": {"count": len(neutral_preds), "accuracy": calc_acc(neutral_preds)},
            }
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
