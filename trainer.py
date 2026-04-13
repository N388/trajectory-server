"""
BTC Trajectory — Model Trainer
تدريب نموذج XGBoost على البيانات التاريخية
يُشغّل يدوياً أو تلقائياً بعد جمع بيانات كافية
"""
import os
import json
import logging
import numpy as np
from datetime import datetime, timezone

import httpx

from config import (
    SB_URL, SB_HEADERS, MODEL_PATH,
    MIN_TRAINING_SAMPLES, PREDICTION_HORIZON_MIN,
)

logger = logging.getLogger("trainer")

# المؤشرات المستخدمة كمدخلات للنموذج
FEATURE_COLS = [
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


async def fetch_training_data() -> list:
    """
    تحميل بيانات التدريب من Supabase
    يحتاج جدول btc_features فيه المؤشرات المحسوبة
    """
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            # تحميل كل البيانات المتاحة
            resp = await client.get(
                f"{SB_URL}/rest/v1/btc_features",
                params={
                    "select": "*",
                    "order": "time.asc",
                    "limit": 100000,
                },
                headers=SB_HEADERS,
            )
            rows = resp.json()

        if not isinstance(rows, list):
            logger.error("Invalid response from Supabase")
            return []

        logger.info(f"Fetched {len(rows)} training samples")
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch training data: {e}")
        return []


def prepare_dataset(rows: list) -> tuple:
    """
    تحضير البيانات للتدريب
    - حساب الهدف (target): هل ارتفع السعر بعد 10 دقائق؟
    - تقسيم زمني (مو عشوائي!)
    """
    if len(rows) < MIN_TRAINING_SAMPLES:
        logger.warning(f"Not enough data: {len(rows)} < {MIN_TRAINING_SAMPLES}")
        return None, None, None, None

    # تحويل لـ numpy
    X_list = []
    y_list = []

    # ترتيب حسب الوقت
    rows.sort(key=lambda r: r.get("time", ""))

    # فترة التوقع بالنقاط (كل 10 ثوانٍ = 60 نقطة لـ 10 دقائق)
    lookahead = PREDICTION_HORIZON_MIN * 6  # 10 * 6 = 60 نقطة

    for i in range(len(rows) - lookahead):
        row = rows[i]
        future_row = rows[i + lookahead]

        # استخراج المؤشرات
        features = []
        skip = False
        for col in FEATURE_COLS:
            val = row.get(col)
            if val is None:
                skip = True
                break
            features.append(float(val))

        if skip:
            continue

        # الهدف: هل ارتفع السعر؟
        current_price = float(row.get("price", 0))
        future_price = float(future_row.get("price", 0))

        if current_price <= 0 or future_price <= 0:
            continue

        direction = 1 if future_price > current_price else 0
        X_list.append(features)
        y_list.append(direction)

    if len(X_list) < 1000:
        logger.warning(f"Not enough valid samples after filtering: {len(X_list)}")
        return None, None, None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # تقسيم زمني: 70% تدريب، 15% اختبار، 15% تحقق
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(
        f"Dataset: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    logger.info(
        f"Class balance: train_up={y_train.mean():.2%}, "
        f"val_up={y_val.mean():.2%}, test_up={y_test.mean():.2%}"
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), FEATURE_COLS


def train_model(
    train_data: tuple,
    val_data: tuple,
    test_data: tuple,
    feature_names: list,
) -> dict:
    """
    تدريب نموذج XGBoost
    Returns: {"model_path", "accuracy", "metrics"}
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return {"error": "xgboost not installed"}

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # معاملات مضبوطة لتجنب overfitting
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "seed": 42,
    }

    # تدريب مع early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    # تقييم على بيانات الاختبار (لم يرها النموذج أبداً)
    test_pred = model.predict(dtest)
    test_direction = (test_pred > 0.5).astype(int)
    test_accuracy = (test_direction == y_test).mean()

    # تقييم على بيانات التدريب (للمقارنة)
    train_pred = model.predict(dtrain)
    train_direction = (train_pred > 0.5).astype(int)
    train_accuracy = (train_direction == y_train).mean()

    # أهمية المؤشرات
    importance = model.get_score(importance_type="gain")
    total_imp = sum(importance.values()) if importance else 1
    top_features = sorted(
        [(k, v / total_imp) for k, v in importance.items()],
        key=lambda x: -x[1]
    )[:10]

    logger.info(f"Train accuracy: {train_accuracy:.2%}")
    logger.info(f"Test accuracy:  {test_accuracy:.2%}")
    logger.info(f"Top features: {top_features[:5]}")

    # تحذير من overfitting
    if train_accuracy - test_accuracy > 0.1:
        logger.warning(
            f"⚠️ Possible overfitting! Train={train_accuracy:.2%} vs Test={test_accuracy:.2%}"
        )

    # حفظ النموذج
    os.makedirs(os.path.dirname(MODEL_PATH) or "models", exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # حفظ metadata
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "best_iteration": model.best_iteration,
        "top_features": [{"name": k, "importance": round(v, 4)} for k, v in top_features],
    }
    meta_path = MODEL_PATH.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "model_path": MODEL_PATH,
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "samples": len(y_train) + len(y_val) + len(y_test),
        "top_features": top_features[:5],
        "overfitting_warning": train_accuracy - test_accuracy > 0.1,
    }


async def run_training():
    """
    تشغيل عملية التدريب الكاملة
    """
    logger.info("Starting model training...")

    # 1. تحميل البيانات
    rows = await fetch_training_data()
    if not rows:
        return {"error": "No training data available"}

    # 2. تحضير البيانات
    result = prepare_dataset(rows)
    train_data, val_data, test_data, features = result

    if train_data is None:
        return {"error": f"Not enough data. Need {MIN_TRAINING_SAMPLES}+ samples."}

    # 3. تدريب
    metrics = train_model(train_data, val_data, test_data, features)

    return metrics
