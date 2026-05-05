"""
BTC Trajectory — train_model.py
تدريب نموذج Logistic Regression (L1) على بيانات Supabase

Schema المؤكد:
    btc_predictions: id, time (timestamptz), price_at_prediction,
                     direction, confidence, actual_price_after, was_correct
    btc_features:    id, time (timestamptz), price, + 22 ميزة فنية

الربط: btc_predictions.time = btc_features.time (نفس النوع، join مباشر)

الاستخدام:
    python train_model.py              # تدريب كامل
    python train_model.py --dry-run   # فحص البيانات فقط
    python train_model.py --skip-wf   # بدون Walk-Forward (أسرع)

المتطلبات:
    pip install supabase scikit-learn pandas numpy joblib python-dotenv
"""

import os
import sys
import json
import argparse
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# الإعداد
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# الـ 22 ميزة — مطابقة تماماً لأعمدة btc_features
FEATURE_COLUMNS = [
    "obi", "obi_rate", "weighted_depth_imb",
    "micro_price_diff", "vamp_diff", "spread_pct",
    "volume_delta", "trade_intensity", "taker_buy_ratio",
    "rsi_14", "ema_9_diff", "ema_21_diff", "ema_50_diff",
    "macd_hist", "macd_line",
    "bb_position", "bb_width",
    "atr_14", "atr_pct",
    "momentum_1m", "momentum_5m", "volatility_5m",
]

RULE_BASED_ACCURACY = 0.516
TRAIN_RATIO         = 0.80
WALK_FORWARD_FOLDS  = 5
MIN_SAMPLES         = 500
DEFAULT_OUTPUT      = "btc_model.pkl"


# ═══════════════════════════════════════════════════════════════
# الاتصال بـ Supabase
# ═══════════════════════════════════════════════════════════════

def get_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        log.error("أضف SUPABASE_URL و SUPABASE_KEY في .env")
        sys.exit(1)
    try:
        from supabase import create_client
        client = create_client(url, key)
        log.info(f"✓ Supabase متصل: {url[:45]}...")
        return client
    except ImportError:
        log.error("pip install supabase")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# جلب البيانات
# ═══════════════════════════════════════════════════════════════

def fetch_all(client, table: str, query_builder) -> pd.DataFrame:
    """جلب كل الصفوف بالدفعات — Supabase حد 1000 سطر/طلب.

    query_builder: callable يُرجع query جديد في كل استدعاء (لتفادي تراكم
    معاملات range على نفس الكائن).
    """
    rows, offset, batch = [], 0, 1000
    while True:
        r = query_builder().range(offset, offset + batch - 1).execute()
        if not r.data:
            break
        rows.extend(r.data)
        log.info(f"  {table}: {len(rows):,} سطر...")
        if len(r.data) < batch:
            break
        offset += batch
    return pd.DataFrame(rows)


def load_data(client) -> pd.DataFrame:
    """
    جلب التوقعات والمؤشرات ودمجهم.

    Target الحقيقي:
        label = actual_price_after > price_at_prediction
        مستقل تماماً عن نظام القواعد — حقيقة السوق فقط.
    """

    # ── التوقعات المقيّمة ──
    log.info("جلب btc_predictions...")
    pred_df = fetch_all(
        client, "btc_predictions",
        lambda: client.table("btc_predictions")
        .select("id, time, price_at_prediction, direction, actual_price_after, was_correct")
        .neq("direction", "neutral")
        .not_.is_("was_correct", "null")
        .not_.is_("actual_price_after", "null")
        .order("time", desc=False),
    )
    log.info(f"✓ التوقعات المُقيّمة: {len(pred_df):,}")

    if pred_df.empty:
        log.error("لا توجد توقعات مُقيّمة")
        sys.exit(1)

    # ── المؤشرات في نفس الفترة الزمنية ──
    t_min = pred_df["time"].min()
    t_max = pred_df["time"].max()
    log.info(f"جلب btc_features ({t_min[:10]} → {t_max[:10]})...")

    feat_df = fetch_all(
        client, "btc_features",
        lambda: client.table("btc_features")
        .select("time, " + ", ".join(FEATURE_COLUMNS))
        .gte("time", t_min)
        .lte("time", t_max)
        .order("time", desc=False),
    )
    log.info(f"✓ المؤشرات: {len(feat_df):,}")

    if feat_df.empty:
        log.error("btc_features فارغ في هذه الفترة")
        sys.exit(1)

    # ── Merge على time (timestamptz في كلا الجدولين) ──
    df = pd.merge(pred_df, feat_df, on="time", how="inner")
    log.info(f"✓ بعد الدمج: {len(df):,} سجل")

    if len(df) < MIN_SAMPLES:
        log.error(f"بعد الدمج: {len(df)} سجل — الحد الأدنى {MIN_SAMPLES}")
        log.error("تحقق من أن time في كلا الجدولين يتطابق تماماً بالثانية")
        sys.exit(1)

    # ── Label الحقيقي من حركة السعر ──
    df["label"] = (df["actual_price_after"] > df["price_at_prediction"]).astype(int)

    # ── ترتيب زمني + تنظيف ──
    df = df.sort_values("time").reset_index(drop=True)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0)

    up_pct = df["label"].mean() * 100
    log.info(f"✓ توزيع السوق — صعود: {up_pct:.1f}% | هبوط: {100 - up_pct:.1f}%")

    return df


# ═══════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════

def walk_forward(df: pd.DataFrame, n_folds: int = WALK_FORWARD_FOLDS) -> dict:
    """
    Walk-Forward: تدريب على الماضي، اختبار على المستقبل فقط.
    الاختبار الوحيد الذي يعكس الواقع في التداول الخوارزمي.
    """
    log.info(f"\n{'━' * 55}")
    log.info(f"Walk-Forward Validation ({n_folds} folds)")
    log.info(f"{'━' * 55}")

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values
    n = len(X)
    fold_size = n // (n_folds + 1)
    results = []

    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_end  = fold_size * (i + 2)
        if test_end > n:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]

        pipe = _build_pipe(C=0.1)
        pipe.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipe.predict(X_te))
        majority = max(y_te.mean(), 1 - y_te.mean())
        beat = acc > majority

        results.append({
            "fold": i + 1,
            "train_n": train_end,
            "test_n": len(y_te),
            "accuracy": acc,
            "majority_baseline": majority,
            "beat": beat,
        })
        log.info(
            f"  Fold {i + 1}: train={train_end:,} | test={len(y_te):,} | "
            f"acc={acc * 100:.1f}% | baseline={majority * 100:.1f}%  "
            f"{'✓' if beat else '✗'}"
        )

    if not results:
        return {}

    accs = [r["accuracy"] for r in results]
    log.info(f"\n  الملخص: {np.mean(accs) * 100:.2f}% ± {np.std(accs) * 100:.2f}%")
    log.info(f"  تغلب على baseline: {sum(r['beat'] for r in results)}/{len(results)}")

    return {
        "folds": results,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "beats": sum(r["beat"] for r in results),
        "total": len(results),
    }


# ═══════════════════════════════════════════════════════════════
# بناء النموذج
# ═══════════════════════════════════════════════════════════════

def _build_pipe(C: float = 0.1) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def tune_C(X_tr, y_tr, X_val, y_val) -> float:
    """Grid search على C — validation set فقط"""
    candidates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_C, best_acc = 0.1, 0.0
    log.info("  ضبط C (regularization strength)...")
    for C in candidates:
        pipe = _build_pipe(C)
        pipe.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, pipe.predict(X_val))
        marker = " ←" if acc > best_acc else ""
        log.info(f"    C={C:.3f}  acc={acc * 100:.2f}%{marker}")
        if acc > best_acc:
            best_acc, best_C = acc, C
    log.info(f"  ✓ أفضل C={best_C}")
    return best_C


def train_final(df: pd.DataFrame) -> dict:
    log.info(f"\n{'━' * 55}")
    log.info("التدريب النهائي")
    log.info(f"{'━' * 55}")

    n = len(df)
    split = int(n * TRAIN_RATIO)

    X_tr  = df.iloc[:split][FEATURE_COLUMNS].values
    y_tr  = df.iloc[:split]["label"].values
    X_val = df.iloc[split:][FEATURE_COLUMNS].values
    y_val = df.iloc[split:]["label"].values

    log.info(f"تدريب: {len(X_tr):,} | validation: {len(X_val):,}")

    best_C = tune_C(X_tr, y_tr, X_val, y_val)

    pipe = _build_pipe(best_C)
    pipe.fit(X_tr, y_tr)

    y_pred  = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]
    acc     = accuracy_score(y_val, y_pred)
    auc     = roc_auc_score(y_val, y_proba)

    log.info(f"\n  Validation Accuracy : {acc * 100:.2f}%")
    log.info(f"  ROC-AUC             : {auc:.4f}")
    log.info(f"\n{classification_report(y_val, y_pred, target_names=['هبوط', 'صعود'])}")
    log.info(f"  Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

    # أهمية الميزات
    coef = pipe.named_steps["lr"].coef_[0]
    importance = sorted(
        zip(FEATURE_COLUMNS, coef),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    zeroed = [nm for nm, c in zip(FEATURE_COLUMNS, coef) if abs(c) < 1e-6]

    log.info("  أهم 10 مؤشرات (L1 coefficients):")
    for feat, val in importance[:10]:
        arrow = "↑" if val > 0 else "↓"
        log.info(f"    {arrow} {feat:<25} {val:+.4f}")

    if zeroed:
        log.info(f"  صفّرها L1: {zeroed}")

    diff = (acc - RULE_BASED_ACCURACY) * 100
    log.info(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info(f"  نظام القواعد : {RULE_BASED_ACCURACY * 100:.1f}%")
    log.info(f"  هذا النموذج  : {acc * 100:.2f}%")
    log.info(f"  الفرق        : {diff:+.2f}%")
    log.info(f"  النتيجة      : {'✓ تغلب' if diff > 0 else '✗ لم يتغلب'}")
    log.info(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return {
        "pipe": pipe,
        "best_C": best_C,
        "val_acc": acc,
        "auc": auc,
        "diff": diff,
        "beat": diff > 0,
        "importance": importance,
        "zeroed": zeroed,
        "train_n": len(X_tr),
        "val_n": len(X_val),
    }


# ═══════════════════════════════════════════════════════════════
# الحفظ
# ═══════════════════════════════════════════════════════════════

def save(result: dict, wf: dict, df: pd.DataFrame, path: str):
    artifact = {
        "model":         result["pipe"],
        "feature_names": FEATURE_COLUMNS,
        "model_type":    "logistic_regression_l1",
        "metadata": {
            "trained_at":      datetime.now(timezone.utc).isoformat(),
            "total_samples":   len(df),
            "train_samples":   result["train_n"],
            "val_samples":     result["val_n"],
            "best_C":          result["best_C"],
            "val_accuracy":    result["val_acc"],
            "roc_auc":         result["auc"],
            "rule_baseline":   RULE_BASED_ACCURACY,
            "improvement_pct": result["diff"],
            "beat_rules":      result["beat"],
            "walk_forward":    wf,
            "feature_importance": [
                {"name": nm, "coef": float(c)}
                for nm, c in result["importance"]
            ],
            "zeroed_features": result["zeroed"],
        },
    }

    joblib.dump(artifact, path)
    kb = Path(path).stat().st_size / 1024
    log.info(f"\n✓ النموذج محفوظ: {path} ({kb:.1f} KB)")

    rep = path.replace(".pkl", "_report.json")
    with open(rep, "w", encoding="utf-8") as f:
        json.dump(artifact["metadata"], f, ensure_ascii=False, indent=2, default=str)
    log.info(f"✓ التقرير: {rep}")


# ═══════════════════════════════════════════════════════════════
# التعديل المطلوب على predictor.py
# ═══════════════════════════════════════════════════════════════

PATCH = """
# ── استبدل class MLPredictor كاملاً بهذا الكود ──

class MLPredictor:
    \"\"\"Logistic Regression (L1) — بديل XGBoost\"\"\"

    def __init__(self):
        self.pipeline      = None
        self.feature_names = None
        self.metadata      = {}
        self.is_loaded     = False

    def load_model(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            logger.info(f"No ML model at {path}. Rule-based active.")
            return False
        try:
            import joblib
            artifact           = joblib.load(path)
            self.pipeline      = artifact["model"]
            self.feature_names = artifact["feature_names"]
            self.metadata      = artifact.get("metadata", {})
            self.is_loaded     = True
            acc = self.metadata.get("val_accuracy", 0) * 100
            logger.info(f"ML loaded — val_acc={acc:.1f}%")
            return True
        except Exception as e:
            logger.error(f"ML load failed: {e}")
            return False

    def predict(self, features: dict) -> dict | None:
        if not self.is_loaded:
            return None
        try:
            vals  = [features.get(f, 0.0) for f in self.feature_names]
            prob  = float(self.pipeline.predict_proba([vals])[0][1])
            direction  = "up" if prob > 0.5 else "down"
            confidence = abs(prob - 0.5) * 2
            return {
                "direction":   direction,
                "confidence":  round(confidence, 3),
                "probability": round(prob, 4),
                "score":       round(prob * 2 - 1, 4),
            }
        except Exception as e:
            logger.error(f"ML predict failed: {e}")
            return None
"""


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--skip-wf",  action="store_true")
    p.add_argument("--output",   default=DEFAULT_OUTPUT)
    args = p.parse_args()

    log.info("═" * 55)
    log.info("BTC Trajectory — Training Pipeline v1.1")
    log.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("═" * 55)

    client = get_client()
    df     = load_data(client)

    if args.dry_run:
        log.info(f"\n✓ Dry-run — {len(df):,} سجل جاهز")
        log.info(f"  صعود: {df['label'].sum():,} | هبوط: {(df['label']==0).sum():,}")
        log.info(f"  الفترة: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
        return

    wf     = {} if args.skip_wf else walk_forward(df)
    result = train_final(df)
    save(result, wf, df, args.output)

    log.info("\n" + "═" * 55)
    log.info("التعديل المطلوب على predictor.py:")
    log.info("═" * 55)
    print(PATCH)

    if not result["beat"]:
        log.warning("النموذج لم يتغلب على القواعد — لا تنشره على السيرفر بعد")


if __name__ == "__main__":
    main()
