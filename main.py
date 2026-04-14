"""
BTC Trajectory — API Server
FastAPI server يجمع البيانات ويحسب المؤشرات ويرسل التوقعات
"""
import asyncio
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from config import HOST, PORT, CORS_ORIGINS, PREDICTION_INTERVAL_SEC
from data_collector import DataCollector
from predictor import PredictionEngine
from notifier import send_telegram_alert

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ─── Global State ────────────────────────────────────────────
collector = DataCollector()
engine = PredictionEngine()
latest_prediction = {}


# ─── Background Tasks ───────────────────────────────────────
async def prediction_loop():
    """New prediction every 10 seconds, saved to Supabase"""
    global latest_prediction
    while True:
        try:
            if collector.current_price:
                features = collector.indicators.get_feature_vector()
                prediction = engine.predict(collector.current_price, features)
                latest_prediction = prediction

                # Save prediction to Supabase
                asyncio.create_task(collector.save_prediction(prediction))

                asyncio.create_task(send_telegram_alert(prediction))

                # Evaluate predictions from 10 minutes ago
                asyncio.create_task(collector.evaluate_old_predictions(collector.current_price))

                engine.evaluate_predictions(collector.current_price)
        except Exception as e:
            logger.error(f"Prediction error: {e}")

        await asyncio.sleep(PREDICTION_INTERVAL_SEC)


# ─── App Lifecycle ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """تشغيل المهام عند بدء وإيقاف السيرفر"""
    logger.info("Starting BTC Trajectory Server...")
    engine.initialize()

    # تشغيل جمع البيانات و حلقة التوقع
    collector_task = asyncio.create_task(collector.run())
    prediction_task = asyncio.create_task(prediction_loop())

    yield

    logger.info("Shutting down...")
    collector.stop()
    collector_task.cancel()
    prediction_task.cancel()


# ─── FastAPI App ─────────────────────────────────────────────
app = FastAPI(
    title="BTC Trajectory API",
    description="تحليل مباشر وتوقع لسعر البيتكوين",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    index_file = os.path.join(os.path.dirname(__file__), "frontend", "dist", "index.html")
    if os.path.isfile(index_file):
        return FileResponse(index_file)
    return {
        "service": "BTC Trajectory API",
        "status": "running",
        "price": collector.current_price,
    }


@app.head("/")
@app.head("/api/state")
async def head_check():
    from fastapi.responses import Response
    return Response(status_code=200)


@app.get("/api/prediction")
async def get_prediction():
    """
    التوقع الحالي — يستدعيه React كل 10 ثوانٍ
    """
    if not latest_prediction:
        return {"error": "No prediction yet. Waiting for data..."}

    return {
        "direction": latest_prediction.get("direction"),
        "confidence": latest_prediction.get("confidence"),
        "score": latest_prediction.get("score"),
        "method": latest_prediction.get("method"),
        "trajectory": latest_prediction.get("trajectory", []),
        "price": latest_prediction.get("price"),
        "timestamp": latest_prediction.get("timestamp"),
    }


@app.get("/api/features")
async def get_features():
    """المؤشرات الحالية"""
    return collector.indicators.get_feature_vector()


@app.get("/api/accuracy")
async def get_accuracy():
    """دقة التوقعات"""
    return engine.get_accuracy()


@app.get("/api/feature-importance")
async def get_feature_importance():
    """أهمية المؤشرات"""
    return engine.get_feature_importance()


@app.get("/api/state")
async def get_state():
    """الحالة الكاملة — للتشخيص"""
    state = collector.get_state()
    state["prediction"] = {
        "direction": latest_prediction.get("direction"),
        "confidence": latest_prediction.get("confidence"),
        "method": latest_prediction.get("method"),
    }
    state["accuracy"] = engine.get_accuracy()
    return state


@app.get("/api/signals")
async def get_signals():
    """إشارات المؤشرات الفردية — لعرضها في الواجهة"""
    features = collector.indicators.get_feature_vector()
    from predictor import rule_based_predict
    result = rule_based_predict(features)
    return {
        "signals": result.get("signals", {}),
        "composite_score": result.get("score", 0),
    }


@app.post("/api/train")
async def trigger_training():
    """
    تشغيل تدريب النموذج يدوياً
    ⚠️ يحتاج بيانات كافية في Supabase
    """
    from trainer import run_training
    result = await run_training()
    if "error" not in result:
        engine.initialize()  # إعادة تحميل النموذج الجديد
    return result


@app.get("/api/test-telegram")
async def test_telegram():
    """Test Telegram notification"""
    test_pred = {
        "direction": "up",
        "confidence": 0.99,
        "price": collector.current_price or 0,
        "method": "test",
    }
    import notifier
    notifier._last_alert_time = 0  # Reset cooldown for test
    await send_telegram_alert(test_pred)
    return {"status": "sent", "chat_id": "5164211"}


# ═══════════════════════════════════════════════════════════════
# Serve frontend static files
# ═══════════════════════════════════════════════════════════════
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(frontend_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dir, "assets")), name="assets")

    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        file_path = os.path.join(frontend_dir, path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dir, "index.html"))


# ═══════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
