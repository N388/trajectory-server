"""
BTC Trajectory Server — Configuration
"""
import os

# ─── Supabase ────────────────────────────────────────────────
SB_URL = os.getenv("SB_URL", "https://rgspgaoqzpljwjkhqsmo.supabase.co")
SB_KEY = os.getenv("SB_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJnc3BnYW9xenBsandqa2hxc21vIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU5MTkwMjIsImV4cCI6MjA5MTQ5NTAyMn0.B0UXZVJ0pvUOnX6-WqQWjUh5RMYQTSysvJksDPStwY8")
SB_HEADERS = {
    "Content-Type": "application/json",
    "apikey": SB_KEY,
    "Authorization": f"Bearer {SB_KEY}",
}

# ─── Binance WebSocket Streams ───────────────────────────────
WS_BASE = "wss://stream.binance.com:9443/ws"
WS_COMBINED = "wss://data-stream.binance.vision/stream?streams=btcusdt@ticker/btcusdt@depth20@1000ms/btcusdt@kline_1m/btcusdt@aggTrade"

# ─── Binance REST ────────────────────────────────────────────
REST_KLINES = "https://data-api.binance.vision/api/v3/klines"

# ─── Prediction Settings ─────────────────────────────────────
PREDICTION_HORIZON_MIN = 10       # توقع 10 دقائق للأمام
PREDICTION_INTERVAL_SEC = 10      # توقع جديد كل 10 ثوانٍ
TRAJECTORY_POINTS = 120           # عدد نقاط المنحنى
CONFIDENCE_THRESHOLD = 0.55       # الحد الأدنى لعرض التوقع
SAVE_INTERVAL_SEC = 10            # حفظ في Supabase كل 10 ثوانٍ

# ─── Data Collection ─────────────────────────────────────────
KLINE_HISTORY_COUNT = 500         # عدد الشموع التاريخية عند البدء
MAX_PRICE_HISTORY = 86400         # أقصى عدد نقاط سعر محفوظة (24 ساعة @ 1/ث)
MAX_TRADE_BUFFER = 5000           # أقصى عدد صفقات محفوظة مؤقتاً
INDICATOR_WINDOW = 60             # نافذة حساب المؤشرات (عدد شموع)

# ─── ML Model ────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model.json")
RETRAIN_INTERVAL_HOURS = 168      # إعادة تدريب كل أسبوع
MIN_TRAINING_SAMPLES = 10000      # الحد الأدنى لبدء التدريب

# ─── Server ──────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = ["*"]
