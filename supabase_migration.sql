-- ═══════════════════════════════════════════════════════════════
-- BTC Trajectory — Supabase Migration
-- شغّل هذا في Supabase SQL Editor
-- ═══════════════════════════════════════════════════════════════

-- إضافة أعمدة المؤشرات للجدول الموجود
ALTER TABLE btc_prices
  ADD COLUMN IF NOT EXISTS obi REAL,
  ADD COLUMN IF NOT EXISTS rsi REAL,
  ADD COLUMN IF NOT EXISTS volume_delta REAL;

-- جدول جديد لحفظ كل المؤشرات (للتدريب)
CREATE TABLE IF NOT EXISTS btc_features (
  id BIGSERIAL PRIMARY KEY,
  time TIMESTAMPTZ DEFAULT NOW(),
  price REAL NOT NULL,

  -- Order Book Microstructure
  obi REAL,
  obi_rate REAL,
  weighted_depth_imb REAL,
  micro_price_diff REAL,
  vamp_diff REAL,
  spread_pct REAL,

  -- Trade Flow
  volume_delta REAL,
  trade_intensity REAL,
  taker_buy_ratio REAL,

  -- Technical Indicators
  rsi_14 REAL,
  ema_9_diff REAL,
  ema_21_diff REAL,
  ema_50_diff REAL,
  macd_hist REAL,
  macd_line REAL,
  bb_position REAL,
  bb_width REAL,
  atr_14 REAL,
  atr_pct REAL,

  -- Momentum & Volatility
  momentum_1m REAL,
  momentum_5m REAL,
  volatility_5m REAL
);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_btc_features_time ON btc_features (time);

-- جدول لتسجيل التوقعات والنتائج (للتقييم)
CREATE TABLE IF NOT EXISTS btc_predictions (
  id BIGSERIAL PRIMARY KEY,
  time TIMESTAMPTZ DEFAULT NOW(),
  price_at_prediction REAL NOT NULL,
  direction TEXT NOT NULL,       -- 'up', 'down', 'neutral'
  confidence REAL NOT NULL,
  score REAL,
  method TEXT,                   -- 'rules' or 'ml'
  actual_price_after REAL,       -- يُملأ بعد 10 دقائق
  was_correct BOOLEAN            -- يُحسب بعد 10 دقائق
);

CREATE INDEX IF NOT EXISTS idx_btc_predictions_time ON btc_predictions (time);

-- ═══════════════════════════════════════════════════════════════
-- RLS (Row Level Security) — اجعلها مفتوحة للقراءة
-- ═══════════════════════════════════════════════════════════════
ALTER TABLE btc_features ENABLE ROW LEVEL SECURITY;
ALTER TABLE btc_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow anon read btc_features" ON btc_features
  FOR SELECT USING (true);
CREATE POLICY "Allow anon insert btc_features" ON btc_features
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anon read btc_predictions" ON btc_predictions
  FOR SELECT USING (true);
CREATE POLICY "Allow anon insert btc_predictions" ON btc_predictions
  FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow anon update btc_predictions" ON btc_predictions
  FOR UPDATE USING (true);
