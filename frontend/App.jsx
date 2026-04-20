import { useState, useEffect, useRef, useCallback } from "react";

// ─── Binance WebSocket ────────────────────────────────────────
const WS_TICKER = "wss://stream.binance.com:9443/ws/btcusdt@ticker";
const WS_DEPTH  = "wss://stream.binance.com:9443/ws/btcusdt@depth20@1000ms";

// ─── Prediction API ──────────────────────────────────────────
const API_URL = "";

// ─── Constants ───────────────────────────────────────────────
const TRAJ_MIN  = 10;
const TRAJ_PTS  = 120;
const HIST_MIN  = 1440; // 24 ساعة
const UPDATE_MS = 10000;

const safeNum = (v, fb = 0) => { const n = Number(v); return isFinite(n) ? n : fb; };

function calcTrajectory(price, imb, startTime) {
  const endPrice = price * (1 + imb * 0.008);
  return Array.from({ length: TRAJ_PTS + 1 }, (_, i) => {
    const t    = i / TRAJ_PTS;
    const ease = t * t * (3 - 2 * t);
    const base = price + (endPrice - price) * ease;
    const wave = Math.sin(t * Math.PI * 3) * Math.abs(imb) * price * 0.0004 * (1 - t * 0.7);
    return { time: startTime + t * TRAJ_MIN * 60000, price: base + wave };
  });
}

function calcImbalance(hist) {
  if (hist.length < 2) return 0;
  const now = Date.now();
  // Use last 60 seconds of data for more stable imbalance
  const recent = hist.filter(p => now - p.time < 60000);
  if (recent.length < 2) {
    // Fallback to last 10 points
    const slice = hist.slice(-Math.min(hist.length, 10));
    const pct = (slice.at(-1).price - slice[0].price) / slice[0].price;
    return Math.max(-1, Math.min(1, pct / 0.003));
  }
  const pct = (recent.at(-1).price - recent[0].price) / recent[0].price;
  return Math.max(-1, Math.min(1, pct / 0.003));
}

// ─── Main Component ───────────────────────────────────────────
export default function App() {
  const canvasRef  = useRef(null);
  const priceHist  = useRef([]);
  const curTrajs   = useRef([]);
  const cloudTrajs = useRef([]);
  const lastMin    = useRef(null);
  const animId     = useRef(null);
  const pulseT     = useRef(0);
  const dpr        = useRef(1);
  const wsTickRef  = useRef(null);
  const wsDepthRef = useRef(null);
  const livePrice  = useRef(null);
  // ── Y-axis zoom ───────────────────────────────────────────
  const yZoom       = useRef(1);
  const isDraggingY = useRef(false);
  const dragStartY  = useRef(0);
  // ── X-axis zoom (horizontal) ───────────────────────────────
  const xViewMin    = useRef(60);
  const isDraggingX = useRef(false);
  const dragStartX2 = useRef(0);
  const dragStartY2 = useRef(0);
  // ── X-axis bar drag (zoom time) ───────────────────────────
  const isDraggingXBar = useRef(false);
  const dragStartXBar  = useRef(0);
  const isDraggingTime = useRef(false);
  // ── Pan offsets ────────────────────────────────────────────
  const xPanOffset  = useRef(0);     // horizontal pan offset in ms
  const yPanOffset  = useRef(0);     // vertical pan offset in price units
  // ── Crosshair ─────────────────────────────────────────────
  const mousePos    = useRef(null);  // {clientX, clientY} for crosshair
  const bidHistory  = useRef([]);

  const autoFollow  = useRef(true);   // true = chart follows current price
  const [showFollow, setShowFollow] = useState(false); // show "back to now" button

  const [info, setInfo] = useState({
    price: null, change24h: 0, bidPct: 50, askPct: 50,
    connected: false, error: null, loaded: false,
    acc60: null,   // accuracy last 60 min
    accAll: null,  // accuracy all time
    predDirection: null, predConfidence: 0, predMethod: null,
    obi: undefined, rsi: 0, atr: 0,
  });

  // ── Alert System ──────────────────────────────────────────
  const lastAlertTime = useRef(0);
  const alertEnabled = useRef(true);

  const playAlertSound = useCallback((direction) => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "sine";
      if (direction === "up") {
        osc.frequency.setValueAtTime(440, ctx.currentTime);
        osc.frequency.linearRampToValueAtTime(660, ctx.currentTime + 0.3);
      } else {
        osc.frequency.setValueAtTime(660, ctx.currentTime);
        osc.frequency.linearRampToValueAtTime(440, ctx.currentTime + 0.3);
      }
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.5);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.5);
    } catch(_) {}
  }, []);

  const sendPushNotification = useCallback((direction, confidence, price) => {
    if (!("Notification" in window)) return;
    if (Notification.permission !== "granted") return;
    const arrow = direction === "up" ? "🟢 صعود ▲" : "🔴 هبوط ▼";
    new Notification("BTC Trajectory", {
      body: `${arrow}\nالسعر: $${price?.toLocaleString("en")}\nالثقة: ${(confidence*100).toFixed(0)}%`,
      icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>₿</text></svg>",
      tag: "btc-alert",
    });
  }, []);

  const checkAndAlert = useCallback((direction, confidence, price) => {
    if (!alertEnabled.current) return;
    if (direction === "neutral") return;
    if (confidence < 0.45) return;
    const now = Date.now();
    if (now - lastAlertTime.current < 300000) return;
    lastAlertTime.current = now;
    playAlertSound(direction);
    sendPushNotification(direction, confidence, price);
  }, [playAlertSound, sendPushNotification]);

  // ── localStorage persistence ────────────────────────────
  const STORAGE_KEY = "btc_chart_state";
  const saveChartState = useCallback(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        xViewMin: xViewMin.current,
        yZoom: yZoom.current,
        xPanOffset: xPanOffset.current,
        yPanOffset: yPanOffset.current,
        autoFollow: autoFollow.current,
      }));
    } catch (_) {}
  }, []);

  // ── Load history + accuracy from Supabase on mount ────────
  useEffect(() => {
    // Restore chart state from localStorage
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (saved) {
        if (isFinite(saved.xViewMin))    xViewMin.current   = saved.xViewMin;
        if (isFinite(saved.yZoom))       yZoom.current      = saved.yZoom;
        if (isFinite(saved.xPanOffset))  xPanOffset.current = saved.xPanOffset;
        if (isFinite(saved.yPanOffset))  yPanOffset.current = saved.yPanOffset;
        if (typeof saved.autoFollow === "boolean") {
          autoFollow.current = saved.autoFollow;
          if (!saved.autoFollow) setShowFollow(true);
        }
      }
    } catch (_) {}

    fetch(`${API_URL}/api/history`)
      .then(r => r.json())
      .then(rows => {
        if (rows.length > 0) {
          priceHist.current = rows;
          setInfo(d => ({ ...d, loaded: true, price: rows.at(-1).price }));
        } else {
          setInfo(d => ({ ...d, loaded: true }));
        }
      })
      .catch(() => setInfo(d => ({ ...d, loaded: true })));
  }, []);

  // Save state on page unload
  useEffect(() => {
    const onUnload = () => saveChartState();
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [saveChartState]);

  // Request push notification permission
  useEffect(() => {
    if ("Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  // Fetch accuracy from API every 60s
  useEffect(() => {
    const fetchAccuracy = async () => {
      try {
        const res = await fetch(`${API_URL}/api/accuracy`);
        const data = await res.json();
        setInfo(d => ({ ...d,
          acc60: data.last_1h,
          accAll: data.all_time,
        }));
      } catch (_) {}
    };
    fetchAccuracy();
    const id = setInterval(fetchAccuracy, 60000);
    return () => clearInterval(id);
  }, []);

  // ── Trajectory from API every 10s (fallback to local) ────
  useEffect(() => {
    let controller = null;
    const id = setInterval(async () => {
      if (controller) controller.abort();
      controller = new AbortController();
      try {
        const res = await fetch(`${API_URL}/api/prediction`, { signal: controller.signal });
        const data = await res.json();
        if (data.error || !data.trajectory) return;

        const now = Date.now();
        const min = Math.floor(now / 60000);

        if (lastMin.current !== null && min !== lastMin.current) {
          cloudTrajs.current = [...cloudTrajs.current, ...curTrajs.current]
            .filter(t => now - t.t0 < 5 * 60000);
          curTrajs.current = [];
        }
        lastMin.current = min;

        const pts = data.trajectory.map(p => ({ time: p.time, price: p.price }));
        curTrajs.current.push({ t0: now, pts });
        if (curTrajs.current.length > 6) curTrajs.current.shift();

        if (data.confidence !== undefined) {
          setInfo(d => ({ ...d,
            predDirection: data.direction,
            predConfidence: data.confidence,
            predMethod: data.method,
          }));
        }
        checkAndAlert(data.direction, data.confidence, data.price);
      } catch(e) {
        if (e.name === 'AbortError') return;
        const price = livePrice.current;
        if (price) {
          const imb = calcImbalance(priceHist.current);
          const pts = calcTrajectory(price, imb, Date.now());
          curTrajs.current.push({ t0: Date.now(), pts });
          if (curTrajs.current.length > 6) curTrajs.current.shift();
        }
      }

      // Also fetch features
      try {
        const fRes = await fetch(`${API_URL}/api/features`, { signal: controller?.signal });
        const fData = await fRes.json();
        if (fData.obi !== undefined) {
          setInfo(d => ({ ...d, obi: fData.obi, rsi: fData.rsi_14, atr: fData.atr_pct, indicatorsReady: fData.rsi_14 !== 0 }));
        }
      } catch(_) {}
    }, 10000);
    return () => { clearInterval(id); if (controller) controller.abort(); };
  }, []);

  // ── WebSocket: Ticker ────────────────────────────────────
  const connectTicker = useCallback(() => {
    if (wsTickRef.current) wsTickRef.current.close();
    const ws = new WebSocket(WS_TICKER);
    ws.onopen = () => setInfo(d => ({ ...d, connected: true, error: null }));
    ws.onmessage = e => {
      const d = JSON.parse(e.data);
      const price     = parseFloat(d.c);
      const change24h = parseFloat(d.P);
      if (!price) return;
      livePrice.current = price;
      const now = Date.now();
      // Reject outlier spikes (>0.5% jump) and duplicate timestamps (<500ms)
      const lastP = priceHist.current.at(-1);
      const isSpike = lastP && Math.abs(price - lastP.price) / lastP.price > 0.005;
      const isDupe = lastP && Math.abs(now - lastP.time) < 500;
      if (!isSpike && !isDupe) {
        priceHist.current.push({ time: now, price });
      }
      priceHist.current = priceHist.current.filter(p => now - p.time < HIST_MIN * 60000);
      // Generate first trajectory immediately on first price
      if (curTrajs.current.length === 0) {
        const imb = calcImbalance(priceHist.current);
        const pts = calcTrajectory(price, imb, now);
        const min = Math.floor(now / 60000);
        lastMin.current = min;
        curTrajs.current.push({ t0: now, pts });
      }
      setInfo(d2 => ({ ...d2, price, change24h }));
    };
    ws.onerror = () => setInfo(d => ({ ...d, error: "خطأ في الاتصال" }));
    ws.onclose = () => { setInfo(d => ({ ...d, connected: false })); setTimeout(connectTicker, 3000); };
    wsTickRef.current = ws;
  }, []);

  // ── WebSocket: Depth ────────────────────────────────────
  const connectDepth = useCallback(() => {
    if (wsDepthRef.current) wsDepthRef.current.close();
    const ws = new WebSocket(WS_DEPTH);
    ws.onmessage = e => {
      const d = JSON.parse(e.data);
      const bidVol = (d.bids || []).reduce((s, [, q]) => s + parseFloat(q), 0);
      const askVol = (d.asks || []).reduce((s, [, q]) => s + parseFloat(q), 0);
      const total  = bidVol + askVol;
      const rawBidPct = total > 0 ? (bidVol / total) * 100 : 50;
      bidHistory.current.push(rawBidPct);
      if (bidHistory.current.length > 10) bidHistory.current.shift();
      const smoothBid = bidHistory.current.reduce((a, b) => a + b, 0) / bidHistory.current.length;
      setInfo(d2 => ({ ...d2, bidPct: Math.round(smoothBid * 10) / 10, askPct: Math.round((100 - smoothBid) * 10) / 10 }));
    };
    ws.onclose = () => setTimeout(connectDepth, 3000);
    wsDepthRef.current = ws;
  }, []);

  useEffect(() => {
    connectTicker(); connectDepth();
    return () => { wsTickRef.current?.close(); wsDepthRef.current?.close(); };
  }, [connectTicker, connectDepth]);

  // ── Canvas DPR ───────────────────────────────────────────
  // Resize canvas whenever window size changes
  useEffect(() => {
    const resizeCanvas = () => {
      dpr.current = Math.min(window.devicePixelRatio || 1, 2);
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      canvas.width  = rect.width  * dpr.current;
      canvas.height = rect.height * dpr.current;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    // Also resize after layout settles
    const t = setTimeout(resizeCanvas, 100);
    return () => { window.removeEventListener("resize", resizeCanvas); clearTimeout(t); };
  }, []);

  // ── Interaction: Pan, Zoom, Crosshair (TradingView style) ─
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const PL = 74, PR = 14, PT = 20, PB = 38;

    const getCanvasXY = (clientX, clientY) => {
      const rect = canvas.getBoundingClientRect();
      const x = (clientX - rect.left) * (canvas.width / rect.width) / (dpr.current || 1);
      const y = (clientY - rect.top) * (canvas.height / rect.height) / (dpr.current || 1);
      return { x, y };
    };
    const isOnYAxis = (clientX) => getCanvasXY(clientX, 0).x < PL;
    const isOnChart = (clientX) => getCanvasXY(clientX, 0).x >= PL;

    // ── Y-axis drag = zoom Y ──
    const onMouseDown = (e) => {
      if (!isOnYAxis(e.clientX)) return;
      isDraggingY.current = true;
      dragStartY.current  = e.clientY;
      e.preventDefault();
      e.stopPropagation();
    };
    const onMouseMoveY = (e) => {
      if (!isDraggingY.current) return;
      e.preventDefault();
      const dy = dragStartY.current - e.clientY;
      yZoom.current = Math.max(0.05, Math.min(20, yZoom.current * (1 + dy * 0.005)));
      dragStartY.current = e.clientY;
      saveChartState();
    };
    const onMouseUpY = () => { isDraggingY.current = false; };

    // ── Chart area drag = pan (X + Y) ──
    const onMouseDownX = (e) => {
      if (e.button !== 0) return;
      const { x, y } = getCanvasXY(e.clientX, e.clientY);
      const H = canvas.getBoundingClientRect().height;
      if (x < PL) return; // Y-axis area, handled separately
      if (y > H - PB) {
        // Time axis bar - drag to zoom horizontally
        isDraggingTime.current = true;
        dragStartX2.current = e.clientX;
        e.preventDefault();
        return;
      }
      // Normal chart area - pan
      isDraggingX.current = true;
      dragStartX2.current = e.clientX;
      dragStartY2.current = e.clientY;
      e.preventDefault();
      e.stopPropagation();
    };
    const onMouseMoveX = (e) => {
      if (isDraggingTime.current) {
        e.preventDefault();
        const dx = e.clientX - dragStartX2.current;
        const zoomFactor = 1 - dx * 0.003;
        xViewMin.current = Math.max(2, Math.min(1440, xViewMin.current * zoomFactor));
        dragStartX2.current = e.clientX;
        saveChartState();
        return;
      }
      if (!isDraggingX.current) return;
      e.preventDefault();
      const dx = e.clientX - dragStartX2.current;
      const dy = e.clientY - dragStartY2.current;
      const rect = canvas.getBoundingClientRect();
      const CW = rect.width - PL - PR;
      const CH = rect.height - PT - PB;
      const totalTimeMs = (xViewMin.current + TRAJ_MIN) * 60000;
      // Pan X: convert pixel delta to time delta
      xPanOffset.current += (dx / CW) * totalTimeMs;
      // Pan Y: convert pixel delta to price delta (need current pRange)
      // We approximate pRange from current zoom state
      const allP = [
        ...priceHist.current.map(p => p.price),
        ...curTrajs.current.flatMap(t => t.pts.map(p => p.price)),
      ];
      if (allP.length > 0) {
        let rawMin = Infinity, rawMax = -Infinity;
        for (let i = 0; i < allP.length; i++) {
          if (allP[i] < rawMin) rawMin = allP[i];
          if (allP[i] > rawMax) rawMax = allP[i];
        }
        const halfRange = Math.max((rawMax - rawMin) * 0.15, rawMin * 0.001) + (rawMax - rawMin) / 2;
        const zoomedHalf = halfRange / yZoom.current;
        const pRange = zoomedHalf * 2;
        yPanOffset.current += (dy / CH) * pRange;
      }
      dragStartX2.current = e.clientX;
      dragStartY2.current = e.clientY;
      autoFollow.current = false;
      setShowFollow(true);
      saveChartState();
    };
    const onMouseUpX = () => { isDraggingX.current = false; isDraggingTime.current = false; };

    // ── X-axis bar drag = zoom time ──
    const isOnXBar = (clientX, clientY) => {
      const { x, y } = getCanvasXY(clientX, clientY);
      const H = canvas.height / (dpr.current || 1);
      return x >= PL && y > H - PB;
    };
    const onMouseDownXBar = (e) => {
      if (e.button !== 0 || !isOnXBar(e.clientX, e.clientY)) return;
      isDraggingXBar.current = true;
      dragStartXBar.current  = e.clientX;
      e.preventDefault();
      e.stopPropagation();
    };
    const onMouseMoveXBar = (e) => {
      if (!isDraggingXBar.current) return;
      e.preventDefault();
      const dx = e.clientX - dragStartXBar.current;
      const sensitivity = 0.008;
      // drag right = zoom out (more time), drag left = zoom in (less time)
      xViewMin.current = Math.max(2, Math.min(1440, xViewMin.current * (1 + dx * sensitivity)));
      dragStartXBar.current = e.clientX;
      autoFollow.current = false;
      setShowFollow(true);
      saveChartState();
    };
    const onMouseUpXBar = () => { isDraggingXBar.current = false; };

    // ── Wheel = zoom toward cursor ──
    const onWheel = (e) => {
      e.preventDefault();
      const { x, y } = getCanvasXY(e.clientX, e.clientY);
      const rect = canvas.getBoundingClientRect();
      const W = rect.width, H = rect.height;
      const CW = W - PL - PR, CH = H - PT - PB;
      // Zoom factor: scroll up = zoom in (negative deltaY), scroll down = zoom out
      const zoomFactor = 1 + e.deltaY * 0.001;
      // Where is the cursor in the chart (0..1)?
      const chartXRatio = Math.max(0, Math.min(1, (x - PL) / CW));
      const chartYRatio = Math.max(0, Math.min(1, (y - PT) / CH));
      // Zoom X (time)
      const oldXViewMin = xViewMin.current;
      xViewMin.current = Math.max(2, Math.min(1440, xViewMin.current * zoomFactor));
      // Adjust pan so the point under cursor stays fixed
      const totalOld = (oldXViewMin + TRAJ_MIN) * 60000;
      const totalNew = (xViewMin.current + TRAJ_MIN) * 60000;
      // The cursor time = timeEnd + xPanOffset - (1 - chartXRatio) * totalOld
      // After zoom, we want the same time under cursor:
      // xPanOffset_new = xPanOffset + (1 - chartXRatio) * (totalNew - totalOld)
      xPanOffset.current += (1 - chartXRatio) * (totalNew - totalOld);
      // Zoom Y (price)
      const oldYZoom = yZoom.current;
      yZoom.current = Math.max(0.05, Math.min(20, yZoom.current / zoomFactor));
      // Adjust yPanOffset so price under cursor stays fixed
      // The cursor is at chartYRatio from top. Price at cursor:
      // p = maxP - chartYRatio * pRange
      // After zoom change, we need to adjust yPanOffset to keep this price at same Y
      const allP = [
        ...priceHist.current.map(p => p.price),
        ...curTrajs.current.flatMap(t => t.pts.map(p => p.price)),
      ];
      if (allP.length > 0) {
        let rawMin = Infinity, rawMax = -Infinity;
        for (let i = 0; i < allP.length; i++) {
          if (allP[i] < rawMin) rawMin = allP[i];
          if (allP[i] > rawMax) rawMax = allP[i];
        }
        const mid = (rawMin + rawMax) / 2;
        const halfRangeBase = Math.max((rawMax - rawMin) * 0.15, rawMin * 0.001) + (rawMax - rawMin) / 2;
        const oldHalf = halfRangeBase / oldYZoom;
        const newHalf = halfRangeBase / yZoom.current;
        const oldMid = mid + yPanOffset.current;
        // Price at cursor with old zoom: oldMid + oldHalf - chartYRatio * 2 * oldHalf
        const priceAtCursor = oldMid + oldHalf * (1 - 2 * chartYRatio);
        // With new zoom, we want same price at same Y:
        // newMid + newHalf * (1 - 2*chartYRatio) = priceAtCursor
        // newMid = priceAtCursor - newHalf*(1 - 2*chartYRatio)
        // yPanOffset_new = newMid - mid = priceAtCursor - newHalf*(1 - 2*chartYRatio) - mid
        yPanOffset.current = priceAtCursor - newHalf * (1 - 2 * chartYRatio) - mid;
      }
      if (!autoFollow.current || Math.abs(xPanOffset.current) > 1000) {
        autoFollow.current = false;
        setShowFollow(true);
      }
      saveChartState();
    };

    // ── Touch support ──
    const onTouchStart = (e) => {
      if (e.touches.length !== 1) return;
      const t = e.touches[0];
      if (isOnYAxis(t.clientX)) {
        isDraggingY.current = true;
        dragStartY.current  = t.clientY;
      } else {
        isDraggingX.current  = true;
        dragStartX2.current  = t.clientX;
        dragStartY2.current  = t.clientY;
      }
    };
    const onTouchMove = (e) => {
      if (e.touches.length !== 1) return;
      e.preventDefault();
      const t = e.touches[0];
      if (isDraggingY.current) {
        const dy = dragStartY.current - t.clientY;
        yZoom.current = Math.max(0.05, Math.min(20, yZoom.current * (1 + dy * 0.005)));
        dragStartY.current = t.clientY;
      }
      if (isDraggingX.current) {
        const dx = t.clientX - dragStartX2.current;
        const dy = t.clientY - dragStartY2.current;
        const rect = canvas.getBoundingClientRect();
        const CW = rect.width - PL - PR;
        const CH = rect.height - PT - PB;
        const totalTimeMs = (xViewMin.current + TRAJ_MIN) * 60000;
        xPanOffset.current += (dx / CW) * totalTimeMs;
        const allP = priceHist.current.map(p => p.price);
        if (allP.length > 0) {
          let rawMin = Infinity, rawMax = -Infinity;
          for (let i = 0; i < allP.length; i++) {
            if (allP[i] < rawMin) rawMin = allP[i];
            if (allP[i] > rawMax) rawMax = allP[i];
          }
          const halfRange = Math.max((rawMax - rawMin) * 0.15, rawMin * 0.001) + (rawMax - rawMin) / 2;
          const pRange = (halfRange / yZoom.current) * 2;
          yPanOffset.current += (dy / CH) * pRange;
        }
        dragStartX2.current = t.clientX;
        dragStartY2.current = t.clientY;
        autoFollow.current = false;
        setShowFollow(true);
      }
      saveChartState();
    };
    const onTouchEnd = () => { isDraggingY.current = false; isDraggingX.current = false; };

    // ── Double-click = reset ──
    const onDblClick = (e) => {
      if (isOnYAxis(e.clientX)) {
        yZoom.current = 1;
        yPanOffset.current = 0;
      } else {
        xViewMin.current = 60;
        xPanOffset.current = 0;
        yPanOffset.current = 0;
        yZoom.current = 1;
        autoFollow.current = true;
        setShowFollow(false);
      }
      saveChartState();
    };

    // ── Crosshair tracking ──
    const onMouseMoveCrosshair = (e) => {
      if (isDraggingY.current || isDraggingX.current) return;
      const { x, y } = getCanvasXY(e.clientX, e.clientY);
      if (x >= PL && x <= PL + (canvas.width / (dpr.current||1)) - PL - PR && y >= PT && y <= PT + (canvas.height / (dpr.current||1)) - PT - PB) {
        mousePos.current = { x, y };
      } else {
        mousePos.current = null;
      }
    };
    const onMouseLeave = () => { mousePos.current = null; };

    canvas.addEventListener("mousedown",  onMouseDown);
    canvas.addEventListener("mousedown",  onMouseDownXBar);
    canvas.addEventListener("mousedown",  onMouseDownX);
    window.addEventListener("mousemove",  onMouseMoveY);
    window.addEventListener("mousemove",  onMouseMoveXBar);
    window.addEventListener("mousemove",  onMouseMoveX);
    window.addEventListener("mousemove",  onMouseMoveCrosshair);
    window.addEventListener("mouseup",    onMouseUpY);
    window.addEventListener("mouseup",    onMouseUpXBar);
    window.addEventListener("mouseup",    onMouseUpX);
    canvas.addEventListener("wheel",      onWheel, { passive: false });
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    window.addEventListener("touchmove",  onTouchMove,  { passive: false });
    window.addEventListener("touchend",   onTouchEnd);
    canvas.addEventListener("dblclick",   onDblClick);
    canvas.addEventListener("mouseleave", onMouseLeave);

    return () => {
      canvas.removeEventListener("mousedown",  onMouseDown);
      canvas.removeEventListener("mousedown",  onMouseDownXBar);
      canvas.removeEventListener("mousedown",  onMouseDownX);
      window.removeEventListener("mousemove",  onMouseMoveY);
      window.removeEventListener("mousemove",  onMouseMoveXBar);
      window.removeEventListener("mousemove",  onMouseMoveX);
      window.removeEventListener("mousemove",  onMouseMoveCrosshair);
      window.removeEventListener("mouseup",    onMouseUpY);
      window.removeEventListener("mouseup",    onMouseUpXBar);
      window.removeEventListener("mouseup",    onMouseUpX);
      canvas.removeEventListener("wheel",      onWheel);
      canvas.removeEventListener("touchstart", onTouchStart);
      window.removeEventListener("touchmove",  onTouchMove);
      window.removeEventListener("touchend",   onTouchEnd);
      canvas.removeEventListener("dblclick",   onDblClick);
      canvas.removeEventListener("mouseleave", onMouseLeave);
    };
  }, [saveChartState]);


  // ── Draw ────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const d   = dpr.current;
    const W   = canvas.width  / d;
    const H   = canvas.height / d;
    const now = Date.now();
    pulseT.current = (pulseT.current + 0.025) % (Math.PI * 2);

    ctx.save();
    ctx.scale(d, d);

    const PL = 74, PR = 14, PT = 20, PB = 38;
    const CW = W - PL - PR, CH = H - PT - PB;
    // Apply pan offset: when autoFollow, ignore xPanOffset
    const panX = autoFollow.current ? 0 : xPanOffset.current;
    const timeEnd   = now + TRAJ_MIN * 60000 - panX;
    const timeStart = timeEnd - (xViewMin.current + TRAJ_MIN) * 60000;
    const timeRange = timeEnd - timeStart;

    const allP = [
      ...priceHist.current.map(p => p.price),
      ...curTrajs.current.flatMap(t => t.pts.map(p => p.price)),
      ...cloudTrajs.current.flatMap(t => t.pts.map(p => p.price)),
    ];

    ctx.fillStyle = "#040810";
    ctx.fillRect(0, 0, W, H);
    for (let y = 0; y < H; y += 3) { ctx.fillStyle = "rgba(0,0,0,0.06)"; ctx.fillRect(0, y, W, 1); }

    if (allP.length === 0) {
      ctx.fillStyle = "#1a2a3a"; ctx.font = "13px 'Courier New'"; ctx.textAlign = "center";
      ctx.fillText("جاري الاتصال بـ Binance...", W / 2, H / 2 - 10);
      ctx.fillStyle = "#0e1a26"; ctx.font = "10px 'Courier New'";
      ctx.fillText("بيانات مباشرة ومجانية", W / 2, H / 2 + 12);
      ctx.restore(); return;
    }

    let rawMin = Infinity, rawMax = -Infinity;
    for (let i = 0; i < allP.length; i++) {
      if (allP[i] < rawMin) rawMin = allP[i];
      if (allP[i] > rawMax) rawMax = allP[i];
    }
    const mid  = (rawMin + rawMax) / 2 + yPanOffset.current;
    const halfRange = Math.max((rawMax - rawMin) * 0.15, rawMin * 0.001) + (rawMax - rawMin) / 2;
    // yZoom: >1 = zoom out (more range), <1 = zoom in (less range)
    const zoomedHalf = halfRange / yZoom.current;
    const minP = mid - zoomedHalf, maxP = mid + zoomedHalf, pRange = maxP - minP;
    const tx = t => PL + ((t - timeStart) / timeRange) * CW;
    const ty = p => PT + CH - ((p - minP) / pRange) * CH;
    const nowX = tx(now);
    const totalView = xViewMin.current + TRAJ_MIN; // total visible minutes

    // Grid — iterate over visible time range
    ctx.lineWidth = 1;
    const gridStep = totalView <= 30 ? 2 : totalView <= 120 ? 10 : totalView <= 360 ? 30 : 60;
    const gridStepMs = gridStep * 60000;
    {
      const firstGrid = Math.floor(timeStart / gridStepMs) * gridStepMs;
      for (let t = firstGrid; t <= timeEnd; t += gridStepMs) {
        const x = tx(t);
        if (x < PL - 1 || x > PL + CW + 1) continue;
        ctx.strokeStyle = "rgba(255,255,255,0.024)";
        ctx.beginPath(); ctx.moveTo(x, PT); ctx.lineTo(x, PT + CH); ctx.stroke();
      }
    }
    // Y grid lines are drawn with Y labels below

    ctx.fillStyle = "rgba(0,0,0,0.14)";
    const clampedNowX = Math.max(PL, Math.min(PL + CW, nowX));
    ctx.fillRect(PL, PT, Math.max(0, clampedNowX - PL), CH);

    ctx.setLineDash([3, 5]); ctx.strokeStyle = "rgba(80,150,255,0.18)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(nowX, PT); ctx.lineTo(nowX, PT + CH); ctx.stroke();
    ctx.setLineDash([]);

    ctx.font = "9px 'Courier New'"; ctx.fillStyle = "rgba(60,90,130,0.4)"; ctx.textAlign = "center";
    if (nowX - PL > 55) ctx.fillText("السابق", PL + (nowX - PL) / 2, PT + 13);
    if (PL + CW - nowX > 55) ctx.fillText("التوقع", nowX + (PL + CW - nowX) / 2, PT + 13);

    // Cloud
    cloudTrajs.current.forEach(traj => {
      const a = Math.max(0.01, 0.11 - (now - traj.t0) / 1000 / 700);
      ctx.strokeStyle = `rgba(80,140,255,${a})`; ctx.lineWidth = 0.8;
      ctx.beginPath(); let s = false;
      traj.pts.forEach(p => {
        if (p.time < timeStart) return;
        const x = tx(p.time), y = ty(p.price);
        s ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), s = true);
      });
      ctx.stroke();
    });

    // Previous trajectories
    curTrajs.current.slice(0, -1).forEach((traj, idx, arr) => {
      const a  = 0.07 + (idx / Math.max(arr.length, 1)) * 0.2;
      const up = traj.pts.at(-1).price >= traj.pts[0].price;
      ctx.strokeStyle = up ? `rgba(0,210,120,${a})` : `rgba(255,55,95,${a})`; ctx.lineWidth = 1;
      ctx.beginPath(); let s = false;
      traj.pts.forEach(p => { const x = tx(p.time), y = ty(p.price); s ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), s = true); });
      ctx.stroke();
    });

    // Latest trajectory
    const latest = curTrajs.current.at(-1);
    if (latest) {
      const up = latest.pts.at(-1).price >= latest.pts[0].price;
      const cv = up ? "0,255,140" : "255,50,90";
      [[8,0.05],[4,0.11],[2,0.22]].forEach(([lw,a]) => {
        ctx.strokeStyle = `rgba(${cv},${a})`; ctx.lineWidth = lw;
        ctx.beginPath(); let s = false;
        latest.pts.forEach(p => { const x = tx(p.time), y = ty(p.price); s ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), s = true); });
        ctx.stroke();
      });
      const endX = tx(latest.pts.at(-1).time);
      const lg = ctx.createLinearGradient(nowX, 0, endX, 0);
      lg.addColorStop(0, `rgba(${cv},0.95)`); lg.addColorStop(0.45, `rgba(${cv},0.65)`); lg.addColorStop(1, `rgba(${cv},0.07)`);
      ctx.strokeStyle = lg; ctx.lineWidth = 2; ctx.shadowColor = `rgb(${cv})`; ctx.shadowBlur = 12;
      ctx.beginPath(); let s = false;
      latest.pts.forEach(p => { const x = tx(p.time), y = ty(p.price); s ? ctx.lineTo(x, y) : (ctx.moveTo(x, y), s = true); });
      ctx.stroke(); ctx.shadowBlur = 0;
      const ep = latest.pts.at(-1);
      const epX = Math.min(tx(ep.time), PL + CW - 60); // clamp inside canvas
      const epY = ty(ep.price);
      ctx.fillStyle = `rgba(${cv},0.75)`; ctx.beginPath(); ctx.arc(tx(ep.time), epY, 3, 0, Math.PI * 2); ctx.fill();
      ctx.font = "10px 'Courier New'"; ctx.fillStyle = `rgba(${cv},0.9)`; ctx.textAlign = "left";
      ctx.fillText(`$${ep.price.toLocaleString("en",{maximumFractionDigits:0})}`, epX + 6, epY + 4);
    }

    // Price history — break line across gaps
    if (priceHist.current.length > 1) {
      ctx.strokeStyle = "rgba(220,235,255,0.92)"; ctx.lineWidth = 1.8;
      ctx.beginPath();
      let started = false;
      priceHist.current.forEach((p, i) => {
        const x = tx(p.time), y = ty(p.price);
        if (i === 0) { ctx.moveTo(x, y); started = true; return; }
        const prevP = priceHist.current[i - 1];
        const timeGap = p.time - prevP.time;
        const priceJump = Math.abs(p.price - prevP.price) / prevP.price;
        if (timeGap > 120000 || priceJump > 0.003) {
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    // Pulsing dot
    const cp = priceHist.current.at(-1)?.price;
    if (cp !== undefined) {
      const cy    = ty(cp);
      const pulse = 0.4 + 0.6 * (Math.sin(pulseT.current) * 0.5 + 0.5);
      ctx.fillStyle = `rgba(255,255,255,${0.1 * pulse})`;
      ctx.beginPath(); ctx.arc(nowX, cy, 3 + pulse * 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.shadowColor = "#88ccff"; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(nowX, cy, 3.5, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
      ctx.font = "bold 13px 'Courier New'"; ctx.fillStyle = "#ddeeff"; ctx.textAlign = "left";
      ctx.fillText(`$${cp.toLocaleString("en")}`, nowX + 10, cy - 7);
    }

    // Y labels — nice intervals that move with pan/zoom (like X-axis)
    ctx.font = "10px 'Courier New'"; ctx.fillStyle = "rgba(65,95,135,0.75)"; ctx.textAlign = "right";
    {
      // Calculate a "nice" step for price labels
      const targetLabels = 6;
      const rawStep = pRange / targetLabels;
      const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const niceSteps = [1, 2, 5, 10, 20, 50];
      let priceStep = mag;
      for (const ns of niceSteps) {
        if (mag * ns >= rawStep) { priceStep = mag * ns; break; }
      }
      const startP = Math.ceil(minP / priceStep) * priceStep;
      for (let p = startP; p <= maxP; p += priceStep) {
        const y = ty(p);
        if (y < PT - 5 || y > PT + CH + 5) continue;
        ctx.fillText(`$${safeNum(p).toLocaleString("en", {maximumFractionDigits: 0})}`, PL - 6, y + 3);
        // Draw grid line at this price
        ctx.strokeStyle = "rgba(255,255,255,0.024)";
        ctx.beginPath(); ctx.moveTo(PL, y); ctx.lineTo(PL + CW, y); ctx.stroke();
      }
    }

    // X labels — iterate over visible time range with nice intervals
    ctx.textAlign = "center";
    let step = totalView <= 30 ? 2 : totalView <= 120 ? 10 : totalView <= 360 ? 30 : 60;
    const pixelsPerMin = CW / totalView;
    if (pixelsPerMin * step < 45) step = Math.ceil(45 / pixelsPerMin);
    const stepMs = step * 60000;
    {
      const firstLabel = Math.floor(timeStart / stepMs) * stepMs;
      for (let t = firstLabel; t <= timeEnd; t += stepMs) {
        const x = tx(t);
        if (x < PL || x > PL + CW) continue;
        const m = (t - now) / 60000; // minutes from now
        const isNow = Math.abs(m) < step * 0.3;
        ctx.fillStyle = isNow ? "rgba(100,175,255,0.95)" : "rgba(65,95,135,0.65)";
        ctx.font = isNow ? "bold 10px 'Courier New'" : "10px 'Courier New'";
        let label;
        if (isNow) {
          label = "الآن";
        } else if (Math.abs(m) >= 60) {
          const h = Math.floor(Math.abs(m) / 60);
          const min2 = Math.round(Math.abs(m) % 60);
          label = `${m < 0 ? "-" : "+"}${h}س${min2 > 0 ? min2 + "د" : ""}`;
        } else {
          label = `${m > 0 ? "+" : ""}${Math.round(m)}د`;
        }
        ctx.fillText(label, x, PT + CH + 22);
      }
    }

    // ── Crosshair indicator ──
    if (mousePos.current) {
      const mx = mousePos.current.x, my = mousePos.current.y;
      if (mx >= PL && mx <= PL + CW && my >= PT && my <= PT + CH) {
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "rgba(120,170,230,0.35)";
        ctx.lineWidth = 0.7;
        // Horizontal line
        ctx.beginPath(); ctx.moveTo(PL, my); ctx.lineTo(PL + CW, my); ctx.stroke();
        // Vertical line
        ctx.beginPath(); ctx.moveTo(mx, PT); ctx.lineTo(mx, PT + CH); ctx.stroke();
        ctx.setLineDash([]);
        // Price label on Y axis
        const cursorPrice = maxP - ((my - PT) / CH) * pRange;
        ctx.fillStyle = "rgba(30,55,90,0.9)";
        ctx.fillRect(0, my - 9, PL - 2, 18);
        ctx.font = "10px 'Courier New'"; ctx.fillStyle = "#88bbff"; ctx.textAlign = "right";
        ctx.fillText(`$${safeNum(cursorPrice).toLocaleString("en", {maximumFractionDigits:0})}`, PL - 6, my + 4);
        // Time label on X axis
        const cursorTime = timeStart + ((mx - PL) / CW) * timeRange;
        const diffMin = (cursorTime - now) / 60000;
        let timeLabel;
        if (Math.abs(diffMin) < 1) {
          timeLabel = "الآن";
        } else if (Math.abs(diffMin) >= 60) {
          const h = Math.floor(Math.abs(diffMin) / 60);
          const min2 = Math.round(Math.abs(diffMin) % 60);
          timeLabel = `${diffMin < 0 ? "-" : "+"}${h}س${min2 > 0 ? min2 + "د" : ""}`;
        } else {
          timeLabel = `${diffMin > 0 ? "+" : ""}${Math.round(diffMin)}د`;
        }
        const labelW = ctx.measureText(timeLabel).width + 12;
        ctx.fillStyle = "rgba(30,55,90,0.9)";
        ctx.fillRect(mx - labelW / 2, PT + CH + 2, labelW, 18);
        ctx.font = "10px 'Courier New'"; ctx.fillStyle = "#88bbff"; ctx.textAlign = "center";
        ctx.fillText(timeLabel, mx, PT + CH + 14);
      }
    }

    ctx.restore();
  }, []);

  useEffect(() => {
    const loop = () => { draw(); animId.current = requestAnimationFrame(loop); };
    animId.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId.current);
  }, [draw]);

  const { price, change24h, bidPct, askPct, connected, error, loaded, acc60, accAll } = info;
  const ch24up   = change24h >= 0;
  const isUp     = bidPct >= askPct;
  const dotColor = error ? "#ff5566" : connected ? "#00ff88" : "#ffcc00";
  const statusText = error ? error : connected
    ? `متصل · بيانات مباشرة · ${loaded ? "تم تحميل السجل ✓" : "جاري تحميل السجل..."}`
    : "جاري الاتصال...";

  return (
    <div style={{ background:"#040810", height:"100vh", overflow:"hidden", color:"#c0d0e0", fontFamily:"'Courier New',Courier,monospace", direction:"rtl", display:"flex", flexDirection:"column", alignItems:"center", padding:"14px 10px", gap:8, boxSizing:"border-box" }}>

      <div style={{ width:"100%", maxWidth:880, display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:4 }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ width:8, height:8, borderRadius:"50%", background:dotColor, boxShadow:`0 0 8px ${dotColor}`, animation:"blink 2s infinite" }} />
          <span style={{ fontSize:"min(10px, 2vw)", color:"#334455" }}>{statusText}</span>
        </div>
        <span style={{ fontSize:"min(12px, 2.5vw)", color:"#1a2a3a", letterSpacing:3 }}>BTC/USDT · TRAJECTORY</span>
      </div>

      <div style={{ width:"100%", maxWidth:880, display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(130px, 1fr))", gap:8 }}>
        {[
          { label:"السعر الحالي",      value: price ? `$${safeNum(price).toLocaleString("en",{maximumFractionDigits:0})}` : "---", color:"#c0d8ff" },
          { label:"تغيّر 24 ساعة",     value: price ? `${ch24up?"▲":"▼"} ${Math.abs(safeNum(change24h)).toFixed(2)}%` : "---", color:ch24up?"#00ff88":"#ff4466" },
          { label:"زخم الشراء",        value: price ? `${safeNum(bidPct).toFixed(0)}%` : "---", color:"#00dd77" },
          { label:"زخم البيع",         value: price ? `${safeNum(askPct).toFixed(0)}%` : "---", color:"#ff5577" },
          { label:"دقة التوقع 60د",    value: acc60  != null ? `${acc60}%`  : "---", color: acc60  >= 60 ? "#00ff88" : acc60  >= 50 ? "#ffcc00" : "#ff4466" },
          { label:"دقة التوقع الكلية", value: accAll != null ? `${accAll}%` : "---", color: accAll >= 60 ? "#00ff88" : accAll >= 50 ? "#ffcc00" : "#ff4466" },
        ].map(({label,value,color}) => (
          <div key={label} style={{ background:"rgba(255,255,255,0.025)", border:"1px solid rgba(255,255,255,0.05)", borderRadius:6, padding:"8px 10px", textAlign:"center" }}>
            <div style={{ fontSize:9, color:"#445566", marginBottom:3 }}>{label}</div>
            <div style={{ fontSize:15, fontWeight:"bold", color }}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{ width:"100%", maxWidth:880 }}>
        <div style={{ display:"flex", height:4, borderRadius:2, overflow:"hidden" }}>
          <div style={{ flex:safeNum(bidPct,50), background:"linear-gradient(90deg,#002a14,#00aa55)", transition:"flex 0.8s ease" }} />
          <div style={{ flex:safeNum(askPct,50), background:"linear-gradient(90deg,#aa2233,#3a0008)", transition:"flex 0.8s ease" }} />
        </div>
        <div style={{ display:"flex", justifyContent:"space-between", fontSize:9, color:"#334455", marginTop:3 }}>
          <span style={{color:"#009944"}}>ضغط شراء {isUp?"▲":""}</span>
          <span style={{color:"#993344"}}>ضغط بيع {!isUp?"▼":""}</span>
        </div>
      </div>

      {info.obi !== undefined && (
        <div style={{ width:"100%", maxWidth:880 }}>
          <div style={{ display:"flex", justifyContent:"space-between", fontSize:9, color:"#334455", marginBottom:3 }}>
            <span style={{color: info.obi > 0 ? "#009944" : "#993344"}}>
              OBI: {info.obi > 0 ? "ضغط شراء" : "ضغط بيع"} ({(Math.abs(info.obi)*100).toFixed(0)}%)
            </span>
            {info.rsi > 0 && <span style={{color: info.rsi > 70 ? "#ff4466" : info.rsi < 30 ? "#00ff88" : "#556677"}}>RSI: {info.rsi.toFixed(0)}</span>}
          </div>
          <div style={{ display:"flex", height:3, borderRadius:2, overflow:"hidden", background:"rgba(255,255,255,0.03)" }}>
            <div style={{ width:`${Math.max(2, (0.5 + info.obi/2) * 100)}%`, background: info.obi > 0 ? "linear-gradient(90deg,#004422,#00aa55)" : "linear-gradient(90deg,#aa2233,#440011)", transition:"width 0.8s ease" }} />
          </div>
        </div>
      )}
      {info.indicatorsReady === false && info.obi !== undefined && (
        <div style={{ width:"100%", maxWidth:880, fontSize:9, color:"#445566", textAlign:"center", padding:2 }}>
          ⏳ المؤشرات الفنية قيد التحميل — ستعمل خلال دقائق
        </div>
      )}

      <div style={{ width:"100%", maxWidth:880, position:"relative", flex:1, minHeight:0, display:"flex", flexDirection:"column" }}>
        <canvas ref={canvasRef} style={{ width:"100%", flex:1, minHeight:0, borderRadius:8, border:"1px solid rgba(255,255,255,0.04)", display:"block", cursor:"crosshair", touchAction:"none", userSelect:"none" }} />
        {showFollow && (
          <button onClick={() => {
            autoFollow.current = true;
            xViewMin.current = 60;
            xPanOffset.current = 0;
            yPanOffset.current = 0;
            yZoom.current = 1;
            setShowFollow(false);
            saveChartState();
          }} style={{
            position:"absolute", bottom:14, right:14,
            background:"rgba(80,150,255,0.15)",
            border:"1px solid rgba(80,150,255,0.4)",
            borderRadius:"50%", color:"#88bbff",
            fontSize:16, width:32, height:32,
            display:"flex", alignItems:"center", justifyContent:"center",
            cursor:"pointer", backdropFilter:"blur(4px)",
            transition:"opacity 0.3s",
          }}>
            ◎
          </button>
        )}
        {info.predDirection && (
          <div style={{
            position:"absolute", top:8, right:8,
            background:"rgba(8,12,24,0.85)",
            border:"1px solid rgba(255,255,255,0.08)",
            borderRadius:8, padding:"8px 14px",
            backdropFilter:"blur(6px)",
            fontFamily:"'Courier New',monospace",
            fontSize:11, lineHeight:1.6,
            display:"flex", flexDirection:"column", alignItems:"flex-end", gap:2,
          }}>
            <div style={{display:"flex",alignItems:"center",gap:6}}>
              <span style={{color:"#556677",fontSize:9}}>التوقع</span>
              <span style={{
                color: info.predDirection === "up" ? "#00ff88" : info.predDirection === "down" ? "#ff4466" : "#556677",
                fontWeight:"bold", fontSize:14,
              }}>
                {info.predDirection === "up" ? "▲ صعود" : info.predDirection === "down" ? "▼ هبوط" : "— محايد"}
              </span>
            </div>
            <div style={{display:"flex",alignItems:"center",gap:6}}>
              <span style={{color:"#556677",fontSize:9}}>الثقة</span>
              <span style={{
                color: info.predConfidence > 0.4 ? "#00ff88" : info.predConfidence > 0.2 ? "#ffcc00" : "#556677",
                fontWeight:"bold",
              }}>
                {(info.predConfidence * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{fontSize:9,color:"#334455"}}>
              {info.predMethod === "ml" ? "🧠 ML" : "📊 قواعد"}
            </div>
          </div>
        )}
      </div>
      <div style={{ fontSize:9, color:"#1a2a3a", textAlign:"center" }}>
        سحب على الرسم: تحريك · عجلة الماوس: تكبير/تصغير · سحب محور السعر: زوم السعر · دبل كليك: إعادة
      </div>

      <div style={{ display:"flex", gap:16, fontSize:10, color:"#445566", flexWrap:"wrap", justifyContent:"center" }}>
        {[["#c0d8ff","السعر"],["#00ff88","توقع ↑"],["#ff4466","توقع ↓"],["rgba(80,140,255,0.7)","سحابة"]].map(([c,l])=>(
          <span key={l} style={{display:"flex",alignItems:"center",gap:4}}>
            <span style={{color:c,fontSize:14}}>─</span>
            <span>{l}</span>
          </span>
        ))}
        <span style={{color:"#223344",marginRight:8}}>· سحب: تحريك · عجلة: زوم · دبل كليك: إعادة</span>
      </div>

      <style>{`@keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
    </div>
  );
}
