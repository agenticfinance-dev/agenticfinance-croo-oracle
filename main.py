import os
import time
import requests
import numpy as np
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

app = FastAPI()

# ===== CONFIG =====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0"))
CHAT_ID = os.environ.get("CHAT_ID")

PAYMENTS_ENABLED = False # Keep false for CROO judging
PAYMENT_PROVIDER_TOKEN = ""

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()

# 1. MULTI-ASSET SUPPORT - CROO REQUIREMENT
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "CROUSDT"]

# ===== STATE =====
cache = {"signals": {}, "last_scan": 0}
signal_history = []
users_db = {}
last_alerted = {}

# 4. PERFORMANCE TRACKING
performance = {"wins": 0, "losses": 0, "total": 0}

# ===== DATA HELPERS =====
def get_binance_klines(symbol, interval="1h", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit},
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def get_mexc_klines(symbol, interval="1h", limit=100):
    try:
        url = "https://api.mexc.com/api/v3/klines"
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit},
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def get_coingecko_price(asset):
    mapping = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
               "XRPUSDT": "ripple", "CROUSDT": "crypto-com-chain"}
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
            params={"ids": mapping[asset], "vs_currencies": "usd"}, timeout=8)
        if r.status_code == 200: return float(r.json()[mapping[asset]]["usd"])
    except: pass
    return None

def get_current_price(symbol):
    klines = get_binance_klines(symbol, "1h", 1)
    if klines: return float(klines[-1][4])
    klines = get_mexc_klines(symbol, "1h", 1)
    if klines: return float(klines[-1][4])
    return get_coingecko_price(symbol) or 0

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return np.array([50.0] * len(closes))
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down!= 0 else 0
    rsi = np.zeros_like(closes)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(closes)):
        delta = deltas[i - 1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down!= 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calc_ema(closes, period):
    if len(closes) < period: return np.array([closes[-1]] if len(closes) > 0 else [0])
    return np.convolve(closes, np.ones(period)/period, mode='valid')

# ===== USER MGMT =====
def is_pro(user_id: int) -> bool:
    user = users_db.get(user_id, {})
    if not user: return False
    if user.get("plan") == "lifetime": return True
    expires = user.get("pro_expires")
    return expires and datetime.now() < expires

def activate_pro(user_id: int, days: int = 30):
    if user_id not in users_db: users_db[user_id] = {}
    users_db[user_id]["plan"] = "pro"
    users_db[user_id]["pro_expires"] = datetime.now() + timedelta(days=days)

# 6. MULTI-TIMEFRAME ANALYSIS
def analyze_timeframe(symbol, interval):
    klines = get_binance_klines(symbol, interval)
    source = "binance"
    if not klines:
        klines = get_mexc_klines(symbol, interval)
        source = "mexc"
    if not klines or len(klines) < 50:
        return {"confidence": 0, "price": 0, "source": "none"}

    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    price = closes[-1]
    rsi = calc_rsi(closes)[-1]
    ema20 = calc_ema(closes, 20)[-1] if len(calc_ema(closes, 20)) > 0 else price
    ema50 = calc_ema(closes, 50)[-1] if len(calc_ema(closes, 50)) > 0 else price

    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False

    long_score = 0
    if rsi < 45: long_score += 30
    if price > ema50: long_score += 25
    if 2 < pullback < 12: long_score += 25
    if vol_spike: long_score += 20

    short_score = 0
    if rsi > 55: short_score += 25
    if price < ema50: short_score += 25
    if 2 < bounce < 12: short_score += 20
    rally_to_resistance = ((ema20 - price) / price * 100) if price < ema20 else 0
    if 0 < rally_to_resistance < 3: short_score += 15
    if vol_spike: short_score += 15

    confidence = max(long_score, short_score)
    direction = "LONG" if long_score >= short_score else "SHORT"
    return {"confidence": confidence, "price": price, "rsi": rsi, "ema20": ema20,
            "ema50": ema50, "pullback": pullback, "bounce": bounce, "source": source,
            "direction": direction, "vol_spike": vol_spike}

def analyze_asset(symbol):
    tf_15m = analyze_timeframe(symbol, "15m")
    tf_1h = analyze_timeframe(symbol, "1h")
    tf_4h = analyze_timeframe(symbol, "4h")

    if tf_1h["price"] == 0:
        return {"asset": symbol.replace("USDT", ""), "signal": "NONE", "confidence": 0}

    # Weighted: 4h=50%, 1h=30%, 15m=20%
    confidence = int(tf_15m["confidence"] * 0.2 + tf_1h["confidence"] * 0.3 + tf_4h["confidence"] * 0.5)
    direction = tf_1h["direction"]
    price = tf_1h["price"]
    rsi = tf_1h["rsi"]

    reasons = []
    if rsi < 45 and direction == "LONG": reasons.append("✅ RSI Oversold")
    if rsi > 55 and direction == "SHORT": reasons.append("✅ RSI Overbought")
    if price > tf_1h["ema50"] and direction == "LONG": reasons.append("✅ Above EMA50")
    if price < tf_1h["ema50"] and direction == "SHORT": reasons.append("✅ Below EMA50")
    if 2 < tf_1h["pullback"] < 12 and direction == "LONG": reasons.append(f"✅ Dip {tf_1h['pullback']:.1f}%")
    if 2 < tf_1h["bounce"] < 12 and direction == "SHORT": reasons.append(f"✅ Bounce {tf_1h['bounce']:.1f}%")
    if tf_1h["vol_spike"]: reasons.append("✅ Volume Spike")

    signal = "NONE"
    if confidence >= 75: signal = "BUY" if direction == "LONG" else "SHORT"
    elif confidence >= 50: signal = "WATCH"

    stop_loss = round(price * 0.95, 4) if signal == "BUY" else round(price * 1.05, 4) if signal == "SHORT" else 0
    take_profit = round(price * 1.10, 4) if signal == "BUY" else round(price * 0.90, 4) if signal == "SHORT" else 0

    return {
        "asset": symbol.replace("USDT", ""), "price": round(price, 4), "signal": signal,
        "confidence": confidence, "direction": direction, "entry": round(price, 4) if signal in ["BUY", "SHORT"] else 0,
        "stop_loss": stop_loss, "take_profit": take_profit, "rsi": round(rsi, 1), "reasons": reasons,
        "timeframes": {"15m": tf_15m["confidence"], "1h": tf_1h["confidence"], "4h": tf_4h["confidence"]},
        "source": tf_1h["source"], "timestamp": datetime.utcnow().isoformat()
    }

def check_closed_signals():
    for signal in signal_history:
        if signal.get("status") == "open" and signal.get("entry", 0) > 0:
            current_price = get_current_price(signal["asset"] + "USDT")
            if current_price == 0: continue
            if signal["direction"] == "LONG":
                if current_price >= signal["take_profit"]:
                    signal["pnl"] = round((signal["take_profit"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"; performance["wins"] += 1; performance["total"] += 1
                elif current_price <= signal["stop_loss"]:
                    signal["pnl"] = round((signal["stop_loss"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"; performance["losses"] += 1; performance["total"] += 1
            elif signal["direction"] == "SHORT":
                if current_price <= signal["take_profit"]:
                    signal["pnl"] = round((signal["entry"] - signal["take_profit"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"; performance["wins"] += 1; performance["total"] += 1
                elif current_price >= signal["stop_loss"]:
                    signal["pnl"] = round((signal["entry"] - signal["stop_loss"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"; performance["losses"] += 1; performance["total"] += 1

# 7. AUTO TELEGRAM ALERTS
async def send_alert(signal):
    if not bot or signal["confidence"] < 75: return
    if signal["asset"] in last_alerted and time.time() - last_alerted[signal["asset"]] < 3600: return

    msg = f"🚨 NEW {signal['signal']} SIGNAL\n\n"
    msg += f"Asset: {signal['asset']}\nConfidence: {signal['confidence']}%\n\n"
    msg += f"Entry:\n{signal['entry']}\n\nTarget:\n{signal['take_profit']}\n\n"
    msg += f"Stop:\n{signal['stop_loss']}\n\nReasons:\n" + "\n".join(signal['reasons'])

    if CHAT_ID:
        try: await bot.send_message(chat_id=CHAT_ID, text=msg)
        except: pass
    last_alerted[signal["asset"]] = time.time()

def run_scanner():
    check_closed_signals()
    results = {}
    for asset in ASSETS:
        data = analyze_asset(asset)
        if data and data["signal"]!= "NONE":
            results[asset] = data
            if data["signal"] in ["BUY", "SHORT"]:
                data["status"] = "open"
                signal_history.append(data)
                asyncio.create_task(send_alert(data))
    signal_history[:] = signal_history[-100:]
    cache["signals"] = results
    cache["last_scan"] = time.time()
    return results

# 8. SCAN SCHEDULER - Every 5 minutes
async def scanner_loop():
    while True:
        run_scanner()
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await bot.set_webhook(url=webhook_url)
    asyncio.create_task(scanner_loop())

# ===== TELEGRAM =====
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        user_id = data["message"]["from"]["id"]
        await handle_message(chat_id, text, user_id)
    elif "callback_query" in data:
        query = data["callback_query"]
        await handle_callback(query["message"]["chat"]["id"], query["data"], query["from"]["id"])
    return JSONResponse({"ok": True})

async def handle_message(chat_id, text, user_id):
    if not bot: return
    if text == "/start":
        # 13. BETTER MENU
        keyboard = [
            [InlineKeyboardButton("📊 Scan Markets", callback_data="scan_all"),
             InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard")],
            [InlineKeyboardButton("🔍 Best Signal", callback_data="best_signal")],
            [InlineKeyboardButton("📈 BTC", callback_data="BTCUSDT"),
             InlineKeyboardButton("📈 ETH", callback_data="ETHUSDT"),
             InlineKeyboardButton("📈 SOL", callback_data="SOLUSDT")],
            [InlineKeyboardButton("📈 XRP", callback_data="XRPUSDT"),
             InlineKeyboardButton("📈 CRO", callback_data="CROUSDT")],
            [InlineKeyboardButton("💎 Upgrade", callback_data="buy_cmd")]
        ]
        msg = "🔮 CROO AI Oracle - Market Intelligence\n\n"
        msg += "Autonomous scanning every 5 min\n"
        msg += "Assets: BTC, ETH, SOL, XRP, CRO\n"
        msg += "Demo: All features FREE for judges\n\n"
        msg += "/scan /best /leaderboard /buy /sell"
        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard))

    elif text in ["/scan", "/signals"]:
        run_scanner(); await send_leaderboard(chat_id)
    elif text == "/best":
        run_scanner()
        signals = [s for s in cache["signals"].values() if s["signal"] in ["BUY", "SHORT"]]
        if not signals:
            await bot.send_message(chat_id=chat_id, text="No high-confidence signals right now.")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x["confidence"]))
    elif text == "/leaderboard": await send_leaderboard(chat_id)
    elif text == "/buy": await handle_buy(chat_id, user_id)
    elif text == "/sell": await handle_sell(chat_id, user_id)

# 14. RICH SIGNAL CARDS
async def send_rich_card(chat_id, s):
    msg = f"🚨 {s['signal']} SIGNAL\n\n"
    msg += f"Asset: {s['asset']}\nConfidence: {s['confidence']}%\n\n"
    msg += f"Entry:\n{s['entry']}\n\nTarget:\n{s['take_profit']}\n\nStop:\n{s['stop_loss']}\n\n"
    msg += f"Reasons:\n" + "\n".join(s['reasons'])
    msg += f"\n\nTimeframes: 15m:{s['timeframes']['15m']} 1h:{s['timeframes']['1h']} 4h:{s['timeframes']['4h']}"
    await bot.send_message(chat_id=chat_id, text=msg)

async def send_leaderboard(chat_id):
    run_scanner()
    signals = sorted(cache["signals"].values(), key=lambda x: x["confidence"], reverse=True)
    msg = "🏆 LEADERBOARD\n\n"
    for i, s in enumerate(signals[:5], 1):
        msg += f"{i}. {s['asset']} - {s['confidence']}% {s['signal']}\n"
    await bot.send_message(chat_id=chat_id, text=msg)

async def handle_buy(chat_id, user_id):
    if PAYMENTS_ENABLED:
        await bot.send_message(chat_id=chat_id, text="Payments enable post-hackathon...")
    else:
        if is_pro(user_id):
            await bot.send_message(chat_id=chat_id, text="You're already Pro ✅")
        else:
            activate_pro(user_id, 999)
            await bot.send_message(chat_id=chat_id, text="✅ DEMO: Pro activated\n\nAll signals unlocked for Croo judging.\nMonetization: Post-launch via Telegram Payments")

async def handle_sell(chat_id, user_id):
    if not is_pro(user_id):
        await bot.send_message(chat_id=chat_id, text="You're on Free plan.")
    else:
        users_db[user_id]["plan"] = "free"
        users_db[user_id]["pro_expires"] = None
        await bot.send_message(chat_id=chat_id, text="✅ DEMO: Pro cancelled\n\nIn production: Cancels recurring billing.")

async def handle_callback(chat_id, data, user_id):
    if data == "scan_all": run_scanner(); await send_leaderboard(chat_id)
    elif data == "best_signal":
        run_scanner()
        signals = [s for s in cache["signals"].values() if s["signal"] in ["BUY", "SHORT"]]
        if signals: await send_rich_card(chat_id, max(signals, key=lambda x: x["confidence"]))
    elif data == "leaderboard": await send_leaderboard(chat_id)
    elif data == "buy_cmd": await handle_buy(chat_id, user_id)
    elif data == "sell_cmd": await handle_sell(chat_id, user_id)
    elif data in ASSETS:
        run_scanner()
        s = cache["signals"].get(data)
        if s: await send_rich_card(chat_id, s)

# ===== API ENDPOINTS =====
@app.get("/")
def root():
    return {"status": "CROO AI Oracle Online", "mode": "hackathon", "payments": PAYMENTS_ENABLED}

@app.get("/oracle")
def oracle():
    run_scanner()
    return cache["signals"]

# 2. BEST SIGNAL ENDPOINT
@app.get("/best_signal")
def best_signal():
    run_scanner()
    signals = [s for s in cache["signals"].values() if s["signal"] in ["BUY", "SHORT"]]
    if not signals: return {"asset": "NONE", "signal": "NONE", "confidence": 0}
    best = max(signals, key=lambda x: x["confidence"])
    return {"asset": best["asset"], "signal": best["signal"], "confidence": best["confidence"]}

# 3. LEADERBOARD ENDPOINT
@app.get("/leaderboard")
def leaderboard():
    run_scanner()
    signals = sorted(cache["signals"].values(), key=lambda x: x["confidence"], reverse=True)
    return [{"asset": s["asset"], "confidence": s["confidence"]} for s in signals]

# 4. PERFORMANCE TRACKING
@app.get("/performance")
def get_performance():
    check_closed_signals()
    win_rate = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0
    return {"total_signals": performance["total"], "wins": performance["wins"],
            "losses": performance["losses"], "win_rate": f"{win_rate}%"}

# 5. AGENT QUERY ENDPOINT - CROO A2A
@app.post("/agent/query")
async def agent_query(request: Request):
    data = await request.json()
    task = data.get("task", "")
    run_scanner()
    if task == "find_best_pullback":
        signals = [s for s in cache["signals"].values() if "Dip" in str(s.get("reasons", []))]
        if not signals: return {"error": "No pullback found"}
        best = max(signals, key=lambda x: x["confidence"])
        return {"asset": best["asset"], "confidence": best["confidence"], "signal": best["signal"]}
    return best_signal()

# 9. SIGNAL HISTORY
@app.get("/history")
def history():
    return signal_history[-50:]

# 10. CAP METADATA
@app.get("/cap/metadata")
def cap_metadata():
    return {"agent": "CROO AI Oracle", "version": "9.0", "category": "Market Intelligence",
            "callable": True, "supports": ["BTC", "ETH", "SOL", "XRP", "CRO"]}

# 11. AGENT PRICING
@app.get("/pricing")
def pricing():
    return {"free": "5 requests/day", "pro": "Unlimited", "enterprise": "API Access"}

# 12. AGENT CAPABILITIES
@app.get("/capabilities")
def capabilities():
    return {"features": ["pullback_detection", "bounce_detection", "rally_to_resistance",
            "multi_timeframe_analysis", "confidence_scoring", "market_intelligence",
            "signal_ranking", "auto_scanning"]}

@app.get("/cap/health")
def cap_health():
    return {"agent": "CROO AI Oracle", "status": "active", "assets": ASSETS,
            "data_sources": ["Binance", "MEXC", "CoinGecko"], "uptime_seconds": int(time.time() - start_time)}
