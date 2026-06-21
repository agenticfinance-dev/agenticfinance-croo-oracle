import os
import time
import requests
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

app = FastAPI()

# ===== CONFIG =====
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
CHAT_ID = os.environ.get("CHAT_ID") # Your ID for ALPHA alerts
PREMIUM_CHAT_IDS = os.environ.get("PREMIUM_CHAT_IDS", "").split(",") # Paid users: "123,456,789"

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()

ASSETS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT",
    "DOGEUSDT","ADAUSDT","TRXUSDT","LINKUSDT","AVAXUSDT"
]

# ===== STATE =====
cache = {"signals": {}, "last_scan": 0}
signal_history = []
last_alerted = {}

# ===== HELPERS =====
def get_binance_klines(symbol, interval="1h", limit=100):
    try:
        url = "https://api.binance.com/api/v3/klines"
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Binance failed for {symbol}: {e}")
    return None

def get_mexc_klines(symbol, interval="1h", limit=100):
    try:
        url = "https://api.mexc.com/api/v3/klines"
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"MEXC failed for {symbol}: {e}")
    return None

def get_coingecko_price(asset):
    mapping = {
        "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "XRPUSDT": "ripple",
        "SOLUSDT": "solana", "DOGEUSDT": "dogecoin", "ADAUSDT": "cardano",
        "TRXUSDT": "tron", "LINKUSDT": "chainlink", "AVAXUSDT": "avalanche-2"
    }
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": mapping[asset], "vs_currencies": "usd"},
            timeout=8
        )
        if r.status_code == 200:
            return float(r.json()[mapping[asset]]["usd"])
    except Exception as e:
        print(f"CoinGecko failed for {asset}: {e}")
    return None

def get_current_price(symbol):
    klines = get_binance_klines(symbol, "1h", 1)
    if klines:
        return float(klines[-1][4])
    price = get_coingecko_price(symbol)
    return price if price else 0

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return np.array([50.0] * len(closes))
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
    if len(closes) < period:
        return np.array([closes[-1]] if len(closes) > 0 else [0])
    return np.convolve(closes, np.ones(period)/period, mode='valid')

def analyze_asset(symbol, timeframe="1h"):
    klines = get_binance_klines(symbol, timeframe)
    source = "binance"
    if not klines:
        klines = get_mexc_klines(symbol, timeframe)
        source = "mexc"
    
    if not klines or len(klines) < 50:
        price = get_coingecko_price(symbol)
        return {
            "asset": symbol, "price": round(price, 4) if price else 0, 
            "confidence": 0, "signal": "NONE", "direction": "NEUTRAL", "rsi": 0, 
            "reasons": ["No OHLCV Data"], "ema20": 0, "ema50": 0, 
            "pullback_pct": 0, "bounce_pct": 0, "source": "price_only", 
            "entry": 0, "stop_loss": 0, "take_profit": 0, "status": "na",
            "pnl": 0, "timestamp": datetime.utcnow().isoformat()
        }
    
    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    price = closes[-1]
    
    rsi = calc_rsi(closes)[-1]
    ema20_arr = calc_ema(closes, 20)
    ema50_arr = calc_ema(closes, 50)
    ema20 = ema20_arr[-1] if len(ema20_arr) > 0 else price
    ema50 = ema50_arr[-1] if len(ema50_arr) > 0 else price
    
    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False
    
    # ===== LONG CONDITIONS =====
    long_score = 0
    long_reasons = []
    
    if rsi < 40:
        long_score += 25
        long_reasons.append(f"RSI {rsi:.1f}")
    if price > ema50:
        long_score += 25
        long_reasons.append("Above EMA50")
    if 3 < pullback < 10:
        long_score += 25
        long_reasons.append(f"Dip {pullback:.1f}%")
    if vol_spike:
        long_score += 25
        long_reasons.append("Vol Spike")
    
    # ===== SHORT CONDITIONS =====
    short_score = 0
    short_reasons = []
    
    if rsi > 60:
        short_score += 25
        short_reasons.append(f"RSI {rsi:.1f}")
    if price < ema50:
        short_score += 25
        short_reasons.append("Below EMA50")
    if 3 < bounce < 10:
        short_score += 25
        short_reasons.append(f"Bounce {bounce:.1f}%")
    if vol_spike:
        short_score += 25
        short_reasons.append("Vol Spike")
    
    # ===== DECIDE DIRECTION & SIGNAL =====
    if long_score >= short_score:
        confidence = long_score
        direction = "LONG"
        reasons = long_reasons
        
        if confidence >= 75:
            signal = "ALPHA_LONG"
            stop_loss = round(price * 0.97, 4)
            take_profit = round(price * 1.09, 4)
        elif confidence >= 60:
            signal = "BUY"
            stop_loss = round(price * 0.96, 4)
            take_profit = round(price * 1.08, 4)
        elif confidence >= 45:
            signal = "WATCH_LONG"
            stop_loss = 0
            take_profit = 0
        else:
            signal = "NONE"
            stop_loss = 0
            take_profit = 0
    else:
        confidence = short_score
        direction = "SHORT"
        reasons = short_reasons
        
        if confidence >= 75:
            signal = "ALPHA_SHORT"
            stop_loss = round(price * 1.03, 4)
            take_profit = round(price * 0.91, 4)
        elif confidence >= 60:
            signal = "SHORT"
            stop_loss = round(price * 1.04, 4)
            take_profit = round(price * 0.92, 4)
        elif confidence >= 45:
            signal = "WATCH_SHORT"
            stop_loss = 0
            take_profit = 0
        else:
            signal = "NONE"
            stop_loss = 0
            take_profit = 0
    
    return {
        "asset": symbol,
        "price": round(price, 4),
        "rsi": round(rsi, 1),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "pullback_pct": round(pullback, 2),
        "bounce_pct": round(bounce, 2),
        "confidence": confidence,
        "signal": signal,
        "direction": direction,
        "entry": round(price, 4) if signal not in ["NONE", "WATCH_LONG", "WATCH_SHORT"] else 0,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "reasons": reasons,
        "source": source,
        "status": "open" if signal in ["ALPHA_LONG", "ALPHA_SHORT", "BUY", "SHORT"] else "na",
        "pnl": 0,
        "timestamp": datetime.utcnow().isoformat()
    }

def check_closed_signals():
    global signal_history
    for signal in signal_history:
        if signal.get("status") == "open" and signal.get("entry", 0) > 0:
            current_price = get_current_price(signal["asset"])
            if current_price == 0:
                continue
            if signal["direction"] == "LONG":
                if current_price >= signal["take_profit"]:
                    signal["pnl"] = round((signal["take_profit"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"
                elif current_price <= signal["stop_loss"]:
                    signal["pnl"] = round((signal["stop_loss"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"
            elif signal["direction"] == "SHORT":
                if current_price <= signal["take_profit"]:
                    signal["pnl"] = round((signal["entry"] - signal["take_profit"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"
                elif current_price >= signal["stop_loss"]:
                    signal["pnl"] = round((signal["entry"] - signal["stop_loss"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"

def run_scanner():
    global signal_history
    check_closed_signals()
    
    results = {}
    for asset in ASSETS:
        data = analyze_asset(asset, "1h")
        if data:
            results[asset] = data
            
            if data["signal"] in ["ALPHA_LONG", "ALPHA_SHORT"] and asset not in last_alerted and bot:
                asyncio.create_task(send_alpha_alert(asset, data))
                last_alerted[asset] = time.time()
            
            if data["signal"] not in ["NONE"]:
                signal_history.append(data)
    
    signal_history = signal_history[-100:]
    cache["signals"] = results
    cache["last_scan"] = time.time()
    return results

async def send_alpha_alert(asset, data):
    if not bot:
        return
    
    direction_emoji = "🚀" if data["direction"] == "LONG" else "🔻"
    msg = f"{direction_emoji} ALPHA {data['direction']}: {asset.replace('USDT','')}\n"
    msg += f"Entry: ${data['entry']}\nSL: ${data['stop_loss']}\nTP: ${data['take_profit']}\n"
    msg += f"Conf: {data['confidence']}/100\nReasons: {', '.join(data['reasons'])}"
    
    if CHAT_ID:
        try:
            await bot.send_message(chat_id=CHAT_ID, text=msg)
        except:
            pass
    
    for premium_id in PREMIUM_CHAT_IDS:
        if premium_id.strip():
            try:
                await bot.send_message(chat_id=premium_id.strip(), text=msg)
            except:
                pass

@app.on_event("startup")
async def startup_event():
    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await bot.set_webhook(url=webhook_url)
        print(f"Webhook set: {webhook_url}")
    run_scanner()

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")
        await handle_message(chat_id, text)
    elif "callback_query" in data:
        query = data["callback_query"]
        chat_id = query["message"]["chat"]["id"]
        data_btn = query["data"]
        await handle_callback(chat_id, data_btn)
    return JSONResponse({"ok": True})

async def handle_message(chat_id, text):
    if text == "/start" and bot:
        keyboard = [
            [InlineKeyboardButton("📊 Scan All", callback_data="scan_all"),
             InlineKeyboardButton("📈 Leaderboard", callback_data="leaderboard")],
            [InlineKeyboardButton("🔍 BTC", callback_data="BTCUSDT"),
             InlineKeyboardButton("🔍 ETH", callback_data="ETHUSDT"),
             InlineKeyboardButton("🔍 SOL", callback_data="SOLUSDT")],
            [InlineKeyboardButton("🔍 XRP", callback_data="XRPUSDT"),
             InlineKeyboardButton("🔍 DOGE", callback_data="DOGEUSDT"),
             InlineKeyboardButton("🔍 ADA", callback_data="ADAUSDT")],
            [InlineKeyboardButton("🔍 TRX", callback_data="TRXUSDT"),
             InlineKeyboardButton("🔍 LINK", callback_data="LINKUSDT"),
             InlineKeyboardButton("🔍 AVAX", callback_data="AVAXUSDT")],
            [InlineKeyboardButton("📊 Status", callback_data="status")]
        ]
        msg = "📊 CROO AI Oracle\n\nFree: BUY/SHORT signals\nPremium: ALPHA signals\n\nTap Scan All"
        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_callback(chat_id, data):
    if not bot:
        return
    if data == "scan_all":
        run_scanner()
        signals = cache["signals"]
        buys = [s for s in signals.values() if s["signal"] == "BUY"]
        shorts = [s for s in signals.values() if s["signal"] == "SHORT"]
        watch = [s for s in signals.values() if s["signal"] in ["WATCH_LONG", "WATCH_SHORT"]]
        
        msg_parts = []
        if buys:
            msg_parts.append("💰 BUY:\n" + "\n".join([f"{s['asset'].replace('USDT','')}: {s['confidence']}/100\nEntry ${s['entry']} | SL ${s['stop_loss']}" for s in buys]))
        if shorts:
            msg_parts.append("🔻 SHORT:\n" + "\n".join([f"{s['asset'].replace('USDT','')}: {s['confidence']}/100\nEntry ${s['entry']} | SL ${s['stop_loss']}" for s in shorts]))
        if watch:
            msg_parts.append("👀 WATCH:\n" + "\n".join([f"{s['asset'].replace('USDT','')}: {s['confidence']}/100 {s['direction']}" for s in watch]))
        
        if msg_parts:
            msg = "\n\n".join(msg_parts)
        else:
            msg = "No signals. Scanned 9 coins:\n" + "\n".join([f"{k.replace('USDT','')}: {v['confidence']}/100 {v['direction']}" for k,v in signals.items()])
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif data in ASSETS:
        run_scanner()
        s = cache["signals"].get(data)
        if not s:
            msg = f"Error fetching {data.replace('USDT','')}"
        elif s["signal"] == "NONE":
            msg = f"No setup for {data.replace('USDT','')}.\nRSI: {s['rsi']} | Conf: {s['confidence']}/100\nSource: {s.get('source','unknown')}"
        elif s["signal"] in ["ALPHA_LONG", "ALPHA_SHORT"]:
            msg = f"🔒 {s['signal']}: {s['asset'].replace('USDT','')}\nUpgrade to Premium for entry/SL/TP"
        else:
            msg = f"{s['signal']}: {s['asset'].replace('USDT','')}\n"
            msg += f"Entry: ${s['entry']}\nSL: ${s['stop_loss']}\nTP: ${s['take_profit']}\n"
            msg += f"Conf: {s['confidence']}/100 | RSI: {s['rsi']}\nReasons: {', '.join(s['reasons'])}"
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif data == "status":
        uptime = int(time.time() - start_time)
        last = int(time.time() - cache["last_scan"]) if cache["last_scan"] else 0
        open_signals = len([s for s in signal_history if s.get("status") == "open"])
        msg = f"📊 CROO Oracle Status\nUptime: {uptime//3600}h {(uptime%3600)//60}m\nLast scan: {last}s ago\nOpen: {open_signals}\nTotal: {len(signal_history)}"
        await bot.send_message(chat_id=chat_id, text=msg)
    
    elif data == "leaderboard":
        wins = [s for s in signal_history if s.get("status") == "win"][-5:]
        if not wins:
            msg = "No closed wins yet. ALPHA signals target 70% win rate."
        else:
            msg = "🏆 Recent Wins:\n" + "\n".join([f"{s['asset'].replace('USDT','')}: +{s['pnl']}% | {s['direction']}" for s in wins])
        await bot.send_message(chat_id=chat_id, text=msg)

@app.get("/")
def root():
    return {"status": "CROO Oracle Online", "uptime": int(time.time() - start_time)}

@app.get("/debug")
def debug():
    run_scanner()
    return cache["signals"]

@app.get("/oracle")
def oracle(asset: str = None):
    run_scanner()
    if asset:
        return cache["signals"].get(asset.upper(), {"error": "Asset not found"})
    return cache["signals"]

@app.post("/oracle")
async def oracle_post(request: Request):
    data = await request.json()
    asset = data.get("asset", "").upper()
    run_scanner()
    return cache["signals"].get(asset, {"error": "Asset not found"})

@app.get("/stats")
def stats():
    check_closed_signals()
    closed = [s for s in signal_history if s.get("status") in ["win", "loss"]]
    wins = [s for s in closed if s["status"] == "win"]
    total = len(closed)
    win_rate = round(len(wins)/total*100, 1) if total else 0
    avg_pnl = round(sum(s["pnl"] for s in closed)/total, 2) if total else 0
    return {
        "total_closed": total,
        "wins": len(wins),
        "losses": total - len(wins),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl,
        "last_signal": signal_history[-1] if signal_history else None
    }

@app.get("/cap/health")
def cap_health():
    return {
        "agent": "CROO Oracle",
        "status": "active",
        "assets": ASSETS,
        "uptime_seconds": int(time.time() - start_time),
        "version": "3.0-croo"
    }
