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

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0"))
CHAT_ID = os.environ.get("CHAT_ID")

PAYMENTS_ENABLED = False
PAYMENT_PROVIDER_TOKEN = ""

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "AVAXUSDT", "DOGEUSDT", "TRXUSDT", "ADAUSDT", "TONUSDT"
]

CG_MAP = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "XRPUSDT": "ripple", "BNBUSDT": "binancecoin", "AVAXUSDT": "avalanche-2",
    "DOGEUSDT": "dogecoin", "TRXUSDT": "tron", "ADAUSDT": "cardano", "TONUSDT": "the-open-network"
}

KRAKEN_MAP = {
    "BTCUSDT": "XBTUSD", "ETHUSDT": "ETHUSD", "SOLUSDT": "SOLUSD",
    "XRPUSDT": "XRPUSD", "BNBUSDT": "BNBUSD", "AVAXUSDT": "AVAXUSD",
    "DOGEUSDT": "DOGEUSD", "TRXUSDT": "TRXUSD", "ADAUSDT": "ADAUSD", "TONUSDT": None
}

# ==================== STATE ====================
cache = {"signals": {}, "last_scan": 0, "market_regime": "neutral", "fear_greed": 50}
signal_history = []
users_db = {}
last_alerted = {}
performance = {"wins": 0, "losses": 0, "total": 0}
agent_memory = {
    "last_100_signals": [], "best_asset": "NONE", "best_asset_win_rate": 0.0,
    "total_calls": 0, "revenue_simulated": 0.0
}
scanner_task = None

# ==================== DATA SOURCES ====================
def fetch_coingecko_ohlcv(asset, days=4):
    try:
        coin_id = CG_MAP[asset]
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        r = requests.get(url, params={"vs_currency": "usd", "days": days, "interval": "hourly"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            prices = data["prices"]
            volumes = data["total_volumes"]
            klines = []
            for i in range(len(prices)):
                ts = prices[i][0]
                close = prices[i][1]
                vol = volumes[i][1] if i < len(volumes) else 0
                klines.append([ts, close, close, close, close, vol])
            return klines[-100:], "CoinGecko"
    except Exception as e:
        print(f"CoinGecko ERR {asset}: {e}")
    return None, None

def fetch_kraken_ohlc(asset):
    try:
        pair = KRAKEN_MAP.get(asset)
        if not pair: return None, None
        url = "https://api.kraken.com/0/public/OHLC"
        r = requests.get(url, params={"pair": pair, "interval": 60}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "error" in data and data["error"]: return None, None
            result_key = list(data["result"].keys())[0]
            return data["result"][result_key][-100:], "Kraken"
    except Exception as e:
        print(f"Kraken ERR {asset}: {e}")
    return None, None

def fetch_coinbase_candles(asset):
    cb_map = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
              "XRPUSDT": "XRP-USD", "BNBUSDT": "BNB-USD", "AVAXUSDT": "AVAX-USD",
              "DOGEUSDT": "DOGE-USD", "TRXUSDT": "TRX-USD", "ADAUSDT": "ADA-USD"}
    try:
        product = cb_map.get(asset)
        if not product: return None, None
        url = f"https://api.exchange.coinbase.com/products/{product}/candles"
        r = requests.get(url, params={"granularity": 3600}, timeout=10)
        if r.status_code == 200:
            klines = r.json()
            klines.reverse()
            return klines[-100:], "Coinbase"
    except Exception as e:
        print(f"Coinbase ERR {asset}: {e}")
    return None, None

def get_ohlcv(asset):
    for func in [fetch_coingecko_ohlcv, fetch_kraken_ohlc, fetch_coinbase_candles]:
        klines, source = func(asset)
        if klines:
            print(f"{source} OK: {asset}")
            return klines, source
    print(f"ALL SOURCES FAILED: {asset}")
    return None, "none"

def get_current_price(asset):
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
            params={"ids": CG_MAP[asset], "vs_currencies": "usd"}, timeout=8)
        if r.status_code == 200: return float(r.json()[CG_MAP[asset]]["usd"])
    except: pass
    return 0

def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.status_code == 200:
            val = int(r.json()["data"][0]["value"])
            cache["fear_greed"] = val
            return val
    except: pass
    return 50

# ==================== INDICATORS ====================
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

def calc_ema(prices, period):
    if len(prices) < period: return np.array([prices[-1]] if len(prices) > 0 else [0])
    alpha = 2 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return np.array(ema)

def grade(confidence):
    if confidence >= 90: return "A+"
    elif confidence >= 80: return "A"
    elif confidence >= 70: return "B"
    elif confidence >= 60: return "C"
    elif confidence >= 50: return "D"
    return "F"

# ==================== USER MANAGEMENT ====================
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

# ==================== CORE ANALYSIS ====================
def detect_regime():
    klines, _ = get_ohlcv("BTCUSDT")
    if not klines or len(klines) < 50: return "neutral"
    closes = np.array([float(k[4]) for k in klines])
    ema50 = calc_ema(closes, 50)[-1]
    return "bullish" if closes[-1] > ema50 else "bearish"

def analyze_asset(symbol):
    klines, source = get_ohlcv(symbol)
    if not klines or len(klines) < 50:
        price = get_current_price(symbol)
        if price > 0:
            return {
                "asset": symbol.replace("USDT", ""), "signal": "WATCH", "confidence": 20,
                "grade": "F", "price": round(price, 4), "entry": round(price, 4),
                "stop_loss": round(price * 0.97, 4), "take_profit": round(price * 1.05, 4),
                "bullish_reasons": ["Price only"], "bearish_reasons": [],
                "missing_conditions": ["Full OHLCV data unavailable"], "source": "price_only", "direction": "NONE"
            }
        return {"asset": symbol.replace("USDT", ""), "signal": "NONE", "confidence": 0, "price": 0, "bullish_reasons": ["No Data"], "bearish_reasons": [], "direction": "NONE"}

    closes = np.array([float(k[4]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])
    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    rsi_val = calc_rsi(closes)[-1]
    ema20 = calc_ema(closes, 20)[-1]
    ema50 = calc_ema(closes, 50)[-1]

    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    avg_vol = np.mean(volumes[-20:])
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False
    price_near_ema20 = abs(price - ema20) / ema20 * 100 < 1.5
    bullish_confirmation = price > prev_close
    bearish_confirmation = price < prev_close

    long_score = 0
    short_score = 0
    bullish_reasons = []
    bearish_reasons = []
    missing_conditions = []

    if rsi_val < 45:
        long_score += 20
        bullish_reasons.append("RSI Oversold")
    elif rsi_val > 55:
        short_score += 20
        bearish_reasons.append("RSI Overbought")
    else:
        missing_conditions.append("RSI neutral")

    if price > ema50:
        long_score += 20
        bullish_reasons.append("Above EMA50")
    elif price < ema50:
        short_score += 20
        bearish_reasons.append("Below EMA50")
    else:
        missing_conditions.append("No clear EMA trend")

    if price > ema50 and 4 < pullback < 12 and price_near_ema20:
        long_score += 20
        bullish_reasons.append(f"Meaningful Dip {pullback:.1f}% to EMA20")
    elif price < ema50 and 4 < bounce < 12 and price_near_ema20:
        short_score += 20
        bearish_reasons.append(f"Dead Cat Bounce {bounce:.1f}% to EMA20")
    else:
        missing_conditions.append("Pullback too shallow/deep or not at EMA20")

    if vol_spike:
        if long_score >= short_score:
            long_score += 20
            bullish_reasons.append("Volume Spike")
        else:
            short_score += 20
            bearish_reasons.append("Volume Spike")
    else:
        missing_conditions.append("No volume confirmation")

    if bullish_confirmation:
        long_score += 20
        bullish_reasons.append("Bullish Confirmation Candle")
    elif bearish_confirmation:
        short_score += 20
        bearish_reasons.append("Bearish Confirmation Candle")
    else:
        missing_conditions.append("No confirmation candle")

    fg = cache["fear_greed"]
    if fg < 25 and long_score >= short_score:
        long_score += 5
        bullish_reasons.append("Extreme Fear")
    if fg > 75 and short_score > long_score:
        short_score += 5
        bearish_reasons.append("Extreme Greed")

    direction = "LONG" if long_score >= short_score else "SHORT"
    confidence = max(long_score, short_score)

    signal = "NONE"
    if confidence >= 70: signal = "BUY" if direction == "LONG" else "SHORT"
    elif confidence >= 50: signal = "WATCH"

    if signal == "BUY":
        stop_loss = round(price * 0.95, 4)
        take_profit = round(price * 1.10, 4)
        entry = round(price, 4)
    elif signal == "SHORT":
        stop_loss = round(price * 1.05, 4)
        take_profit = round(price * 0.90, 4)
        entry = round(price, 4)
    elif signal == "WATCH":
        entry = round(price, 4)
        if direction == "LONG":
            stop_loss = round(price * 0.97, 4)
            take_profit = round(price * 1.05, 4)
        else:
            stop_loss = round(price * 1.03, 4)
            take_profit = round(price * 0.95, 4)
    else:
        stop_loss = 0
        take_profit = 0
        entry = 0

    if not bullish_reasons: bullish_reasons = ["Waiting for setup"]
    if not bearish_reasons: bearish_reasons = ["Waiting for setup"]

    return {
        "asset": symbol.replace("USDT", ""), "price": round(price, 4), "signal": signal,
        "confidence": confidence, "grade": grade(confidence), "direction": direction,
        "entry": entry, "stop_loss": stop_loss, "take_profit": take_profit,
        "rsi": round(rsi_val, 1), "bullish_reasons": bullish_reasons,
        "bearish_reasons": bearish_reasons, "missing_conditions": missing_conditions,
        "source": source, "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"], "timestamp": datetime.utcnow().isoformat()
    }

def update_performance():
    for signal in signal_history:
        if signal.get("status") == "open" and signal.get("entry", 0) > 0:
            current = get_current_price(signal["asset"] + "USDT")
            if current == 0: continue
            if signal["direction"] == "LONG":
                if current >= signal["take_profit"]:
                    signal["pnl"] = round((signal["take_profit"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"; performance["wins"] += 1; performance["total"] += 1
                elif current <= signal["stop_loss"]:
                    signal["pnl"] = round((signal["stop_loss"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"; performance["losses"] += 1; performance["total"] += 1
            elif signal["direction"] == "SHORT":
                if current <= signal["take_profit"]:
                    signal["pnl"] = round((signal["entry"] - signal["take_profit"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"; performance["wins"] += 1; performance["total"] += 1
