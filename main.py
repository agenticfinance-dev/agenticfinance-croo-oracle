import os
import time
import asyncio
import aiohttp
import numpy as np
import json
import random
import websockets
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

# ==================== APP CONFIG ====================
app = FastAPI(
    title="CROO AI Oracle",
    description="Autonomous Crypto Intelligence Agent with Multi-Source Fallback, A2A Capabilities, and Explainable AI",
    version="10.0"
)

# CORS for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0")) if os.environ.get("ADMIN_ID") else 0
CHAT_ID = os.environ.get("CHAT_ID")
PAYMENTS_ENABLED = False
PORT = int(os.getenv("PORT", 8000))

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()

ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "AVAXUSDT", "DOGEUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT"
]

CG_MAP = {
    "BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana",
    "XRPUSDT": "ripple", "BNBUSDT": "binancecoin", "AVAXUSDT": "avalanche-2",
    "DOGEUSDT": "dogecoin", "TRXUSDT": "tron", "ADAUSDT": "cardano", "LINKUSDT": "chainlink"
}

KRAKEN_WS_MAP = {
    "BTCUSDT": "XBT/USD", "ETHUSDT": "ETH/USD", "SOLUSDT": "SOL/USD",
    "XRPUSDT": "XRP/USD", "BNBUSDT": "BNB/USD", "AVAXUSDT": "AVAX/USD",
    "DOGEUSDT": "DOGE/USD", "TRXUSDT": "TRX/USD", "ADAUSDT": "ADA/USD", "LINKUSDT": "LINK/USD"
}

# ==================== STATE ====================
cache = {
    "signals": {},
    "last_scan": 0,
    "last_successful_scan": 0,
    "last_ws_update": 0,
    "market_regime": "neutral",
    "fear_greed": 50,
    "live_prices": {},
    "last_known_prices": {}
}
signal_history = []
users_db = {}
last_alerted = {}
performance = {"wins": 0, "losses": 0, "total": 0, "daily": {}, "weekly": {}}
agent_memory = {
    "last_100_signals": [],
    "best_asset": "NONE",
    "best_asset_win_rate": 0.0,
    "total_calls": 0,
    "revenue_simulated": 0.0
}
scanner_task = None
ws_tasks = []
last_api_call = {}
api_failures = {}
api_semaphore = asyncio.Semaphore(5)
telegram_semaphore = asyncio.Semaphore(1)
recent_signals = set()
cg_cache = {}

# ==================== MEMORY PERSISTENCE ====================
def load_memory():
    global signal_history, performance, agent_memory, cache
    try:
        if os.path.exists("agent_memory.json"):
            with open("agent_memory.json", "r") as f:
                data = json.load(f)
                signal_history = data.get("signal_history", [])
                performance = data.get("performance", {"wins": 0, "losses": 0, "total": 0})
                agent_memory = data.get("agent_memory", {
                    "last_100_signals": [],
                    "best_asset": "NONE",
                    "best_asset_win_rate": 0.0,
                    "total_calls": 0,
                    "revenue_simulated": 0.0
                })
                cache["last_known_prices"] = data.get("last_prices", {})
                print("✅ Memory loaded from disk")
    except Exception as e:
        print(f"⚠️ Memory load failed: {e}")

def save_memory():
    try:
        with open("agent_memory.json", "w") as f:
            json.dump({
                "signal_history": signal_history[-100:],
                "performance": performance,
                "agent_memory": agent_memory,
                "last_prices": cache["last_known_prices"],
                "timestamp": datetime.utcnow().isoformat()
            }, f)
    except Exception as e:
        print(f"⚠️ Memory save failed: {e}")

# ==================== RATE LIMITING ====================
def can_call(name, cooldown=30):
    now = time.time()
    if name not in last_api_call:
        last_api_call[name] = now
        return True
    if now - last_api_call[name] > cooldown:
        last_api_call[name] = now
        return True
    return False

def check_circuit_breaker(name):
    if name in api_failures:
        failures, reset_at = api_failures[name]
        if failures >= 5 and time.time() < reset_at:
            return False
        if time.time() >= reset_at:
            api_failures[name] = (0, time.time() + 600)
    return True

def mark_failure(name):
    if name not in api_failures:
        api_failures[name] = (1, time.time() + 600)
    else:
        failures, reset_at = api_failures[name]
        api_failures[name] = (failures + 1, reset_at)

def mark_success(name):
    if name in api_failures:
        api_failures[name] = (0, time.time() + 600)

# ==================== WEBSOCKET FALLBACK ====================
async def websocket_feed(uri, sub_msg, name, parser):
    retry = 1
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                await ws.send(json.dumps(sub_msg))
                print(f"✅ {name} WebSocket connected")
                retry = 1
                async for msg in ws:
                    data = json.loads(msg)
                    symbol, price = parser(data)
                    if symbol and price:
                        cache["live_prices"][symbol] = price
                        cache["last_known_prices"][symbol] = price
                        cache["last_ws_update"] = time.time()
        except Exception as e:
            print(f"❌ {name} WS error: {e}")
            await asyncio.sleep(retry)
            retry = min(retry * 2, 60)

def parse_bybit(data):
    if "topic" in data and "tickers" in data["topic"]:
        ticker = data["data"]
        return ticker["symbol"], float(ticker["lastPrice"])
    return None, None

def parse_binance(data):
    if "s" in data and "c" in data:
        return data["s"], float(data["c"])
    return None, None

def parse_okx(data):
    if "arg" in data and "data" in data and len(data["data"]) > 0:
        symbol = data["arg"]["instId"].replace("-", "")
        return symbol, float(data["data"][0]["last"])
    return None, None

def parse_kraken(data):
    if isinstance(data, list) and len(data) > 2 and isinstance(data[1], list):
        pair = data[3].replace("/", "")
        return pair, float(data[1][0])
    return None, None

async def start_websockets():
    global ws_tasks
    feeds = [
        ("wss://stream.bybit.com/v5/public/linear", {"op": "subscribe", "args": [f"tickers.{s}" for s in ASSETS]}, "Bybit", parse_bybit),
        ("wss://stream.binance.com:9443/ws", {"method": "SUBSCRIBE", "params": [f"{s.lower()}@ticker" for s in ASSETS], "id": 1}, "Binance", parse_binance),
        ("wss://ws.okx.com:8443/ws/v5/public", {"op": "subscribe", "args": [{"channel": "tickers", "instId": s.replace("USDT","-USDT")} for s in ASSETS]}, "OKX", parse_okx),
        ("wss://ws.kraken.com", {"event": "subscribe", "pair": [KRAKEN_WS_MAP[s] for s in ASSETS if s in KRAKEN_WS_MAP], "subscription": {"name": "ticker"}}, "Kraken", parse_kraken)
    ]
    ws_tasks = []
    for uri, sub_msg, name, parser in feeds:
        task = asyncio.create_task(websocket_feed(uri, sub_msg, name, parser))
        ws_tasks.append(task)
    await asyncio.gather(*ws_tasks, return_exceptions=True)

# ==================== OHLCV FALLBACK CHAIN ====================
async def fetch_okx_ohlc(asset):
    if not can_call(f"okx_{asset}", 30): return None, None
    if not check_circuit_breaker("okx"): return None, None
    try:
        symbol = asset.replace("USDT", "-USDT")
        async with session.get(f"https://www.okx.com/api/v5/market/candles", params={"instId": symbol, "bar": "1H", "limit": "100"}) as r:
            if r.status == 200:
                data = await r.json()
                if data["code"] == "0":
                    raw = data["data"]
                    raw.reverse()
                    klines = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw]
                    mark_success("okx")
                    return klines, "OKX"
    except: pass
    mark_failure("okx")
    return None, None

async def fetch_bybit_ohlc(asset):
    if not can_call(f"bybit_{asset}", 30): return None, None
    if not check_circuit_breaker("bybit"): return None, None
    try:
        async with session.get("https://api.bybit.com/v5/market/kline", params={"category": "linear", "symbol": asset, "interval": "60", "limit": 100}) as r:
            if r.status == 200:
                data = await r.json()
                raw = data["result"]["list"]
                raw.reverse()
                klines = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                mark_success("bybit")
                return klines, "Bybit"
    except: pass
    mark_failure("bybit")
    return None, None

async def fetch_binance_ohlc(asset):
    if not can_call(f"binance_{asset}", 30): return None, None
    if not check_circuit_breaker("binance"): return None, None
    try:
        async with session.get("https://api.binance.com/api/v3/klines", params={"symbol": asset, "interval": "1h", "limit": 100}) as r:
            if r.status == 200:
                raw = await r.json()
                klines = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                mark_success("binance")
                return klines, "Binance"
    except: pass
    mark_failure("binance")
    return None, None

async def fetch_coingecko_ohlcv(asset):
    if not can_call(f"cg_{asset}", 120): return None, None
    if not check_circuit_breaker("coingecko"): return None, None
    if asset in cg_cache and time.time() - cg_cache[asset]["timestamp"] < 300:
        return cg_cache[asset]["data"], "CoinGecko(Cached)"
    try:
        coin_id = CG_MAP[asset]
        async with session.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", params={"vs_currency": "usd", "days": 4, "interval": "hourly"}) as r:
            if r.status == 200:
                data = await r.json()
                prices = data["prices"]
                volumes = data["total_volumes"]
                klines = []
                for i in range(len(prices)):
                    klines.append([prices[i][0], prices[i][1], prices[i][1], prices[i][1], prices[i][1], volumes[i][1] if i < len(volumes) else 0])
                cg_cache[asset] = {"data": klines[-100:], "timestamp": time.time()}
                mark_success("coingecko")
                return klines[-100:], "CoinGecko"
    except: pass
    mark_failure("coingecko")
    return None, None

async def get_ohlcv(asset):
    providers = [fetch_okx_ohlc, fetch_bybit_ohlc, fetch_binance_ohlc, fetch_coingecko_ohlcv]
    for provider in providers:
        try:
            async with api_semaphore:
                data, source = await provider(asset)
                if data and len(data) > 50:
                    return data, source
        except Exception as e:
            print(f"{provider.__name__} error: {e}")
    return None, "none"

# ==================== PRICE FALLBACK CHAIN ====================
async def fetch_binance_price(asset):
    if not can_call(f"binance_price_{asset}", 10): return None
    try:
        async with session.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": asset}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data["price"])
    except: pass
    return None

async def fetch_bybit_price(asset):
    if not can_call(f"bybit_price_{asset}", 10): return None
    try:
        async with session.get("https://api.bybit.com/v5/market/tickers", params={"category": "linear", "symbol": asset}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data["result"]["list"][0]["lastPrice"])
    except: pass
    return None

async def fetch_coingecko_price(asset):
    if not can_call(f"cg_price_{asset}", 60): return None
    try:
        coin_id = CG_MAP[asset]
        async with session.get("https://api.coingecko.com/api/v3/simple/price", params={"ids": coin_id, "vs_currencies": "usd"}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data[coin_id]["usd"])
    except: pass
    return None

async def get_price_fallback(asset):
    if asset in cache["live_prices"] and cache["live_prices"][asset] > 0:
        return cache["live_prices"][asset]
    for fetcher in [fetch_binance_price, fetch_bybit_price, fetch_coingecko_price]:
        price = await fetcher(asset)
        if price:
            cache["last_known_prices"][asset] = price
            return price
    return cache["last_known_prices"].get(asset, 0)

def get_current_price(asset):
    return cache["live_prices"].get(asset, cache["last_known_prices"].get(asset, 0))

# ==================== FEAR & GREED ====================
async def fetch_fear_greed():
    if not can_call("fear_greed", 300):
        return cache["fear_greed"]
    try:
        async with session.get("https://api.alternative.me/fng/") as r:
            if r.status == 200:
                data = await r.json()
                val = int(data["data"][0]["value"])
                cache["fear_greed"] = val
                return val
    except: pass
    return cache["fear_greed"]

# ==================== INDICATORS ====================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return np.array([50.0] * len(closes))
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(closes)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(closes)):
        delta = deltas[i - 1]
        upval = delta if delta > 0 else 0.
        downval = -delta if delta < 0 else 0.
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calc_ema(prices, period):
    if len(prices) < period:
        return np.array([prices[-1]] if prices else [0])
    alpha = 2 / (period + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    return np.array(ema)

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.01
    tr = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr.append(max(hl, hc, lc))
    if len(tr) < period:
        return 0.01
    return np.mean(tr[-period:])

def grade(confidence):
    if confidence >= 90: return "A+"
    elif confidence >= 80: return "A"
    elif confidence >= 70: return "B"
    elif confidence >= 60: return "C"
    elif confidence >= 50: return "D"
    return "F"

# ==================== ENTRY ZONE CALCULATOR ====================
def calculate_entries(price, atr, direction="LONG", signal_type="BUY"):
    """Calculate entry zones instead of current price - CROO compliant"""
    
    if direction == "LONG" and signal_type in ["BUY", "WATCH"]:
        # Multiple entries below current price (0.3-1.5% below)
        aggressive_pct = 0.002  # 0.2% below
        moderate_pct = 0.005    # 0.5% below
        conservative_pct = 0.010 # 1.0% below
        dca_1_pct = 0.020       # 2.0% below
        dca_2_pct = 0.035       # 3.5% below
        
        entries = {
            "aggressive": round(price * (1 - aggressive_pct), 4),
            "moderate": round(price * (1 - moderate_pct), 4),
            "conservative": round(price * (1 - conservative_pct), 4),
            "dca_1": round(price * (1 - dca_1_pct), 4),
            "dca_2": round(price * (1 - dca_2_pct), 4),
        }
        
        # ATR-based entry
        atr_entry = round(price - (atr * 0.3), 4)
        
        # Recommended = moderate (0.5% below)
        recommended = entries["moderate"]
        
        # Entry zone = moderate to conservative
        entry_zone = f"${entries['moderate']} - ${entries['conservative']}"
        
        # Stop loss = 3% below moderate entry
        stop_loss = round(entries['moderate'] * 0.97, 4)
        
        # Take profit = 2x risk
        risk = entries['moderate'] - stop_loss
        take_profit = round(entries['moderate'] + (risk * 2), 4)
        
        # Position sizing suggestion
        position_sizing = "50% at moderate, 30% at conservative, 20% at DCA"
        
    elif direction == "SHORT" and signal_type in ["SHORT", "WATCH"]:
        # Multiple entries above current price
        aggressive_pct = 0.002  # 0.2% above
        moderate_pct = 0.005    # 0.5% above
        conservative_pct = 0.010 # 1.0% above
        dca_1_pct = 0.020       # 2.0% above
        dca_2_pct = 0.035       # 3.5% above
        
        entries = {
            "aggressive": round(price * (1 + aggressive_pct), 4),
            "moderate": round(price * (1 + moderate_pct), 4),
            "conservative": round(price * (1 + conservative_pct), 4),
            "dca_1": round(price * (1 + dca_1_pct), 4),
            "dca_2": round(price * (1 + dca_2_pct), 4),
        }
        
        # ATR-based entry
        atr_entry = round(price + (atr * 0.3), 4)
        
        recommended = entries["moderate"]
        entry_zone = f"${entries['moderate']} - ${entries['conservative']}"
        stop_loss = round(entries['moderate'] * 1.03, 4)
        risk = stop_loss - entries['moderate']
        take_profit = round(entries['moderate'] - (risk * 2), 4)
        position_sizing = "50% at moderate, 30% at conservative, 20% at DCA"
        
    else:
        return {
            "entry": round(price, 4),
            "entry_zone": "N/A",
            "entries": {"current": round(price, 4)},
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A"
        }
    
    risk_reward = round(abs(take_profit - recommended) / abs(stop_loss - recommended), 2) if abs(stop_loss - recommended) > 0 else 0
    
    return {
        "entry": recommended,
        "entry_zone": entry_zone,
        "entries": entries,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": f"{risk_reward}:1",
        "position_sizing": position_sizing,
        "atr_entry": atr_entry
    }

# ==================== USER MANAGEMENT ====================
def is_pro(user_id: int) -> bool:
    user = users_db.get(user_id, {})
    if not user:
        return False
    if user.get("plan") == "lifetime":
        return True
    expires = user.get("pro_expires")
    return expires and datetime.now() < expires

def activate_pro(user_id: int, days: int = 30):
    if user_id not in users_db:
        users_db[user_id] = {}
    users_db[user_id]["plan"] = "pro"
    users_db[user_id]["pro_expires"] = datetime.now() + timedelta(days=days)

# ==================== CORE ANALYSIS ====================
async def detect_regime():
    klines, _ = await get_ohlcv("BTCUSDT")
    if not klines or len(klines) < 50:
        return cache["market_regime"]
    closes = np.array([float(k[4]) for k in klines])
    if len(closes) < 50:
        return cache["market_regime"]
    ema50 = calc_ema(closes, 50)[-1]
    return "bullish" if closes[-1] > ema50 else "bearish"

async def analyze_asset(symbol):
    klines, source = await get_ohlcv(symbol)
    if not klines or len(klines) < 50:
        price = await get_price_fallback(symbol)
        if price > 0:
            entry_data = calculate_entries(price, 0.01, "LONG", "WATCH")
            return {
                "asset": symbol.replace("USDT", ""),
                "signal": "WATCH",
                "confidence": 20,
                "grade": "F",
                "price": round(price, 4),
                "entry": entry_data["entry"],
                "entry_zone": entry_data["entry_zone"],
                "entries": entry_data["entries"],
                "stop_loss": entry_data["stop_loss"],
                "take_profit": entry_data["take_profit"],
                "risk_reward": entry_data["risk_reward"],
                "position_sizing": entry_data["position_sizing"],
                "bullish_reasons": ["Price only - awaiting full data"],
                "bearish_reasons": [],
                "missing_conditions": ["Full OHLCV data unavailable"],
                "source": "price_only",
                "direction": "NONE",
                "risk": "HIGH",
                "holding_period": "N/A"
            }
        return {
            "asset": symbol.replace("USDT", ""),
            "signal": "NONE",
            "confidence": 0,
            "price": 0,
            "entry": 0,
            "entry_zone": "N/A",
            "entries": {},
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A",
            "bullish_reasons": ["No Data"],
            "bearish_reasons": [],
            "direction": "NONE",
            "risk": "UNKNOWN",
            "holding_period": "N/A"
        }

    closes = np.array([float(k[4]) for k in klines])
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])

    price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else price
    rsi_val = calc_rsi(closes)[-1]
    ema20 = calc_ema(closes, 20)[-1]
    ema50 = calc_ema(closes, 50)[-1]
    atr = calc_atr(highs, lows, closes)

    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
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
        bullish_reasons.append(f"RSI Oversold ({rsi_val:.1f})")
    elif rsi_val > 55:
        short_score += 20
        bearish_reasons.append(f"RSI Overbought ({rsi_val:.1f})")
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
        bullish_reasons.append(f"Dip {pullback:.1f}% to EMA20")
    elif price < ema50 and 4 < bounce < 12 and price_near_ema20:
        short_score += 20
        bearish_reasons.append(f"Bounce {bounce:.1f}% to EMA20")
    else:
        missing_conditions.append("Pullback too shallow/deep")

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
        bullish_reasons.append("Bullish Confirmation")
    elif bearish_confirmation:
        short_score += 20
        bearish_reasons.append("Bearish Confirmation")
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
    if confidence >= 60:
        signal = "BUY" if direction == "LONG" else "SHORT"
    elif confidence >= 40:
        signal = "WATCH"

    # ===== ENTRY ZONE CALCULATION =====
    if signal in ["BUY", "WATCH"] and direction == "LONG":
        entry_data = calculate_entries(price, atr, "LONG", signal)
        risk = "LOW" if atr / price < 0.02 else "MEDIUM"
        holding_period = "1-3 days"
    elif signal in ["SHORT", "WATCH"] and direction == "SHORT":
        entry_data = calculate_entries(price, atr, "SHORT", signal)
        risk = "LOW" if atr / price < 0.02 else "MEDIUM"
        holding_period = "1-3 days"
    else:
        entry_data = {
            "entry": round(price, 4),
            "entry_zone": "N/A",
            "entries": {"current": round(price, 4)},
            "stop_loss": 0,
            "take_profit": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A"
        }
        risk = "N/A"
        holding_period = "N/A"

    if not bullish_reasons:
        bullish_reasons = ["Waiting for setup"]
    if not bearish_reasons:
        bearish_reasons = ["Waiting for setup"]

    return {
        "asset": symbol.replace("USDT", ""),
        "price": round(price, 4),
        "signal": signal,
        "confidence": confidence,
        "grade": grade(confidence),
        "direction": direction,
        "entry": entry_data["entry"],
        "entry_zone": entry_data["entry_zone"],
        "entries": entry_data["entries"],
        "stop_loss": entry_data["stop_loss"],
        "take_profit": entry_data["take_profit"],
        "risk_reward": entry_data["risk_reward"],
        "position_sizing": entry_data["position_sizing"],
        "rsi": round(rsi_val, 1),
        "atr": round(atr, 4),
        "risk": risk,
        "holding_period": holding_period,
        "bullish_reasons": bullish_reasons,
        "bearish_reasons": bearish_reasons,
        "missing_conditions": missing_conditions,
        "source": source,
        "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"],
        "pullback_pct": round(pullback, 2),
        "timestamp": datetime.utcnow().isoformat()
    }

# ==================== PERFORMANCE TRACKING ====================
async def update_performance():
    for signal in signal_history:
        if signal.get("status") == "open" and signal.get("entry", 0) > 0:
            current = get_current_price(signal["asset"] + "USDT")
            if current == 0:
                continue
            if signal["direction"] == "LONG":
                if current >= signal["take_profit"]:
                    signal["pnl"] = round((signal["take_profit"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"
                    performance["wins"] += 1
                    performance["total"] += 1
                elif current <= signal["stop_loss"]:
                    signal["pnl"] = round((signal["stop_loss"] - signal["entry"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"
                    performance["losses"] += 1
                    performance["total"] += 1
            elif signal["direction"] == "SHORT":
                if current <= signal["take_profit"]:
                    signal["pnl"] = round((signal["entry"] - signal["take_profit"]) / signal["entry"] * 100, 2)
                    signal["status"] = "win"
                    performance["wins"] += 1
                    performance["total"] += 1
                elif current >= signal["stop_loss"]:
                    signal["pnl"] = round((signal["entry"] - signal["stop_loss"]) / signal["entry"] * 100, 2)
                    signal["status"] = "loss"
                    performance["losses"] += 1
                    performance["total"] += 1
    update_memory()
    save_memory()

def update_memory():
    agent_memory["last_100_signals"] = signal_history[-100:]
    stats = {}
    for sig in signal_history:
        if sig.get("status") in ["win", "loss"]:
            asset = sig["asset"]
            if asset not in stats:
                stats[asset] = {"wins": 0, "total": 0}
            stats[asset]["total"] += 1
            if sig["status"] == "win":
                stats[asset]["wins"] += 1
    best = None
    best_rate = 0
    for asset, s in stats.items():
        if s["total"] >= 3:
            rate = s["wins"] / s["total"]
            if rate > best_rate:
                best_rate = rate
                best = asset
    agent_memory["best_asset"] = best or "NONE"
    agent_memory["best_asset_win_rate"] = round(best_rate * 100, 1) if best_rate > 0 else 0

# ==================== ALERTS ====================
async def send_alert(signal):
    if not bot or signal["confidence"] < 60:
        return
    if signal["asset"] in last_alerted and time.time() - last_alerted[signal["asset"]] < 3600:
        return

    msg = f"🚨 {signal['signal']} SIGNAL\n\n"
    msg += f"Asset: {signal['asset']}\n"
    msg += f"Confidence: {signal['confidence']}% ({signal['grade']})\n"
    msg += f"Risk: {signal['risk']}\n\n"
    msg += f"Entry Zone: {signal['entry_zone']}\n"
    msg += f"Entry: ${signal['entry']}\n"
    msg += f"Target: ${signal['take_profit']}\n"
    msg += f"Stop: ${signal['stop_loss']}\n"
    msg += f"R:R: {signal['risk_reward']}\n\n"
    msg += f"Position Sizing:\n{signal['position_sizing']}\n\n"
    msg += f"Reasons:\n"
    reasons = signal['bullish_reasons'] if signal['direction'] == 'LONG' else signal['bearish_reasons']
    msg += "\n".join([f"✅ {r}" for r in reasons[:5]])
    msg += f"\n\nMarket: {signal['market_regime'].upper()} | F&G: {signal['fear_greed']}"
    msg += f"\nHolding Period: {signal['holding_period']}"

    if CHAT_ID:
        async with telegram_semaphore:
            try:
                await bot.send_message(chat_id=CHAT_ID, text=msg)
                await asyncio.sleep(0.2)
            except:
                pass

    last_alerted[signal["asset"]] = time.time()

# ==================== SCANNER ====================
async def scan_all():
    print(f"🔄 AUTO SCAN {datetime.utcnow()}")
    await fetch_fear_greed()
    await update_performance()
    cache["market_regime"] = await detect_regime()

    results = {}
    for asset in ASSETS:
        data = await analyze_asset(asset)
        if data:
            results[asset] = data
            signal_key = f"{data['asset']}_{data['signal']}_{data['direction']}"
            if data["signal"] in ["BUY", "SHORT"] and signal_key not in recent_signals:
                data["status"] = "open"
                signal_history.append(data)
                agent_memory["total_calls"] += 1
                agent_memory["revenue_simulated"] += 0.01
                recent_signals.add(signal_key)
                asyncio.create_task(send_alert(data))

    signal_history[:] = signal_history[-100:]
    cache["signals"] = results
    cache["last_scan"] = time.time()
    cache["last_successful_scan"] = time.time()
    save_memory()

    for asset in list(last_alerted.keys()):
        if time.time() - last_alerted[asset] > 86400:
            del last_alerted[asset]

    print(f"✅ Scan complete. {len(results)} assets analyzed.")
    return results

async def scanner_loop():
    print("🚀 Auto scanner started")
    while True:
        try:
            await scan_all()
            jitter = random.randint(-15, 15)
            await asyncio.sleep(300 + jitter)
        except Exception as e:
            print(f"❌ Scanner error: {e}")
            await asyncio.sleep(60)

# ==================== A2A ENDPOINT ====================
@app.post("/a2a")
async def a2a(request: Request):
    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    agent = data.get("agent", "Unknown")
    request_type = data.get("request", "")

    if request_type == "best_trade":
        await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            return JSONResponse({"response": {"message": "No signals available"}})
        best = max(signals, key=lambda x: x.get("confidence", 0))
        return JSONResponse({
            "response": {
                "asset": best.get("asset"),
                "signal": best.get("signal"),
                "confidence": best.get("confidence"),
                "entry_zone": best.get("entry_zone"),
                "entry": best.get("entry"),
                "tp": best.get("take_profit"),
                "sl": best.get("stop_loss"),
                "risk_reward": best.get("risk_reward"),
                "risk": best.get("risk")
            },
            "from_agent": "CROO Oracle",
            "to_agent": agent
        })

    elif request_type == "market_intel":
        await scan_all()
        return JSONResponse({
            "response": {
                "market_regime": cache["market_regime"],
                "fear_greed": cache["fear_greed"],
                "signals": len([s for s in cache["signals"].values() if s.get("signal") in ["BUY", "SHORT"]]),
                "top_asset": agent_memory["best_asset"]
            },
            "from_agent": "CROO Oracle",
            "to_agent": agent
        })

    return JSONResponse({"error": f"Unknown request: {request_type}"})

# ==================== TELEGRAM ====================
async def send_rich_card(chat_id, s):
    if s.get("signal") == "NONE":
        msg = "⏳ NO TRADE SETUP\n\n"
    elif s.get("signal") == "WATCH":
        msg = "⚠️ WATCHLIST SETUP\n\n"
    else:
        msg = f"🚨 {s.get('signal')} SIGNAL\n\n"

    msg += f"Asset: {s.get('asset')}\n"
    msg += f"Confidence: {s.get('confidence')}% ({s.get('grade')})\n"
    msg += f"Risk: {s.get('risk')}\n"
    msg += f"Price: ${s.get('price')}\n\n"
    msg += f"Entry Zone: {s.get('entry_zone')}\n"
    msg += f"Entry: ${s.get('entry')}\n"
    msg += f"Target: ${s.get('take_profit')}\n"
    msg += f"Stop: ${s.get('stop_loss')}\n"
    msg += f"R:R: {s.get('risk_reward')}\n\n"
    msg += f"Position Sizing:\n{s.get('position_sizing')}\n\n"

    if s.get('direction') == 'LONG':
        msg += f"Bullish Reasons:\n" + "\n".join([f"✅ {r}" for r in s.get('bullish_reasons', ['None'])[:5]])
    else:
        msg += f"Bearish Reasons:\n" + "\n".join([f"✅ {r}" for r in s.get('bearish_reasons', ['None'])[:5]])

    msg += f"\n\nMarket: {s.get('market_regime','').upper()} | F&G: {s.get('fear_greed')}"
    msg += f"\nSource: {s.get('source', 'N/A')} | Hold: {s.get('holding_period', 'N/A')}"
    msg += f"\nPullback: {s.get('pullback_pct', 0)}%"
    await bot.send_message(chat_id=chat_id, text=msg)

# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "agent": "CROO AI Oracle",
        "version": "10.0",
        "assets": len(ASSETS),
        "status": "online",
        "uptime": str(timedelta(seconds=int(time.time() - start_time))),
        "entry_strategy": "Zone-based entries (0.5-1% below/above current price)",
        "endpoints": [
            "/oracle", "/best_signal", "/leaderboard", "/stats",
            "/history", "/agent/query", "/a2a",
            "/cap/metadata", "/cap/health", "/pricing", "/capabilities",
            "/explain/{symbol}", "/why/{symbol}",
            "/portfolio", "/demo", "/business_model",
            "/.well-known/agent.json"
        ]
    }

@app.head("/")
def root_head():
    return {"status": "ok"}

@app.get("/oracle")
async def oracle():
    await scan_all()
    return cache["signals"]

@app.get("/best_signal")
async def best_signal():
    await scan_all()
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    if not signals:
        return JSONResponse({"message": "No signals right now"})
    best = max(signals, key=lambda x: x.get("confidence", 0))
    return JSONResponse({
        "asset": best.get("asset"),
        "signal": best.get("signal"),
        "confidence": best.get("confidence"),
        "grade": best.get("grade"),
        "price": best.get("price"),
        "entry_zone": best.get("entry_zone"),
        "entry": best.get("entry"),
        "tp": best.get("take_profit"),
        "sl": best.get("stop_loss"),
        "risk_reward": best.get("risk_reward"),
        "risk": best.get("risk"),
        "holding_period": best.get("holding_period"),
        "position_sizing": best.get("position_sizing"),
        "reasons": best.get("bullish_reasons") if best.get("direction") == "LONG" else best.get("bearish_reasons")
    })

@app.get("/leaderboard")
async def leaderboard():
    await scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    return JSONResponse([{
        "asset": s.get("asset"),
        "signal": s.get("signal"),
        "confidence": s.get("confidence"),
        "grade": s.get("grade"),
        "price": s.get("price"),
        "entry_zone": s.get("entry_zone"),
        "risk_reward": s.get("risk_reward"),
        "source": s.get("source")
    } for s in signals[:10]])

@app.get("/stats")
async def stats():
    await update_performance()
    accuracy = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0
    return JSONResponse({
        "accuracy": f"{accuracy}%",
        "total_signals": performance["total"],
        "wins": performance["wins"],
        "losses": performance["losses"],
        "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"],
        "best_asset": agent_memory["best_asset"],
        "best_asset_win_rate": f"{agent_memory['best_asset_win_rate']}%"
    })

@app.get("/history")
def history():
    return signal_history[-50:]

@app.post("/agent/query")
async def agent_query(req: Request):
    try:
        data = await req.json()
    except:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    task = data.get("task", "")

    if task == "find_best_pullback":
        await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            return JSONResponse({"message": "No signals"})
        best = max(signals, key=lambda x: x.get("confidence", 0))
        return JSONResponse({
            "asset": best.get("asset"),
            "signal": best.get("signal"),
            "confidence": best.get("confidence"),
            "entry_zone": best.get("entry_zone"),
            "entry": best.get("entry"),
            "tp": best.get("take_profit"),
            "sl": best.get("stop_loss"),
            "risk_reward": best.get("risk_reward"),
            "risk": best.get("risk"),
            "reason": best.get("bullish_reasons") if best.get("direction") == "LONG" else best.get("bearish_reasons")
        })

    elif task == "get_all_signals":
        await scan_all()
        signals = []
        for s in cache["signals"].values():
            if s.get("confidence", 0) > 0:
                signals.append({
                    "asset": s.get("asset"),
                    "signal": s.get("signal"),
                    "confidence": s.get("confidence"),
                    "price": s.get("price"),
                    "grade": s.get("grade"),
                    "entry_zone": s.get("entry_zone"),
                    "risk_reward": s.get("risk_reward")
                })
        return JSONResponse(signals)

    elif task == "get_market_intelligence":
        await scan_all()
        return JSONResponse({
            "timestamp": datetime.utcnow().isoformat(),
            "market_regime": cache["market_regime"],
            "fear_greed": cache["fear_greed"],
            "total_signals": len([s for s in cache["signals"].values() if s.get("signal") in ["BUY", "SHORT"]]),
            "assets_tracked": len(ASSETS)
        })

    elif task == "explain_signal":
        asset = data.get("asset", "BTC")
        await scan_all()
        signal = cache["signals"].get(f"{asset}USDT", {})
        if not signal:
            return JSONResponse({"error": f"No signal for {asset}"})
        return JSONResponse({
            "asset": signal.get("asset"),
            "decision": signal.get("signal"),
            "confidence": signal.get("confidence"),
            "explanation": signal.get("bullish_reasons") if signal.get("direction") == "LONG" else signal.get("bearish_reasons"),
            "risk": signal.get("risk"),
            "holding_period": signal.get("holding_period"),
            "entry_zone": signal.get("entry_zone")
        })

    elif task == "predict_asset":
        await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            return JSONResponse({"message": "No predictions"})
        return JSONResponse([{
            "asset": s.get("asset"),
            "score": s.get("confidence"),
            "signal": s.get("signal"),
            "entry_zone": s.get("entry_zone")
        } for s in sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)[:5]])

    elif task == "rank_assets":
        await scan_all()
        signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
        return JSONResponse([{
            "asset": s.get("asset"),
            "score": s.get("confidence"),
            "signal": s.get("signal"),
            "grade": s.get("grade"),
            "entry_zone": s.get("entry_zone")
        } for s in signals[:5]])

    return JSONResponse({"error": "Unknown task. Use: find_best_pullback, get_all_signals, get_market_intelligence, explain_signal, predict_asset, rank_assets"})

@app.get("/why/{symbol}")
async def why(symbol: str):
    asset = symbol.upper() + "USDT"
    await scan_all()
    signal = cache["signals"].get(asset, {})
    if not signal:
        return JSONResponse({"error": f"No signal for {symbol}"}, status_code=404)

    if signal.get("signal") == "NONE":
        explanation = "No trade setup detected. Missing conditions: " + ", ".join(signal.get("missing_conditions", []))
    else:
        reasons = signal.get("bullish_reasons") if signal.get("direction") == "LONG" else signal.get("bearish_reasons")
        explanation = f"{signal.get('signal')} signal. " + ". ".join(reasons[:3])

    return JSONResponse({
        "asset": signal.get("asset"),
        "decision": signal.get("signal"),
        "confidence": signal.get("confidence"),
        "explanation": explanation,
        "risk": signal.get("risk"),
        "holding_period": signal.get("holding_period"),
        "entry_zone": signal.get("entry_zone"),
        "risk_reward": signal.get("risk_reward"),
        "position_sizing": signal.get("position_sizing"),
        "market_regime": signal.get("market_regime"),
        "fear_greed": signal.get("fear_greed")
    })

@app.post("/portfolio")
async def portfolio(req: Request):
    try:
        data = await req.json()
        capital = float(data.get("capital", 1000))
    except:
        return JSONResponse({"error": "Invalid input. Use {'capital': 1000}"}, status_code=400)

    await scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    if not signals:
        return JSONResponse({"error": "No signals for portfolio allocation"})

    total_conf = sum(s.get("confidence", 0) for s in signals[:5]) or 1
    allocation = {}
    for s in signals[:5]:
        weight = s.get("confidence", 0) / total_conf
        allocation[s.get("asset")] = {
            "amount": round(capital * weight, 2),
            "entry_zone": s.get("entry_zone"),
            "risk_reward": s.get("risk_reward")
        }

    return JSONResponse({
        "capital": capital,
        "allocation": allocation,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.get("/demo")
async def demo():
    await scan_all()
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    best = max(signals, key=lambda x: x.get("confidence", 0)) if signals else None
    accuracy = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0

    return JSONResponse({
        "agent": "CROO AI Oracle",
        "version": "10.0",
        "status": "active",
        "entry_strategy": "Zone-based entries (0.5-1% below/above current price)",
        "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"],
        "best_signal": {
            "asset": best.get("asset") if best else None,
            "signal": best.get("signal") if best else None,
            "confidence": best.get("confidence") if best else None,
            "entry_zone": best.get("entry_zone") if best else None
        } if best else None,
        "top_3_assets": [{
            "asset": s.get("asset"),
            "confidence": s.get("confidence"),
            "signal": s.get("signal"),
            "entry_zone": s.get("entry_zone")
        } for s in signals[:3]] if signals else [],
        "accuracy": f"{accuracy}%",
        "total_signals": performance["total"],
        "uptime": str(timedelta(seconds=int(time.time() - start_time)))
    })

@app.get("/business_model")
def business_model():
    return JSONResponse({
        "free": {
            "requests": "5/day",
            "features": ["Basic signals", "3 assets", "Current price entry only"]
        },
        "pro": {
            "price": "$9.99/month",
            "features": [
                "All assets",
                "Multi-source data",
                "Priority alerts",
                "Full history",
                "Telegram alerts",
                "Entry zone recommendations",
                "Position sizing guidance"
            ]
        },
        "enterprise": {
            "price": "Custom pricing",
            "features": [
                "All assets",
                "Webhook integration",
                "White-label",
                "Dedicated support",
                "Custom alerts",
                "API access",
                "Custom entry strategies"
            ]
        },
        "note": "Payments disabled during CROO Hackathon. All features unlocked for judges."
    })

@app.get("/explain/{symbol}")
async def explain(symbol: str):
    asset = symbol.upper() + "USDT"
    signal = cache["signals"].get(asset, {})
    if not signal:
        return JSONResponse({"error": "No signal found", "symbol": symbol}, status_code=404)
    return JSONResponse({
        "asset": signal.get("asset"),
        "signal": signal.get("signal"),
        "confidence": signal.get("confidence"),
        "grade": signal.get("grade"),
        "bullish_reasons": signal.get("bullish_reasons"),
        "bearish_reasons": signal.get("bearish_reasons"),
        "missing_conditions": signal.get("missing_conditions"),
        "market_regime": signal.get("market_regime"),
        "fear_greed": signal.get("fear_greed"),
        "price": signal.get("price"),
        "entry_zone": signal.get("entry_zone"),
        "entries": signal.get("entries"),
        "take_profit": signal.get("take_profit"),
        "stop_loss": signal.get("stop_loss"),
        "risk_reward": signal.get("risk_reward"),
        "position_sizing": signal.get("position_sizing"),
        "source": signal.get("source"),
        "rsi": signal.get("rsi"),
        "atr": signal.get("atr"),
        "risk": signal.get("risk"),
        "holding_period": signal.get("holding_period"),
        "pullback_pct": signal.get("pullback_pct"),
        "direction": signal.get("direction"),
        "timestamp": signal.get("timestamp")
    })

@app.get("/agent/revenue")
def revenue():
    return JSONResponse({
        "total_calls": agent_memory["total_calls"],
        "revenue_simulated": round(agent_memory["revenue_simulated"], 2),
        "avg_per_call": round(agent_memory["revenue_simulated"] / max(1, agent_memory["total_calls"]), 4)
    })

@app.get("/reputation")
def reputation():
    score = min(100, performance["wins"] * 2)
    return JSONResponse({
        "reputation_score": score,
        "grade": grade(score),
        "signals_generated": performance["total"],
        "win_rate": round(performance["wins"] / max(1, performance["total"]) * 100, 1)
    })

@app.get("/cap/metadata")
def cap_metadata():
    return JSONResponse({
        "agent": "CROO AI Oracle",
        "version": "10.0",
        "category": "Market Intelligence",
        "callable": True,
        "supports": [a.replace("USDT", "") for a in ASSETS],
        "entry_strategy": "Zone-based entries (0.5-1% below/above current price)",
        "features": [
            "pullback_detection",
            "confidence_scoring",
            "market_intelligence",
            "regime_detection",
            "signal_ranking",
            "explainability",
            "auto_alerts",
            "multi_source_data",
            "A2A_compatible",
            "entry_zone_recommendations",
            "position_sizing"
        ],
        "pricing": {
            "free": "5 requests/day",
            "pro": "$9.99/month",
            "enterprise": "Custom pricing"
        }
    })

@app.get("/cap/health")
def cap_health():
    return JSONResponse({
        "agent": "CROO AI Oracle",
        "status": "active",
        "assets": len(ASSETS),
        "uptime": str(timedelta(seconds=int(time.time() - start_time))),
        "payments": "disabled_for_judging",
        "auto_scanner": "active_5min",
        "websocket_feed": "active",
        "entry_strategy": "Zone-based entries (0.5-1% below/above current price)",
        "last_scan": f"{int(time.time() - cache['last_successful_scan'])}s ago" if cache["last_successful_scan"] else "never",
        "last_price_update": f"{int(time.time() - cache['last_ws_update'])}s ago" if cache["last_ws_update"] else "never",
        "signals_generated": performance["total"],
        "active_users": len(users_db),
        "features": ["pullback", "regime", "fear_greed", "A2A", "explainability", "portfolio", "entry_zones"],
        "version": "10.0-croo-final"
    })

@app.get("/pricing")
def pricing():
    return business_model()

@app.get("/capabilities")
def capabilities():
    return JSONResponse({
        "features": [
            "pullback_detection",
            "confidence_scoring",
            "market_intelligence",
            "regime_detection",
            "fear_greed_integration",
            "signal_ranking",
            "explainability",
            "auto_alerts",
            "telegram_integration",
            "A2A_compatible",
            "CAP_metadata",
            "multi_source_data",
            "performance_tracking",
            "agent_memory",
            "portfolio_management",
            "risk_management",
            "entry_zone_recommendations",
            "position_sizing_guidance"
        ],
        "assets": [a.replace("USDT", "") for a in ASSETS],
        "sources": ["Binance", "Bybit", "OKX", "Kraken", "CoinGecko"],
        "entry_strategy": {
            "type": "Zone-based",
            "long_entries": "0.2% - 3.5% below current price",
            "short_entries": "0.2% - 3.5% above current price",
            "position_sizing": "50% moderate, 30% conservative, 20% DCA",
            "risk_reward": "Minimum 2:1"
        },
        "api_endpoints": [
            "/oracle", "/best_signal", "/leaderboard", "/stats",
            "/history", "/agent/query", "/a2a",
            "/cap/metadata", "/cap/health", "/pricing", "/capabilities",
            "/explain/{symbol}", "/why/{symbol}",
            "/agent/revenue", "/reputation", "/portfolio", "/demo",
            "/business_model", "/.well-known/agent.json"
        ]
    })

@app.get("/.well-known/agent.json")
def agent_manifest():
    return {
        "name": "CROO AI Oracle",
        "description": "Autonomous crypto intelligence agent with pullback detection, market regime analysis, entry zone recommendations, and explainable AI",
        "endpoint": "/agent/query",
        "a2a_endpoint": "/a2a",
        "entry_strategy": {
            "type": "Zone-based entries",
            "description": "Never enters at current price. Uses 0.5-1% below/above current price with multiple levels"
        },
        "capabilities": [
            "pullback_detection",
            "market_intelligence",
            "signal_ranking",
            "regime_detection",
            "explainability",
            "multi_source_data",
            "auto_alerts",
            "portfolio_management",
            "entry_zone_recommendations"
        ],
        "assets": [a.replace("USDT", "") for a in ASSETS],
        "version": "10.0"
    }

# ==================== TELEGRAM WEBHOOK ====================
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

# ==================== TELEGRAM HANDLERS ====================
async def handle_buy(chat_id, user_id):
    if PAYMENTS_ENABLED:
        await bot.send_message(chat_id=chat_id, text="Payment processing coming post-hackathon...")
    else:
        if is_pro(user_id):
            await bot.send_message(chat_id=chat_id, text="You're already Pro ✅")
        else:
            activate_pro(user_id, days=999)
            await bot.send_message(
                chat_id=chat_id,
                text="✅ DEMO MODE: Pro activated for hackathon judges\n\nAll features unlocked.\nTry /scan or /best now."
            )

async def handle_sell(chat_id, user_id):
    if not is_pro(user_id):
        await bot.send_message(chat_id=chat_id, text="You're on Free plan. Nothing to cancel.")
    else:
        users_db[user_id]["plan"] = "free"
        users_db[user_id]["pro_expires"] = None
        await bot.send_message(
            chat_id=chat_id,
            text="✅ DEMO: Pro subscription cancelled\n\nBack to Free plan.\nRe-upgrade: /buy"
        )

async def handle_message(chat_id, text, user_id):
    if not bot:
        return

    if text == "/start":
        await scan_all()
        signals = list(cache["signals"].values())
        keyboard = [
            [InlineKeyboardButton("📊 Scan Markets", callback_data="scan_all"),
             InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard")],
            [InlineKeyboardButton("🔍 Best Signal", callback_data="best_signal")],
            [InlineKeyboardButton("📈 BTC", callback_data="BTCUSDT"),
             InlineKeyboardButton("📈 ETH", callback_data="ETHUSDT"),
             InlineKeyboardButton("📈 SOL", callback_data="SOLUSDT")],
            [InlineKeyboardButton("📈 BNB", callback_data="BNBUSDT"),
             InlineKeyboardButton("📈 XRP", callback_data="XRPUSDT")],
            [InlineKeyboardButton("📈 AVAX", callback_data="AVAXUSDT"),
             InlineKeyboardButton("📈 DOGE", callback_data="DOGEUSDT")],
            [InlineKeyboardButton("📈 TRX", callback_data="TRXUSDT"),
             InlineKeyboardButton("📈 ADA", callback_data="ADAUSDT")],
            [InlineKeyboardButton("📈 LINK", callback_data="LINKUSDT"),
             InlineKeyboardButton("💎 Upgrade", callback_data="buy_cmd")]
        ]
        regime = cache["market_regime"].upper()
        top = max(signals, key=lambda x: x.get("confidence", 0)) if signals else None

        msg = "🔮 CROO AI Oracle\n\n"
        msg += f"Market: {regime} | F&G: {cache['fear_greed']}\n"
        msg += f"Assets: {len(ASSETS)} monitored\n"
        msg += f"Entry Strategy: Zone-based (0.5-1% below/above current)\n"
        if top and top.get("confidence", 0) > 0:
            msg += f"\n🔥 Top: {top.get('asset')} {top.get('signal')} {top.get('confidence')}% ({top.get('grade')})\n"
            msg += f"Price: ${top.get('price')} | Zone: {top.get('entry_zone')}\n"
        msg += "\n/scan /best /leaderboard /stats /why BTC"
        await bot.send_message(chat_id=chat_id, text=msg, reply_markup=InlineKeyboardMarkup(keyboard))

    elif text in ["/scan", "/signals"]:
        await send_leaderboard(chat_id)

    elif text == "/best":
        await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            await bot.send_message(chat_id=chat_id, text="No signals yet. Scanning...")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)))

    elif text == "/leaderboard":
        await send_leaderboard(chat_id)

    elif text == "/stats":
        await send_stats(chat_id)

    elif text == "/buy":
        await handle_buy(chat_id, user_id)

    elif text == "/sell":
        await handle_sell(chat_id, user_id)

    elif text.startswith("/why"):
        parts = text.split()
        if len(parts) > 1:
            symbol = parts[1].upper()
            await send_why(chat_id, symbol)

    elif text == "/demo":
        demo_data = await demo()
        await bot.send_message(chat_id=chat_id, text=f"📊 DEMO STATUS\n\n{json.dumps(demo_data, indent=2)}")

async def send_why(chat_id, symbol):
    asset = symbol.upper() + "USDT"
    await scan_all()
    signal = cache["signals"].get(asset, {})
    if not signal:
        await bot.send_message(chat_id=chat_id, text=f"No data for {symbol}")
        return

    msg = f"🧠 WHY {symbol}?\n\n"
    msg += f"Decision: {signal.get('signal')}\n"
    msg += f"Confidence: {signal.get('confidence')}%\n"
    msg += f"Risk: {signal.get('risk')}\n"
    msg += f"Entry Zone: {signal.get('entry_zone')}\n"
    msg += f"R:R: {signal.get('risk_reward')}\n\n"

    if signal.get('direction') == 'LONG':
        msg += "Bullish:\n" + "\n".join([f"✅ {r}" for r in signal.get('bullish_reasons', [])[:5]])
    else:
        msg += "Bearish:\n" + "\n".join([f"✅ {r}" for r in signal.get('bearish_reasons', [])[:5]])

    if signal.get('missing_conditions'):
        msg += f"\n\nMissing:\n" + "\n".join([f"❌ {m}" for m in signal.get('missing_conditions', [])[:3]])

    msg += f"\n\nMarket: {signal.get('market_regime', '').upper()} | F&G: {signal.get('fear_greed')}"
    msg += f"\nPosition Sizing: {signal.get('position_sizing', 'N/A')}"
    await bot.send_message(chat_id=chat_id, text=msg)

async def send_leaderboard(chat_id):
    await scan_all()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    msg = f"🏆 LEADERBOARD | {cache['market_regime'].upper()} | F&G: {cache['fear_greed']}\n\n"
    for i, s in enumerate(signals[:10], 1):
        msg += f"{i}. {s.get('asset','N/A')} - {s.get('confidence',0)}% ({s.get('grade','N/A')}) {s.get('signal','NONE')}\n"
        msg += f" ${s.get('price',0)} | Zone: {s.get('entry_zone','N/A')}\n"
    await bot.send_message(chat_id=chat_id, text=msg)

async def send_stats(chat_id):
    await update_performance()
    win_rate = round(performance["wins"] / performance["total"] * 100, 1) if performance["total"] > 0 else 0
    msg = f"📊 AGENT STATS\n\n"
    msg += f"Total Signals: {performance['total']}\n"
    msg += f"Wins: {performance['wins']}\n"
    msg += f"Losses: {performance['losses']}\n"
    msg += f"Win Rate: {win_rate}%\n"
    msg += f"Best Asset: {agent_memory['best_asset']} ({agent_memory['best_asset_win_rate']}%)\n"
    msg += f"Market Regime: {cache['market_regime'].upper()}\n"
    msg += f"Fear & Greed: {cache['fear_greed']}\n"
    msg += f"Revenue Simulated: ${round(agent_memory['revenue_simulated'], 2)}\n"
    msg += f"Entry Strategy: Zone-based (0.5-1% below/above current)"
    await bot.send_message(chat_id=chat_id, text=msg)

async def handle_callback(chat_id, data, user_id):
    if not bot:
        return

    if data == "scan_all":
        await scan_all()
        await send_leaderboard(chat_id)

    elif data == "leaderboard":
        await send_leaderboard(chat_id)

    elif data == "best_signal":
        await scan_all()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
        if not signals:
            await bot.send_message(chat_id=chat_id, text="No signals yet. Scanning...")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)))

    elif data == "buy_cmd":
        await handle_buy(chat_id, user_id)

    elif data in ASSETS:
        await scan_all()
        s = cache["signals"].get(data, {})
        if not s or s.get("confidence", 0) == 0:
            await bot.send_message(chat_id=chat_id, text=f"No data for {data.replace('USDT','')} yet. Scanning...")
        else:
            await send_rich_card(chat_id, s)

# ==================== STARTUP / SHUTDOWN ====================
@app.on_event("startup")
async def startup_event():
    global session, scanner_task, ws_task

    load_memory()

    timeout = aiohttp.ClientTimeout(total=15)
    session = aiohttp.ClientSession(
        timeout=timeout,
        connector=aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ttl_dns_cache=300
        )
    )

    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        await bot.set_webhook(url=webhook_url)
        print(f"✅ Webhook set: {webhook_url}")

    scanner_task = asyncio.create_task(scanner_loop())
    ws_task = asyncio.create_task(start_websockets())
    await scan_all()
    print("🚀 CROO AI Oracle started successfully")
    print("📊 Entry Strategy: Zone-based (0.5-1% below/above current price)")

@app.on_event("shutdown")
async def shutdown_event():
    global session, scanner_task, ws_task, ws_tasks

    tasks = []
    if scanner_task:
        tasks.append(scanner_task)
    if ws_task:
        tasks.append(ws_task)
    for t in ws_tasks:
        tasks.append(t)

    for task in tasks:
        if task:
            task.cancel()

    await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)

    if session:
        await session.close()

    save_memory()
    print("🛑 CROO AI Oracle shutdown complete")

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
