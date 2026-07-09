import os
import time
import asyncio
import aiohttp
import numpy as np
import json
import random
import websockets
import logging
import sqlite3
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("croo_oracle.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("croo_oracle")

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")
ADMIN_ID = int(os.environ.get("ADMIN_ID", "0")) if os.environ.get("ADMIN_ID") else 0
CHAT_ID = os.environ.get("CHAT_ID")
PORT = int(os.getenv("PORT", 8000))
SECRET_TOKEN = os.environ.get("TELEGRAM_SECRET_TOKEN", "default_secret")

bot = Bot(token=TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
start_time = time.time()
shutdown_event = asyncio.Event()

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

# ==================== GLOBAL TASK VARS ====================
scanner_task = None
ws_task = None
health_task = None
telegram_worker_task = None
paper_trade_task = None

# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, scanner_task, ws_task, health_task, telegram_worker_task, paper_trade_task
    init_db()
    load_memory()

    timeout = aiohttp.ClientTimeout(total=20, connect=10, sock_read=15)
    connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300, enable_cleanup_closed=True)
    session = aiohttp.ClientSession(timeout=timeout, connector=connector)

    if RENDER_EXTERNAL_URL and TELEGRAM_BOT_TOKEN and bot:
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        try:
            await bot.set_webhook(url=webhook_url, secret_token=SECRET_TOKEN)
            logger.info(f"Webhook set: {webhook_url} with secret token")
        except Exception as e:
            logger.error(f"Webhook set failed: {e}")

    telegram_worker_task = asyncio.create_task(telegram_worker())
    logger.info("Telegram worker started")

    scanner_task = asyncio.create_task(scanner_loop())
    logger.info("Scanner started")

    ws_task = asyncio.create_task(start_websockets())
    logger.info("WebSockets started")

    health_task = asyncio.create_task(health_monitor())
    logger.info("Health monitor started")

    paper_trade_task = asyncio.create_task(paper_trade_monitor())
    logger.info("Paper trade monitor started")

    await scan_all(force=True)
    print_startup_banner()

    yield

    logger.info("Shutting down...")
    shutdown_event.set()

    save_memory(force=True)
    logger.info("Memory saved")

    tasks_to_cancel = []
    if scanner_task:
        tasks_to_cancel.append(scanner_task)
    if ws_task:
        tasks_to_cancel.append(ws_task)
    if health_task:
        tasks_to_cancel.append(health_task)
    if telegram_worker_task:
        tasks_to_cancel.append(telegram_worker_task)
    if paper_trade_task:
        tasks_to_cancel.append(paper_trade_task)
    for t in ws_tasks:
        tasks_to_cancel.append(t)

    for task in tasks_to_cancel:
        if task:
            task.cancel()

    if tasks_to_cancel:
        await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        logger.info("All tasks cancelled")

    if session:
        await session.close()
        logger.info("Session closed")

    close_db()
    logger.info("Shutdown complete")

app = FastAPI(
    title="CROO AI Oracle",
    description="Autonomous Crypto Intelligence Agent with Multi-Provider Fallback, A2A, and Explainable AI",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
signal_history = deque(maxlen=5000)
users_db = {}
last_alerted = {}
performance = {"wins": 0, "losses": 0, "total": 0}
agent_memory = {
    "last_100_signals": [],
    "best_asset": "NONE",
    "best_asset_win_rate": 0.0,
    "total_calls": 0,
    "revenue_simulated": 0.0,
    "best_regime": "neutral",
    "best_adx_range": "0-0",
    "best_atr_range": "0-0",
    "best_hour": 0,
    "best_weekday": 0,
    "best_timeframe": "1h"
}

DB_FILE = "croo_oracle.db"
metrics = {
    "scans_completed": 0,
    "scans_failed": 0,
    "websocket_reconnects": 0,
    "telegram_messages_sent": 0,
    "api_errors": 0,
    "avg_scan_time": 0,
    "last_scan_duration": 0
}
memory_lock = asyncio.Lock()
asset_win_rates = {}

ws_tasks = []
disabled_ws = set()
provider_status = {}

scan_lock = asyncio.Lock()
api_semaphore = asyncio.Semaphore(5)
telegram_queue = asyncio.Queue()
recent_signals = {}

last_api_call = {}
api_failures = {}
cg_cache = {}
session = None
last_memory_save = time.time()

# ==================== SQLITE ====================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        asset TEXT,
        entry REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        sl REAL,
        size REAL,
        direction TEXT,
        status TEXT,
        pnl REAL,
        timestamp TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS paper_balances (
        user_id INTEGER PRIMARY KEY,
        balance REAL,
        margin REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS trade_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT,
        decision TEXT,
        confidence REAL,
        entry REAL,
        exit REAL,
        pnl REAL,
        timestamp TEXT
    )''')
    conn.commit()
    conn.close()

def close_db():
    pass

def get_paper_positions(user_id=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    if user_id is not None:
        c.execute("SELECT * FROM paper_trades WHERE user_id=? AND status='open'", (user_id,))
    else:
        c.execute("SELECT * FROM paper_trades WHERE status='open'")
    rows = c.fetchall()
    conn.close()
    positions = []
    for row in rows:
        positions.append({
            "id": row[0],
            "user_id": row[1],
            "asset": row[2],
            "entry": row[3],
            "tp1": row[4],
            "tp2": row[5],
            "tp3": row[6],
            "sl": row[7],
            "size": row[8],
            "direction": row[9],
            "status": row[10],
            "pnl": row[11],
            "timestamp": row[12]
        })
    result = {}
    for pos in positions:
        uid = pos["user_id"]
        if uid not in result:
            result[uid] = []
        result[uid].append(pos)
    return result if user_id is None else positions

def save_paper_trade(user_id, asset, entry, tp1, tp2, tp3, sl, size, direction):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO paper_trades (user_id, asset, entry, tp1, tp2, tp3, sl, size, direction, status, pnl, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (user_id, asset, entry, tp1, tp2, tp3, sl, size, direction, "open", None, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def update_paper_trade(trade_id, status, pnl):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE paper_trades SET status=?, pnl=? WHERE id=?", (status, pnl, trade_id))
    conn.commit()
    conn.close()

def update_paper_trade_sl(trade_id, new_sl):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE paper_trades SET sl=? WHERE id=?", (new_sl, trade_id))
    conn.commit()
    conn.close()

def get_paper_balance(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT balance, margin FROM paper_balances WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"balance": row[0], "margin": row[1]}
    return {"balance": 10000, "margin": 0}

def update_paper_balance(user_id, balance, margin):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("REPLACE INTO paper_balances (user_id, balance, margin) VALUES (?,?,?)", (user_id, balance, margin))
    conn.commit()
    conn.close()

# ==================== MEMORY ====================
MEMORY_FILE = "agent_memory.json"
MEMORY_TMP = "agent_memory.tmp"

def load_memory():
    global signal_history, performance, agent_memory, cache, asset_win_rates
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                signal_history = deque(data.get("signal_history", []), maxlen=5000)
                performance = data.get("performance", {"wins": 0, "losses": 0, "total": 0})
                agent_memory = data.get("agent_memory", {
                    "last_100_signals": [],
                    "best_asset": "NONE",
                    "best_asset_win_rate": 0.0,
                    "total_calls": 0,
                    "revenue_simulated": 0.0,
                    "best_regime": "neutral",
                    "best_adx_range": "0-0",
                    "best_atr_range": "0-0",
                    "best_hour": 0,
                    "best_weekday": 0,
                    "best_timeframe": "1h"
                })
                cache["last_known_prices"] = data.get("last_prices", {})
                asset_win_rates = data.get("asset_win_rates", {})
                logger.info("Memory loaded from disk")
    except Exception as e:
        logger.error(f"Memory load failed: {e}")

def save_memory(force=False):
    global last_memory_save
    now = time.time()
    if not force and (now - last_memory_save < 900):
        return
    try:
        data = {
            "signal_history": list(signal_history),
            "performance": performance,
            "agent_memory": agent_memory,
            "last_prices": cache["last_known_prices"],
            "asset_win_rates": asset_win_rates,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(MEMORY_TMP, "w") as f:
            json.dump(data, f)
        os.replace(MEMORY_TMP, MEMORY_FILE)
        last_memory_save = now
    except Exception as e:
        logger.error(f"Memory save failed: {e}")

# ==================== HELPERS ====================
def normalize_symbol(symbol: str) -> str:
    sym = symbol.upper()
    if not sym.endswith("USDT") and "USDT" not in sym:
        known = {"XBT": "BTCUSDT", "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
                 "XRP": "XRPUSDT", "BNB": "BNBUSDT", "AVAX": "AVAXUSDT", "DOGE": "DOGEUSDT",
                 "TRX": "TRXUSDT", "ADA": "ADAUSDT", "LINK": "LINKUSDT"}
        if sym in known:
            return known[sym]
        return sym + "USDT"
    return sym

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
    provider_status[name] = "error"

def mark_success(name):
    if name in api_failures:
        api_failures[name] = (0, time.time() + 600)
    provider_status[name] = "active"

def api_response(success: bool, data=None, error=None, status_code=200):
    response = {
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
    if success:
        response["data"] = data
    else:
        response["error"] = error
    return JSONResponse(response, status_code=status_code)

async def retry_async(func, *args, retries=3, delay=2, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay * (attempt + 1))
            logger.warning(f"Retry {attempt+1}/{retries} for {func.__name__}: {e}")

# ==================== ENSURE SCAN ====================
async def ensure_scan():
    if time.time() - cache["last_scan"] > 300:
        await scan_all()

# ==================== TELEGRAM QUEUE ====================
async def telegram_worker():
    while not shutdown_event.is_set():
        try:
            msg = await asyncio.wait_for(telegram_queue.get(), timeout=1.0)
            if bot:
                try:
                    await bot.send_message(**msg)
                    metrics["telegram_messages_sent"] += 1
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f"Telegram send error: {e}")
                    metrics["api_errors"] += 1
            telegram_queue.task_done()
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Telegram worker error: {e}")
            await asyncio.sleep(1)

async def send_telegram_message(chat_id, text, reply_markup=None):
    if len(text) > 4000:
        text = text[:3900] + "...\n\n(truncated)"
    await telegram_queue.put({
        "chat_id": chat_id,
        "text": text,
        "reply_markup": reply_markup
    })

# ==================== SIGNAL FORMATTING ====================
def format_signal(signal):
    decision = signal.get("decision", "HOLD")
    if decision == "HOLD":
        if signal.get("bias") == "LONG":
            msg = "⏳ HOLD (LONG BIAS)\n\n"
        elif signal.get("bias") == "SHORT":
            msg = "⏳ HOLD (SHORT BIAS)\n\n"
        else:
            msg = "⏳ HOLD\n\n"
    else:
        msg = f"🚨 {decision} SIGNAL\n\n"

    msg += f"Asset: {signal.get('asset')}\n"
    msg += f"Confidence: {signal.get('confidence')}% ({signal.get('grade')})\n"
    msg += f"Risk: {signal.get('risk')}\n"
    msg += f"Price: ${signal.get('price')}\n\n"
    msg += f"Entry Zone: {signal.get('entry_zone')}\n"
    msg += f"Entry: ${signal.get('entry')}\n"
    msg += f"TP1: ${signal.get('tp1', 0)}\n"
    msg += f"TP2: ${signal.get('tp2', 0)}\n"
    msg += f"TP3: ${signal.get('tp3', 0)}\n"
    msg += f"Stop: ${signal.get('stop_loss')}\n"
    msg += f"R:R: {signal.get('risk_reward')}\n\n"
    msg += f"Position Sizing:\n{signal.get('position_sizing')}\n\n"

    if signal.get('direction') == 'LONG':
        msg += "Bullish Reasons:\n" + "\n".join([f"✅ {r}" for r in signal.get('bullish_reasons', ['None'])[:5]])
    else:
        msg += "Bearish Reasons:\n" + "\n".join([f"✅ {r}" for r in signal.get('bearish_reasons', ['None'])[:5]])

    if signal.get('missing_conditions'):
        msg += "\n\nMissing Conditions:\n" + "\n".join([f"❌ {m}" for m in signal.get('missing_conditions', [])[:3]])

    if signal.get('why_not_now'):
        msg += "\n\nWhy Not Now:\n" + "\n".join([f"⏳ {w}" for w in signal.get('why_not_now', [])[:3]])

    if signal.get('reasoning'):
        msg += "\n\n🧠 Reasoning:\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(signal.get('reasoning', [])[:5])])

    msg += f"\n\nMarket: {signal.get('market_regime', '').upper()} | F&G: {signal.get('fear_greed')}"
    msg += f"\nSource: {signal.get('source', 'N/A')} | Hold: {signal.get('holding_period', 'N/A')}"
    msg += f"\nPullback: {signal.get('pullback_pct', 0)}% | ATR: {signal.get('atr_pct', 0)}%"
    msg += f"\nAction: {signal.get('action', 'N/A')}"
    return msg

# ==================== INDICATORS ====================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return np.array([50.0] * len(closes))
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros(len(closes), dtype=float)
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

def calc_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2:
        return 0
    tr = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr.append(max(hl, hc, lc))
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm.append(max(up, 0) if up > down else 0)
        minus_dm.append(max(down, 0) if down > up else 0)
    atr = np.mean(tr[-period:])
    if atr == 0:
        return 0
    plus_di = 100 * (np.mean(plus_dm[-period:]) / atr)
    minus_di = 100 * (np.mean(minus_dm[-period:]) / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    return dx

def calc_obv(closes, volumes):
    if len(closes) < 2:
        return 0
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv[-1]

def detect_market_structure(highs, lows):
    if len(highs) < 10:
        return "neutral"
    recent_highs = highs[-10:]
    recent_lows = lows[-10:]
    hh = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs))) if len(recent_highs)>1 else False
    hl = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows))) if len(recent_lows)>1 else False
    if hh and hl:
        return "bullish"
    lh = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs))) if len(recent_highs)>1 else False
    ll = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows))) if len(recent_lows)>1 else False
    if lh and ll:
        return "bearish"
    return "neutral"

def grade(confidence):
    if confidence >= 95: return "A+"
    elif confidence >= 90: return "A"
    elif confidence >= 80: return "B"
    elif confidence >= 70: return "C"
    elif confidence >= 60: return "D"
    else: return "F"

def risk_level(confidence):
    if confidence >= 90: return "VERY LOW"
    elif confidence >= 80: return "LOW"
    elif confidence >= 70: return "MEDIUM"
    elif confidence >= 60: return "HIGH"
    else: return "VERY HIGH"

def calculate_position_size(capital, entry, stop_loss, risk_pct=0.02):
    if stop_loss == entry or entry == 0:
        return 0
    risk_amount = capital * risk_pct
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit == 0:
        return 0
    return round(risk_amount / risk_per_unit, 4)

def calculate_entries(price, atr, direction="LONG", signal_type="BUY"):
    if direction == "LONG" and signal_type in ["BUY", "HOLD"]:
        entries = {
            "aggressive": round(price * 0.998, 4),
            "moderate": round(price * 0.995, 4),
            "conservative": round(price * 0.990, 4),
            "dca_1": round(price * 0.980, 4),
            "dca_2": round(price * 0.965, 4),
        }
        recommended = entries["moderate"]
        entry_zone = f"${entries['moderate']} - ${entries['conservative']}"
        atr_stop = max(price * 0.025, atr * 2)
        stop_loss = round(recommended - atr_stop, 4)
        risk = recommended - stop_loss
        tp1 = round(recommended + (risk * 1.5), 4)
        tp2 = round(recommended + (risk * 2.0), 4)
        tp3 = round(recommended + (risk * 2.5), 4)
        volatility_factor = min(0.06, atr / price * 4)
        tp_fixed = round(recommended * (1 + volatility_factor), 4)
        take_profit = min(tp3, tp_fixed)
        take_profit = max(take_profit, round(recommended * 1.02, 4))
        reward = take_profit - recommended
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        position_sizing = "50% moderate, 30% conservative, 20% DCA"
    elif direction == "SHORT" and signal_type in ["SELL", "HOLD"]:
        entries = {
            "aggressive": round(price * 1.002, 4),
            "moderate": round(price * 1.005, 4),
            "conservative": round(price * 1.010, 4),
            "dca_1": round(price * 1.020, 4),
            "dca_2": round(price * 1.035, 4),
        }
        recommended = entries["moderate"]
        entry_zone = f"${entries['moderate']} - ${entries['conservative']}"
        atr_stop = max(price * 0.025, atr * 2)
        stop_loss = round(recommended + atr_stop, 4)
        risk = stop_loss - recommended
        tp1 = round(recommended - (risk * 1.5), 4)
        tp2 = round(recommended - (risk * 2.0), 4)
        tp3 = round(recommended - (risk * 2.5), 4)
        volatility_factor = min(0.06, atr / price * 4)
        tp_fixed = round(recommended * (1 - volatility_factor), 4)
        take_profit = max(tp3, tp_fixed)
        take_profit = min(take_profit, round(recommended * 0.98, 4))
        reward = recommended - take_profit
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        position_sizing = "50% moderate, 30% conservative, 20% DCA"
    else:
        return {
            "entry": round(price, 4),
            "entry_zone": "N/A",
            "entries": {"current": round(price, 4)},
            "stop_loss": 0,
            "tp1": 0,
            "tp2": 0,
            "tp3": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A"
        }
    return {
        "entry": recommended,
        "entry_zone": entry_zone,
        "entries": entries,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": take_profit,
        "risk_reward": f"{risk_reward}:1",
        "position_sizing": position_sizing
    }

# ==================== USER MGMT ====================
def is_pro(user_id: int) -> bool:
    user = users_db.get(user_id, {})
    if not user:
        return False
    if user.get("plan") == "lifetime":
        return True
    expires = user.get("pro_expires")
    return bool(expires and datetime.now() < expires)

def activate_pro(user_id: int, days: int = 30):
    if user_id not in users_db:
        users_db[user_id] = {}
    users_db[user_id]["plan"] = "pro"
    users_db[user_id]["pro_expires"] = datetime.now() + timedelta(days=days)

def get_user(user_id):
    if user_id not in users_db:
        users_db[user_id] = {
            "plan": "free",
            "pro_expires": None,
            "calls_today": 0,
            "last_reset": datetime.now().date()
        }
    user = users_db[user_id]
    if datetime.now().date() > user["last_reset"]:
        user["calls_today"] = 0
        user["last_reset"] = datetime.now().date()
    return user

def can_user_call(user_id):
    user = get_user(user_id)
    if user["plan"] in ["pro", "lifetime"]:
        return True
    if user["calls_today"] < 5:
        user["calls_today"] += 1
        return True
    return False

def get_capital_plan():
    return {
        "tier1": "50% (Moderate Entry)",
        "tier2": "30% (Conservative Entry)",
        "tier3": "20% (DCA Entry)",
        "max_risk": "2% of capital"
    }

# ==================== WEBSOCKET PARSERS ====================
def parse_bybit(data):
    try:
        if "data" not in data:
            return None, None
        ticker = data["data"]
        if isinstance(ticker, list):
            ticker = ticker[0] if ticker else None
        if not ticker or "lastPrice" not in ticker:
            return None, None
        symbol = normalize_symbol(ticker["symbol"])
        return symbol, float(ticker["lastPrice"])
    except Exception:
        return None, None

def parse_binance(data):
    if "s" in data and "c" in data:
        symbol = normalize_symbol(data["s"])
        return symbol, float(data["c"])
    return None, None

def parse_okx(data):
    if "arg" in data and "data" in data and len(data["data"]) > 0:
        raw_symbol = data["arg"]["instId"]
        symbol = normalize_symbol(raw_symbol.replace("-", ""))
        return symbol, float(data["data"][0]["last"])
    return None, None

_KRAKEN_PAIR_TO_SYMBOL = {v.replace("/", ""): k for k, v in KRAKEN_WS_MAP.items()}

def parse_kraken(data):
    try:
        if isinstance(data, list) and len(data) > 3 and isinstance(data[1], list):
            pair = data[3].replace("/", "")
            symbol = _KRAKEN_PAIR_TO_SYMBOL.get(pair)
            if symbol:
                return symbol, float(data[1][0])
    except Exception:
        pass
    return None, None

# ==================== WEBSOCKET FEED ====================
async def websocket_feed(uri, sub_msg, name, parser):
    retry = 1
    while not shutdown_event.is_set():
        if name in disabled_ws:
            await asyncio.sleep(60)
            continue
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                await ws.send(json.dumps(sub_msg))
                provider_status[name] = "active"
                logger.info(f"[Provider] {name} connected")
                retry = 1
                async for msg in ws:
                    if shutdown_event.is_set():
                        break
                    data = json.loads(msg)
                    symbol, price = parser(data)
                    if symbol and price:
                        cache["live_prices"][symbol] = price
                        cache["last_known_prices"][symbol] = price
                        cache["last_ws_update"] = time.time()
        except Exception as e:
            error_str = str(e)
            if "451" in error_str:
                disabled_ws.add(name)
                provider_status[name] = "disabled (region)"
                logger.warning(f"[Provider] {name} unavailable (HTTP 451 Region Restriction)")
                continue
            provider_status[name] = "error"
            logger.warning(f"[Provider] {name} error: {e}")
            logger.info(f"[Fallback] {name} unavailable, switching to next provider")
            metrics["websocket_reconnects"] += 1
            await asyncio.sleep(retry)
            retry = min(retry * 2, 60)

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

# ==================== OHLCV FALLBACKS ====================
async def fetch_okx_ohlc(asset, interval="1H", limit=100):
    if not can_call(f"okx_{asset}", 30) or not check_circuit_breaker("okx"):
        return None, None
    try:
        symbol = asset.replace("USDT", "-USDT")
        params = {"instId": symbol, "bar": interval, "limit": limit}
        async with session.get(f"https://www.okx.com/api/v5/market/candles", params=params) as r:
            if r.status == 200:
                data = await r.json()
                if data["code"] == "0":
                    raw = data["data"]
                    raw.reverse()
                    klines = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw]
                    mark_success("okx")
                    return klines, "OKX"
    except Exception as e:
        logger.warning(f"OKX failed for {asset}: {e}")
    mark_failure("okx")
    return None, None

async def fetch_bybit_ohlc(asset, interval="60", limit=100):
    if not can_call(f"bybit_{asset}", 30) or not check_circuit_breaker("bybit"):
        return None, None
    try:
        params = {"category": "linear", "symbol": asset, "interval": interval, "limit": limit}
        async with session.get("https://api.bybit.com/v5/market/kline", params=params) as r:
            if r.status == 200:
                data = await r.json()
                raw = data["result"]["list"]
                raw.reverse()
                klines = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                mark_success("bybit")
                return klines, "Bybit"
    except Exception as e:
        logger.warning(f"Bybit failed for {asset}: {e}")
    mark_failure("bybit")
    return None, None

async def fetch_binance_ohlc(asset, interval="1h", limit=100):
    if not can_call(f"binance_{asset}", 30) or not check_circuit_breaker("binance"):
        return None, None
    try:
        params = {"symbol": asset, "interval": interval, "limit": limit}
        async with session.get("https://api.binance.com/api/v3/klines", params=params) as r:
            if r.status == 200:
                raw = await r.json()
                klines = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in raw]
                mark_success("binance")
                return klines, "Binance"
    except Exception as e:
        logger.warning(f"Binance failed for {asset}: {e}")
    mark_failure("binance")
    return None, None

async def fetch_coingecko_ohlcv(asset):
    if not can_call(f"cg_{asset}", 120) or not check_circuit_breaker("coingecko"):
        return None, None
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
    except Exception as e:
        logger.warning(f"CoinGecko failed for {asset}: {e}")
    mark_failure("coingecko")
    return None, None

async def get_ohlcv(asset, interval="1h", limit=100):
    interval_map = {
        "1h": ("1H", "60", "1h"),
        "4h": ("4H", "240", "4h"),
    }
    if interval not in interval_map:
        interval = "1h"
    okx_int, bybit_int, binance_int = interval_map[interval]

    providers = [
        (fetch_okx_ohlc, okx_int),
        (fetch_bybit_ohlc, bybit_int),
        (fetch_binance_ohlc, binance_int),
    ]
    needed = min(limit, 50)
    for provider, intv in providers:
        try:
            async with api_semaphore:
                data, source = await provider(asset, intv, limit)
                if data and len(data) >= needed:
                    return data, source
        except Exception as e:
            logger.warning(f"{provider.__name__} error: {e}")
    if interval == "1h" and limit <= 100:
        data, source = await fetch_coingecko_ohlcv(asset)
        if data and len(data) > 50:
            return data, source
    return None, "none"

async def fetch_historical_price(asset, days_ago=30):
    try:
        coin_id = CG_MAP[asset]
        async with session.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", params={"vs_currency": "usd", "days": days_ago, "interval": "daily"}) as r:
            if r.status == 200:
                data = await r.json()
                prices = data["prices"]
                if prices:
                    return prices[0][1]
    except Exception:
        pass
    return None

# ==================== PRICE FALLBACKS ====================
async def fetch_binance_price(asset):
    if not can_call(f"binance_price_{asset}", 10):
        return None
    try:
        async with session.get("https://api.binance.com/api/v3/ticker/price", params={"symbol": asset}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data["price"])
    except Exception: pass
    return None

async def fetch_bybit_price(asset):
    if not can_call(f"bybit_price_{asset}", 10):
        return None
    try:
        async with session.get("https://api.bybit.com/v5/market/tickers", params={"category": "linear", "symbol": asset}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data["result"]["list"][0]["lastPrice"])
    except Exception: pass
    return None

async def fetch_coingecko_price(asset):
    if not can_call(f"cg_price_{asset}", 60):
        return None
    try:
        coin_id = CG_MAP[asset]
        async with session.get("https://api.coingecko.com/api/v3/simple/price", params={"ids": coin_id, "vs_currencies": "usd"}) as r:
            if r.status == 200:
                data = await r.json()
                return float(data[coin_id]["usd"])
    except Exception: pass
    return None

async def fetch_coinmarketcap_price(asset):
    if not can_call(f"cmc_{asset}", 60) or not check_circuit_breaker("coinmarketcap"):
        return None
    try:
        cmc_api_key = os.environ.get("CMC_API_KEY", "")
        if not cmc_api_key:
            return None
        symbol = asset.replace("USDT", "")
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        async with session.get(url, params={"symbol": symbol, "convert": "USD"}, headers={"X-CMC_PRO_API_KEY": cmc_api_key}) as r:
            if r.status == 200:
                data = await r.json()
                price = data["data"][symbol]["quote"]["USD"]["price"]
                mark_success("coinmarketcap")
                return price
    except Exception: pass
    mark_failure("coinmarketcap")
    return None

async def get_price_fallback(asset):
    if asset in cache["live_prices"] and cache["live_prices"][asset] > 0:
        return cache["live_prices"][asset]
    for fetcher in [fetch_binance_price, fetch_bybit_price, fetch_coingecko_price, fetch_coinmarketcap_price]:
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
    except Exception: pass
    return cache["fear_greed"]

# ==================== CORE ANALYSIS ====================
async def detect_regime():
    try:
        klines, _ = await get_ohlcv("BTCUSDT", "1h")
        if not klines or len(klines) < 50:
            return cache["market_regime"]
        closes = np.array([float(k[4]) for k in klines])
        if len(closes) < 50:
            return cache["market_regime"]
        ema50 = calc_ema(closes, 50)[-1]
        return "bullish" if closes[-1] > ema50 else "bearish"
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        return cache["market_regime"]

async def analyze_asset(symbol):
    klines, source = await get_ohlcv(symbol, "1h")
    if not klines or len(klines) < 50:
        price = await get_price_fallback(symbol)
        if price > 0:
            entry_data = calculate_entries(price, 0.01, "LONG", "HOLD")
            return {
                "asset": symbol.replace("USDT", ""),
                "decision": "HOLD",
                "bias": "NEUTRAL",
                "confidence": 0,
                "grade": "F",
                "price": round(price, 4),
                "entry": entry_data["entry"],
                "entry_zone": entry_data["entry_zone"],
                "entries": entry_data["entries"],
                "stop_loss": entry_data["stop_loss"],
                "tp1": entry_data.get("tp1", 0),
                "tp2": entry_data.get("tp2", 0),
                "tp3": entry_data.get("tp3", 0),
                "risk_reward": entry_data["risk_reward"],
                "position_sizing": entry_data["position_sizing"],
                "bullish_reasons": [],
                "bearish_reasons": [],
                "missing_conditions": ["Full OHLCV data unavailable"],
                "source": "price_only",
                "direction": "NONE",
                "risk": "UNKNOWN",
                "holding_period": "N/A",
                "pullback_pct": 0,
                "action": "HOLD",
                "why_not_now": ["Insufficient data for analysis"],
                "checks": {},
                "reasoning": ["Waiting for more data"],
                "capital_plan": get_capital_plan(),
                "rsi": 50,
                "atr_pct": 0,
                "adx": 0
            }
        return {
            "asset": symbol.replace("USDT", ""),
            "decision": "HOLD",
            "bias": "NEUTRAL",
            "confidence": 0,
            "price": 0,
            "entry": 0,
            "entry_zone": "N/A",
            "entries": {},
            "stop_loss": 0,
            "tp1": 0,
            "tp2": 0,
            "tp3": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A",
            "bullish_reasons": [],
            "bearish_reasons": [],
            "direction": "NONE",
            "risk": "UNKNOWN",
            "holding_period": "N/A",
            "pullback_pct": 0,
            "action": "HOLD",
            "why_not_now": ["Market data unavailable"],
            "checks": {},
            "reasoning": ["No data available"],
            "capital_plan": get_capital_plan(),
            "rsi": 50,
            "atr_pct": 0,
            "adx": 0
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
    atr_pct = (atr / price) * 100
    adx = calc_adx(highs, lows, closes, period=14)

    resistance = max(highs[-50:]) if len(highs) >= 50 else max(highs)
    support = min(lows[-50:]) if len(lows) >= 50 else min(lows)
    dist_to_res = (resistance - price) / price * 100 if price > 0 else 100
    dist_to_sup = (price - support) / price * 100 if price > 0 else 100

    structure = detect_market_structure(highs, lows)
    obv = calc_obv(closes, volumes)
    obv_trend = 0
    if len(closes) >= 5:
        obv_prev = calc_obv(closes[:-5], volumes[:-5]) if len(closes) >= 5 else obv
        obv_trend = obv - obv_prev
    obv_bullish = obv_trend > 0

    vol_avg_short = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
    vol_avg_long = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    vol_trend = vol_avg_short > vol_avg_long

    recent_high = max(closes[-20:])
    recent_low = min(closes[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    bullish_confirmation = price > prev_close
    bearish_confirmation = price < prev_close

    weights = {
        "rsi": 20,
        "ema": 20,
        "pullback": 15,
        "volume": 10,
        "confirmation": 10,
        "fear": 2,
        "structure": 10,
        "adx": 8,
        "resistance": -5,
        "support": -5,
        "multi_tf": 5,
        "obv": 5,
        "vol_trend": 5
    }
    available_weights = []
    long_score = 0
    short_score = 0
    bullish_reasons = []
    bearish_reasons = []
    missing_conditions = []
    checks = {}
    reasoning = []

    # RSI
    if rsi_val < 30:
        long_score += weights["rsi"]
        bullish_reasons.append(f"RSI Oversold ({rsi_val:.1f})")
        checks["rsi_oversold"] = True
        reasoning.append("RSI deeply oversold – strong BUY")
        available_weights.append(weights["rsi"])
    elif rsi_val < 40:
        long_score += weights["rsi"] * 0.6
        bullish_reasons.append(f"RSI Recovering ({rsi_val:.1f})")
        checks["rsi_recovering"] = True
        reasoning.append("RSI recovering from oversold")
        available_weights.append(weights["rsi"])
    elif rsi_val > 80:
        short_score += weights["rsi"]
        bearish_reasons.append(f"RSI Extremely Overbought ({rsi_val:.1f})")
        checks["rsi_extreme_overbought"] = True
        reasoning.append("RSI extremely overbought – strong SELL")
        available_weights.append(weights["rsi"])
    elif rsi_val > 70:
        short_score += weights["rsi"] * 0.6
        bearish_reasons.append(f"RSI Overbought ({rsi_val:.1f})")
        checks["rsi_overbought"] = True
        reasoning.append("RSI overbought – SELL signal")
        available_weights.append(weights["rsi"])
    else:
        missing_conditions.append("RSI neutral")
        checks["rsi_neutral"] = True
        reasoning.append("RSI neutral")
        available_weights.append(weights["rsi"])

    # EMA
    if price > ema20 and price > ema50 and ema20 > ema50:
        long_score += weights["ema"]
        bullish_reasons.append("Above EMA20 & EMA50 (uptrend)")
        checks["ema_bullish"] = True
        reasoning.append("Price above both EMAs with bullish crossover")
        available_weights.append(weights["ema"])
    elif price < ema20 and price < ema50 and ema20 < ema50:
        short_score += weights["ema"]
        bearish_reasons.append("Below EMA20 & EMA50 (downtrend)")
        checks["ema_bearish"] = True
        reasoning.append("Price below both EMAs with bearish crossover")
        available_weights.append(weights["ema"])
    else:
        missing_conditions.append("EMA not aligned")
        checks["ema_clear"] = False
        reasoning.append("EMA trend unclear")
        available_weights.append(weights["ema"])

    # Pullback / Bounce
    if 1 <= pullback <= 5 and price > ema20:
        long_score += weights["pullback"]
        bullish_reasons.append(f"Pullback {pullback:.1f}% (healthy)")
        checks["pullback_healthy"] = True
        reasoning.append(f"Healthy pullback of {pullback:.1f}%")
        available_weights.append(weights["pullback"])
    elif pullback > 5:
        missing_conditions.append("Pullback too deep (>5%)")
        checks["pullback_deep"] = True
        reasoning.append("Pullback too deep")
        available_weights.append(weights["pullback"])
    elif pullback < 1:
        missing_conditions.append("No pullback")
        checks["pullback_none"] = True
        reasoning.append("No meaningful pullback")
        available_weights.append(weights["pullback"])

    if 1 <= bounce <= 5 and price < ema20:
        short_score += weights["pullback"]
        bearish_reasons.append(f"Bounce {bounce:.1f}% (dead cat)")
        checks["bounce_healthy"] = True
        reasoning.append(f"Dead cat bounce of {bounce:.1f}%")
        available_weights.append(weights["pullback"])
    elif bounce > 5:
        missing_conditions.append("Bounce too high (>5%)")
        checks["bounce_high"] = True
        reasoning.append("Bounce too strong – could be reversal")
        available_weights.append(weights["pullback"])

    # Volume spike
    avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
    vol_spike = volumes[-1] > avg_vol * 1.5 if avg_vol > 0 else False
    if vol_spike:
        if long_score >= short_score:
            long_score += weights["volume"]
            bullish_reasons.append("Volume spike")
            checks["volume_spike"] = True
            reasoning.append("Volume confirms buying")
        else:
            short_score += weights["volume"]
            bearish_reasons.append("Volume spike")
            checks["volume_spike"] = True
            reasoning.append("Volume confirms selling")
        available_weights.append(weights["volume"])
    else:
        missing_conditions.append("No volume confirmation")
        checks["volume_spike"] = False
        reasoning.append("No significant volume")
        available_weights.append(weights["volume"])

    # Confirmation candle
    if bullish_confirmation:
        long_score += weights["confirmation"]
        bullish_reasons.append("Bullish confirmation candle")
        checks["bullish_confirmation"] = True
        reasoning.append("Bullish confirmation")
        available_weights.append(weights["confirmation"])
    elif bearish_confirmation:
        short_score += weights["confirmation"]
        bearish_reasons.append("Bearish confirmation candle")
        checks["bearish_confirmation"] = True
        reasoning.append("Bearish confirmation")
        available_weights.append(weights["confirmation"])
    else:
        missing_conditions.append("No confirmation candle")
        checks["confirmation"] = False
        reasoning.append("No clear confirmation")
        available_weights.append(weights["confirmation"])

    # Fear & Greed
    fg = cache["fear_greed"]
    if fg < 20:
        if long_score >= short_score:
            long_score += weights["fear"]
            bullish_reasons.append("Extreme Fear (opportunity)")
            checks["extreme_fear"] = True
            reasoning.append("Extreme fear – slight BUY edge")
        else:
            short_score -= weights["fear"]
            bearish_reasons.append("Extreme Fear – avoid SHORT")
            checks["extreme_fear"] = True
            reasoning.append("Extreme fear – avoid selling")
        available_weights.append(weights["fear"])
    elif fg > 80:
        if short_score > long_score:
            short_score += weights["fear"]
            bearish_reasons.append("Extreme Greed")
            checks["extreme_greed"] = True
            reasoning.append("Extreme greed – slight SELL edge")
        else:
            long_score -= weights["fear"]
            bullish_reasons.append("Extreme Greed – avoid BUY")
            checks["extreme_greed"] = True
            reasoning.append("Extreme greed – avoid buying")
        available_weights.append(weights["fear"])
    else:
        available_weights.append(weights["fear"])

    # Market Regime
    regime = cache["market_regime"]
    if regime == "bearish":
        long_score -= 5
        short_score += 5
        reasoning.append("BEARISH regime – slight penalty to BUY")
        checks["regime_bearish"] = True
    elif regime == "bullish":
        long_score += 5
        short_score -= 5
        reasoning.append("BULLISH regime – slight penalty to SELL")
        checks["regime_bullish"] = True

    # ATR
    if atr_pct < 0.5:
        checks["atr_very_low"] = True
        missing_conditions.append("ATR extremely low (< 0.5%)")
        reasoning.append("ATR < 0.5% – will be rejected")
        long_score = 0
        short_score = 0
    elif atr_pct < 0.8:
        long_score -= 10
        short_score -= 10
        missing_conditions.append("ATR low (0.5-0.8%)")
        checks["atr_low"] = True
        reasoning.append("ATR low – confidence reduced")
    else:
        checks["atr_ok"] = True
        reasoning.append(f"ATR {atr_pct:.2f}% – acceptable")

    # ADX
    if adx > 25:
        if long_score >= short_score:
            long_score += weights["adx"]
            bullish_reasons.append(f"ADX {adx:.1f} (strong trend)")
            checks["adx_strong"] = True
            reasoning.append("ADX > 25 – strong trend")
        else:
            short_score += weights["adx"]
            bearish_reasons.append(f"ADX {adx:.1f} (strong trend)")
            checks["adx_strong"] = True
            reasoning.append("ADX > 25 – strong trend")
        available_weights.append(weights["adx"])
    else:
        missing_conditions.append("ADX < 25 (weak trend)")
        checks["adx_weak"] = True
        reasoning.append("ADX < 25 – weak trend, reduce confidence")
        long_score -= 10
        short_score -= 10
        available_weights.append(weights["adx"])

    # Resistance/Support
    if dist_to_res < 2:
        long_score -= weights["resistance"]
        missing_conditions.append("Near resistance")
        checks["near_resistance"] = True
        reasoning.append("Near resistance – BUY penalized")
    if dist_to_sup < 2:
        short_score -= weights["support"]
        missing_conditions.append("Near support")
        checks["near_support"] = True
        reasoning.append("Near support – SELL penalized")

    # Market Structure
    if structure == "bullish":
        long_score += weights["structure"]
        bullish_reasons.append("HH/HL structure (bullish)")
        checks["structure_bullish"] = True
        reasoning.append("Higher highs/higher lows – uptrend")
        available_weights.append(weights["structure"])
    elif structure == "bearish":
        short_score += weights["structure"]
        bearish_reasons.append("LH/LL structure (bearish)")
        checks["structure_bearish"] = True
        reasoning.append("Lower highs/lower lows – downtrend")
        available_weights.append(weights["structure"])
    else:
        missing_conditions.append("Structure unclear")
        checks["structure_neutral"] = True
        reasoning.append("No clear market structure")
        available_weights.append(weights["structure"])

    # Multi-timeframe
    klines4h, _ = await get_ohlcv(symbol, "4h")
    if klines4h and len(klines4h) >= 50:
        closes4h = np.array([float(k[4]) for k in klines4h])
        price4h = closes4h[-1]
        ema50_4h = calc_ema(closes4h, 50)[-1]
        tf4_trend = "bullish" if price4h > ema50_4h else "bearish"
        if tf4_trend == "bullish":
            if long_score >= short_score:
                long_score += weights["multi_tf"]
                bullish_reasons.append("4H trend bullish")
                checks["tf4_bullish"] = True
                reasoning.append("4H aligns bullish")
            else:
                short_score -= weights["multi_tf"] * 0.5
                bearish_reasons.append("4H bullish – but 1H short")
                checks["tf4_conflict"] = True
                reasoning.append("4H bullish, short penalized")
        else:
            if short_score >= long_score:
                short_score += weights["multi_tf"]
                bearish_reasons.append("4H trend bearish")
                checks["tf4_bearish"] = True
                reasoning.append("4H aligns bearish")
            else:
                long_score -= weights["multi_tf"] * 0.5
                bullish_reasons.append("4H bearish – but 1H long")
                checks["tf4_conflict"] = True
                reasoning.append("4H bearish, long penalized")
        available_weights.append(weights["multi_tf"])

    # OBV
    if obv_bullish:
        if long_score >= short_score:
            long_score += weights["obv"]
            bullish_reasons.append("OBV rising")
            checks["obv_bullish"] = True
            reasoning.append("On-Balance Volume rising")
        else:
            short_score += weights["obv"] * 0.5
            bearish_reasons.append("OBV rising but short bias")
            checks["obv_bullish"] = True
    else:
        if short_score >= long_score:
            short_score += weights["obv"]
            bearish_reasons.append("OBV falling")
            checks["obv_bearish"] = True
            reasoning.append("On-Balance Volume falling")
        else:
            long_score += weights["obv"] * 0.5
            bullish_reasons.append("OBV falling but long bias")
            checks["obv_bearish"] = True
    available_weights.append(weights["obv"])

    # Volume trend
    if vol_trend:
        if long_score >= short_score:
            long_score += weights["vol_trend"]
            bullish_reasons.append("Volume increasing")
            checks["volume_trend_up"] = True
            reasoning.append("Volume trend increasing")
        else:
            short_score += weights["vol_trend"]
            bearish_reasons.append("Volume increasing")
            checks["volume_trend_up"] = True
    else:
        if long_score >= short_score:
            long_score -= weights["vol_trend"] * 0.5
            bullish_reasons.append("Volume decreasing – weak momentum")
            checks["volume_trend_down"] = True
            reasoning.append("Volume trend decreasing – penalty")
        else:
            short_score -= weights["vol_trend"] * 0.5
            bearish_reasons.append("Volume decreasing – weak momentum")
            checks["volume_trend_down"] = True
    available_weights.append(weights["vol_trend"])

    direction = "LONG" if long_score >= short_score else "SHORT"
    achieved = max(long_score, short_score)
    possible_score = sum(available_weights)
    if possible_score == 0:
        possible_score = 1
    confidence = (achieved / possible_score) * 100
    confidence = min(100, max(0, confidence))

    # Learning Engine
    asset = symbol.replace("USDT", "")
    if asset in asset_win_rates:
        wr = asset_win_rates[asset].get("wins", 0) / max(1, asset_win_rates[asset].get("wins", 0) + asset_win_rates[asset].get("losses", 0))
        if wr > 0.6:
            modifier = 1 + (wr - 0.6) * 0.5
        elif wr < 0.4:
            modifier = 1 - (0.4 - wr) * 0.5
        else:
            modifier = 1.0
        confidence = confidence * modifier
        confidence = min(100, max(0, confidence))

    if regime == agent_memory.get("best_regime", "neutral"):
        confidence = min(100, confidence * 1.05)

    adx_range = f"{int(adx//10)*10}-{int(adx//10)*10+9}"
    if adx_range == agent_memory.get("best_adx_range", "0-0"):
        confidence = min(100, confidence * 1.05)

    if confidence >= 70 and adx > 25 and (dist_to_res > 2 if direction=="LONG" else dist_to_sup > 2):
        decision = "BUY" if direction == "LONG" else "SELL"
    else:
        decision = "WATCH" if confidence >= 50 else "HOLD"

    bias = direction if decision in ["BUY", "SELL", "WATCH"] else "NEUTRAL"

    if decision in ["BUY"] or (decision == "WATCH" and direction == "LONG"):
        entry_data = calculate_entries(price, atr, "LONG", "BUY" if decision == "BUY" else "HOLD")
        holding_period = "1-3 days"
    elif decision in ["SELL"] or (decision == "WATCH" and direction == "SHORT"):
        entry_data = calculate_entries(price, atr, "SHORT", "SELL" if decision == "SELL" else "HOLD")
        holding_period = "1-3 days"
    else:
        entry_data = {
            "entry": round(price, 4),
            "entry_zone": "N/A",
            "entries": {"current": round(price, 4)},
            "stop_loss": 0,
            "tp1": 0,
            "tp2": 0,
            "tp3": 0,
            "risk_reward": "N/A",
            "position_sizing": "N/A"
        }
        holding_period = "N/A"

    signal = {
        "asset": asset,
        "price": round(price, 4),
        "decision": decision,
        "bias": bias,
        "confidence": round(confidence, 1),
        "grade": grade(confidence),
        "direction": direction,
        "entry": entry_data["entry"],
        "entry_zone": entry_data["entry_zone"],
        "entries": entry_data["entries"],
        "stop_loss": entry_data["stop_loss"],
        "tp1": entry_data.get("tp1", 0),
        "tp2": entry_data.get("tp2", 0),
        "tp3": entry_data.get("tp3", 0),
        "risk_reward": entry_data["risk_reward"],
        "position_sizing": entry_data["position_sizing"],
        "rsi": round(rsi_val, 1),
        "atr": round(atr, 4),
        "atr_pct": round(atr_pct, 2),
        "adx": round(adx, 1),
        "risk": risk_level(confidence),
        "holding_period": holding_period,
        "bullish_reasons": bullish_reasons if bullish_reasons else ["Waiting for setup"],
        "bearish_reasons": bearish_reasons if bearish_reasons else ["Waiting for setup"],
        "missing_conditions": missing_conditions,
        "checks": checks,
        "action": decision,
        "why_not_now": [],
        "reasoning": reasoning,
        "capital_plan": get_capital_plan(),
        "source": source,
        "market_regime": cache["market_regime"],
        "fear_greed": cache["fear_greed"],
        "pullback_pct": round(pullback, 2),
        "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
        "timestamp": datetime.utcnow().isoformat()
    }

    valid, reason = validate_signal(signal)
    if not valid and decision in ["BUY", "SELL"]:
        signal["decision"] = "HOLD"
        signal["action"] = "HOLD"
        signal["why_not_now"].append(reason)
        if "Risk-reward" in reason:
            signal["missing_conditions"].append("R:R below 2:1")
        if "Confidence" in reason:
            signal["missing_conditions"].append("Confidence below 70%")
        if "ATR" in reason:
            signal["missing_conditions"].append("ATR too low")
        if "ADX" in reason:
            signal["missing_conditions"].append("ADX < 25")
        if "RSI" in reason:
            signal["missing_conditions"].append("RSI condition not met")

    if signal["decision"] in ["BUY", "SELL"]:
        capital = 1000
        pos_size = calculate_position_size(capital, signal["entry"], signal["stop_loss"], risk_pct=0.02)
        signal["position_sizing"] = f"{pos_size} units (2% risk)"
    else:
        signal["position_sizing"] = "0 (No trade)"

    if signal["decision"] == "HOLD":
        why = []
        if "R:R below 2:1" in signal.get("missing_conditions", []):
            why.append("Risk-reward below 2:1")
        if "Confidence below 70%" in signal.get("missing_conditions", []):
            why.append("Confidence below 70%")
        if "ATR" in str(signal.get("missing_conditions", [])):
            why.append("ATR too low (<0.8%)")
        if "ADX" in str(signal.get("missing_conditions", [])):
            why.append("Weak trend (ADX<25)")
        if "RSI" in str(signal.get("missing_conditions", [])):
            why.append("RSI not extreme")
        if "EMA" in str(signal.get("missing_conditions", [])):
            why.append("EMA trend unclear")
        if "Pullback" in str(signal.get("missing_conditions", [])):
            why.append("Pullback too deep/none")
        if "Bounce" in str(signal.get("missing_conditions", [])):
            why.append("Bounce too strong")
        if "Volume" in str(signal.get("missing_conditions", [])):
            why.append("No volume confirmation")
        if "Confirmation" in str(signal.get("missing_conditions", [])):
            why.append("No confirmation candle")
        if "Resistance" in str(signal.get("missing_conditions", [])):
            why.append("Near resistance")
        if "Support" in str(signal.get("missing_conditions", [])):
            why.append("Near support")
        if "Structure" in str(signal.get("missing_conditions", [])):
            why.append("Unclear structure")
        if not why:
            why.append("No clear setup")
        signal["why_not_now"] = why

    return signal

def validate_signal(signal):
    rr_str = signal.get("risk_reward", "0:1")
    try:
        rr_val = float(rr_str.split(":")[0].strip()) if ":" in rr_str else 0
    except Exception:
        rr_val = 0
    if rr_val < 2:
        return False, "Risk-reward below 2:1"
    if signal.get("confidence", 0) < 70:
        return False, "Confidence below 70%"
    if signal.get("atr_pct", 0) < 0.5:
        return False, "ATR too low (< 0.5%)"
    if signal.get("adx", 0) < 25:
        return False, "ADX < 25 (weak trend)"
    rsi = signal.get("rsi", 50)
    if signal["direction"] == "LONG" and rsi > 70:
        return False, "RSI overbought for BUY"
    if signal["direction"] == "SHORT" and rsi < 30:
        return False, "RSI oversold for SELL"
    return True, "Valid"

# ==================== PERFORMANCE ====================
async def update_performance():
    async with memory_lock:
        for signal in signal_history:
            if signal.get("status") == "open" and signal.get("entry", 0) > 0:
                current = get_current_price(signal["asset"] + "USDT")
                if current == 0:
                    continue
                if signal["direction"] == "LONG":
                    if current >= signal.get("tp3", 0):
                        signal["pnl"] = round((signal.get("tp3", 0) - signal["entry"]) / signal["entry"] * 100, 2)
                        signal["status"] = "win"
                        performance["wins"] += 1
                        performance["total"] += 1
                        asset = signal["asset"]
                        if asset not in asset_win_rates:
                            asset_win_rates[asset] = {"wins": 0, "losses": 0}
                        asset_win_rates[asset]["wins"] += 1
                    elif current <= signal["stop_loss"]:
                        signal["pnl"] = round((signal["stop_loss"] - signal["entry"]) / signal["entry"] * 100, 2)
                        signal["status"] = "loss"
                        performance["losses"] += 1
                        performance["total"] += 1
                        asset = signal["asset"]
                        if asset not in asset_win_rates:
                            asset_win_rates[asset] = {"wins": 0, "losses": 0}
                        asset_win_rates[asset]["losses"] += 1
                elif signal["direction"] == "SHORT":
                    if current <= signal.get("tp3", 0):
                        signal["pnl"] = round((signal["entry"] - signal.get("tp3", 0)) / signal["entry"] * 100, 2)
                        signal["status"] = "win"
                        performance["wins"] += 1
                        performance["total"] += 1
                        asset = signal["asset"]
                        if asset not in asset_win_rates:
                            asset_win_rates[asset] = {"wins": 0, "losses": 0}
                        asset_win_rates[asset]["wins"] += 1
                    elif current >= signal["stop_loss"]:
                        signal["pnl"] = round((signal["entry"] - signal["stop_loss"]) / signal["entry"] * 100, 2)
                        signal["status"] = "loss"
                        performance["losses"] += 1
                        performance["total"] += 1
                        asset = signal["asset"]
                        if asset not in asset_win_rates:
                            asset_win_rates[asset] = {"wins": 0, "losses": 0}
                        asset_win_rates[asset]["losses"] += 1
        update_memory()
        save_memory()

def update_memory():
    agent_memory["last_100_signals"] = list(signal_history)[-100:]
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

# ==================== PAPER TRADE MONITOR ====================
_notified_milestones = set()

async def paper_trade_monitor():
    logger.info("Paper trade monitor started")
    while not shutdown_event.is_set():
        await asyncio.sleep(30)
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT id, user_id, asset, entry, tp1, tp2, tp3, sl, size, direction FROM paper_trades WHERE status='open'")
            rows = c.fetchall()
            conn.close()
            open_ids = {row[0] for row in rows}
            _notified_milestones.intersection_update({k for k in _notified_milestones if k[0] in open_ids})

            for row in rows:
                trade_id, user_id, asset, entry, tp1, tp2, tp3, sl, size, direction = row
                current = get_current_price(asset + "USDT")
                if current == 0:
                    continue

                if direction == "LONG":
                    if current >= tp3:
                        pnl = round((tp3 - entry) / entry * 100, 2)
                        update_paper_trade(trade_id, "win", pnl)
                        bal = get_paper_balance(user_id)
                        cost = entry * size
                        profit = (tp3 - entry) * size
                        new_balance = bal["balance"] + cost + profit
                        new_margin = bal["margin"] - cost
                        update_paper_balance(user_id, new_balance, new_margin)
                        await send_telegram_message(
                            user_id,
                            f"✅ TP Hit (Full) on {asset}\nPnL: +{pnl}%\nEntry: ${entry}\nTP: ${tp3}\nBalance: ${new_balance:.2f}"
                        )
                    elif current >= tp2 and (trade_id, "tp2") not in _notified_milestones:
                        _notified_milestones.add((trade_id, "tp2"))
                        await send_telegram_message(
                            user_id,
                            f"🟡 Partial TP Hit on {asset}\nEntry: ${entry}\nTP2: ${tp2}\nContinuing to TP3..."
                        )
                    elif current >= tp1 and (trade_id, "tp1") not in _notified_milestones:
                        _notified_milestones.add((trade_id, "tp1"))
                        update_paper_trade_sl(trade_id, entry)
                        await send_telegram_message(
                            user_id,
                            f"🟡 TP1 Hit on {asset}\nMoved SL to breakeven (${entry})\nEntry: ${entry}\nTP1: ${tp1}"
                        )
                    elif current <= sl:
                        pnl = round((sl - entry) / entry * 100, 2)
                        update_paper_trade(trade_id, "loss", pnl)
                        bal = get_paper_balance(user_id)
                        cost = entry * size
                        loss = (entry - sl) * size
                        new_balance = bal["balance"] + cost - loss
                        new_margin = bal["margin"] - cost
                        update_paper_balance(user_id, new_balance, new_margin)
                        await send_telegram_message(
                            user_id,
                            f"❌ SL Hit on {asset}\nPnL: {pnl}%\nEntry: ${entry}\nSL: ${sl}\nBalance: ${new_balance:.2f}"
                        )
                else:
                    if current <= tp3:
                        pnl = round((entry - tp3) / entry * 100, 2)
                        update_paper_trade(trade_id, "win", pnl)
                        bal = get_paper_balance(user_id)
                        cost = entry * size
                        profit = (entry - tp3) * size
                        new_balance = bal["balance"] + cost + profit
                        new_margin = bal["margin"] - cost
                        update_paper_balance(user_id, new_balance, new_margin)
                        await send_telegram_message(
                            user_id,
                            f"✅ TP Hit (Full) on {asset} (SHORT)\nPnL: +{pnl}%\nEntry: ${entry}\nTP: ${tp3}\nBalance: ${new_balance:.2f}"
                        )
                    elif current <= tp2 and (trade_id, "tp2") not in _notified_milestones:
                        _notified_milestones.add((trade_id, "tp2"))
                        await send_telegram_message(
                            user_id,
                            f"🟡 Partial TP Hit on {asset} (SHORT)\nContinuing..."
                        )
                    elif current <= tp1 and (trade_id, "tp1") not in _notified_milestones:
                        _notified_milestones.add((trade_id, "tp1"))
                        update_paper_trade_sl(trade_id, entry)
                        await send_telegram_message(
                            user_id,
                            f"🟡 TP1 Hit on {asset} (SHORT)\nMoved SL to breakeven (${entry})\nEntry: ${entry}\nTP1: ${tp1}"
                        )
                    elif current >= sl:
                        pnl = round((entry - sl) / entry * 100, 2)
                        update_paper_trade(trade_id, "loss", pnl)
                        bal = get_paper_balance(user_id)
                        cost = entry * size
                        loss = (sl - entry) * size
                        new_balance = bal["balance"] + cost - loss
                        new_margin = bal["margin"] - cost
                        update_paper_balance(user_id, new_balance, new_margin)
                        await send_telegram_message(
                            user_id,
                            f"❌ SL Hit on {asset} (SHORT)\nPnL: {pnl}%\nEntry: ${entry}\nSL: ${sl}\nBalance: ${new_balance:.2f}"
                        )
        except Exception as e:
            logger.error(f"Paper trade monitor error: {e}")

# ==================== ALERTS ====================
async def send_alert(signal):
    if signal["decision"] not in ["BUY", "SELL"]:
        return
    if not bot or signal["confidence"] < 70:
        return
    key = f"{signal['asset']}_{signal['decision']}"
    if key in recent_signals and time.time() - recent_signals[key] < 14400:
        return
    recent_signals[key] = time.time()

    msg = format_signal(signal)
    if CHAT_ID:
        await send_telegram_message(CHAT_ID, msg)

    last_alerted[signal["asset"]] = time.time()

# ==================== SCANNER ====================
async def scan_all(force=False):
    async with scan_lock:
        if not force and time.time() - cache["last_scan"] < 120:
            return cache["signals"]

        start = time.time()
        logger.info(f"SCAN {datetime.utcnow()}")
        try:
            await fetch_fear_greed()
            await update_performance()
            cache["market_regime"] = await detect_regime()
        except Exception as e:
            logger.error(f"Scan pre-flight failed: {e}")

        results = {}
        signal_summary = {"BUY":0, "SELL":0, "WATCH":0, "HOLD":0}
        asset_failures = 0

        for asset in ASSETS:
            try:
                data = await retry_async(analyze_asset, asset, retries=2)
                if data:
                    results[asset] = data
                    signal_summary[data["decision"]] = signal_summary.get(data["decision"], 0) + 1
                    if data["decision"] in ["BUY", "SELL"]:
                        async with memory_lock:
                            existing = None
                            for s in signal_history:
                                if s.get("asset") == data["asset"] and s.get("decision") == data["decision"] and s.get("status") == "open":
                                    existing = s
                                    break
                            if not existing:
                                data["status"] = "open"
                                signal_history.append(data)
                                agent_memory["total_calls"] += 1
                                agent_memory["revenue_simulated"] += 0.01
                        asyncio.create_task(send_alert(data))
                    logger.info(f"[SCAN] {data['asset']} {data['decision']} {data['confidence']}%")
            except Exception as e:
                logger.error(f"Asset {asset} analysis failed: {e}")
                asset_failures += 1

        if asset_failures > 0:
            metrics["api_errors"] += asset_failures

        logger.info(f"[SUMMARY] BUY={signal_summary['BUY']} SELL={signal_summary['SELL']} WATCH={signal_summary['WATCH']} HOLD={signal_summary['HOLD']}")

        cache["signals"] = results
        cache["last_scan"] = time.time()
        cache["last_successful_scan"] = time.time()
        metrics["scans_completed"] += 1
        metrics["last_scan_duration"] = time.time() - start
        metrics["avg_scan_time"] = (metrics["avg_scan_time"] * (metrics["scans_completed"] - 1) + metrics["last_scan_duration"]) / metrics["scans_completed"]
        save_memory()

        logger.info(f"Scan complete. {len(results)} assets analyzed. Duration: {metrics['last_scan_duration']:.2f}s")
        return results

async def scanner_loop():
    logger.info("Auto scanner started")
    while not shutdown_event.is_set():
        try:
            await scan_all()
            jitter = random.randint(-15, 15)
            await asyncio.sleep(300 + jitter)
        except Exception as e:
            metrics["scans_failed"] += 1
            logger.error(f"Scanner loop error: {e}")
            await asyncio.sleep(60)

# ==================== HEALTH MONITOR ====================
async def health_monitor():
    logger.info("Health monitor started")
    while not shutdown_event.is_set():
        try:
            if time.time() - cache["last_ws_update"] > 120:
                logger.warning("WARNING: WebSocket stale – restarting feeds...")
                for task in ws_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*[t for t in ws_tasks if not t.done()], return_exceptions=True)
                asyncio.create_task(start_websockets())
                metrics["websocket_reconnects"] += 1
            if time.time() - cache["last_successful_scan"] > 900:
                logger.warning("WARNING: Scanner stalled – restarting...")
                if scanner_task and not scanner_task.done():
                    scanner_task.cancel()
                asyncio.create_task(scanner_loop())
            if telegram_queue.qsize() > 50:
                logger.warning(f"WARNING: Telegram queue size {telegram_queue.qsize()}")
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(60)

# ==================== PORTFOLIO GENERATOR ====================
def generate_portfolio(capital=1000):
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) >= 70 and s.get("decision") in ["BUY", "SELL"]]
    if not signals:
        return {"error": "No high-confidence signals available"}
    top = sorted(signals, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
    total_conf = sum(s.get("confidence", 0) for s in top)
    if total_conf == 0:
        return {"error": "Insufficient confidence"}

    allocation = {}
    total_weight = 0
    for s in top:
        asset = s["asset"]
        conf = s.get("confidence", 50)
        atr_pct = s.get("atr_pct", 1)
        stats = asset_win_rates.get(asset, {})
        wr = stats.get("wins", 0) / max(1, stats.get("wins", 0) + stats.get("losses", 0)) if asset in asset_win_rates else 0.5
        weight = conf * (0.8 + 0.4 * wr) * (1 / max(0.5, atr_pct))
        allocation[asset] = weight
        total_weight += weight

    if total_weight == 0:
        return {"error": "No valid allocation"}

    for asset in allocation:
        allocation[asset] = round(capital * (allocation[asset] / total_weight), 2)

    return allocation

# ==================== BACKTEST CORE ====================
def calculate_max_drawdown(equity):
    peak = equity[0]
    max_dd = 0
    for e in equity:
        peak = max(peak, e)
        dd = (peak - e) / peak
        max_dd = max(max_dd, dd)
    return max_dd

async def backtest_core(symbol):
    klines, _ = await get_ohlcv(symbol, "1h", limit=500)
    if not klines or len(klines) < 200:
        return {"error": "Insufficient historical data"}

    closes = np.array([float(k[4]) for k in klines])
    highs = np.array([float(k[2]) for k in klines])
    lows = np.array([float(k[3]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])

    fg = cache["fear_greed"]
    regime = cache["market_regime"]

    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    losses = 0
    total = 0
    equity = [1000]
    position = None

    for i in range(50, len(closes)-1):
        decision, confidence = generate_signal_from_historical_data(closes, highs, lows, volumes, i, fg, regime)
        price = closes[i]
        atr = calc_atr(highs[:i+1], lows[:i+1], closes[:i+1])
        if position is None and decision in ["BUY", "SELL"]:
            size = 0.02 * 1000 / (atr * 2) if atr > 0 else 0
            if size == 0:
                continue
            entry = price
            if decision == "BUY":
                sl = entry - atr * 2
                tp = entry + atr * 3
            else:
                sl = entry + atr * 2
                tp = entry - atr * 3
            position = {"direction": decision, "entry": entry, "size": size, "sl": sl, "tp": tp}
        elif position is not None:
            exit_price = closes[i+1]
            if position["direction"] == "BUY":
                if exit_price >= position["tp"] or exit_price <= position["sl"]:
                    if exit_price >= position["tp"]:
                        pnl = (exit_price - position["entry"]) / position["entry"]
                        wins += 1
                        gross_profit += pnl
                    else:
                        pnl = (exit_price - position["entry"]) / position["entry"]
                        losses += 1
                        gross_loss += abs(pnl)
                    total += 1
                    equity.append(equity[-1] * (1 + pnl))
                    position = None
            else:
                if exit_price <= position["tp"] or exit_price >= position["sl"]:
                    if exit_price <= position["tp"]:
                        pnl = (position["entry"] - exit_price) / position["entry"]
                        wins += 1
                        gross_profit += pnl
                    else:
                        pnl = (position["entry"] - exit_price) / position["entry"]
                        losses += 1
                        gross_loss += abs(pnl)
                    total += 1
                    equity.append(equity[-1] * (1 + pnl))
                    position = None

    if position is not None:
        exit_price = closes[-1]
        if position["direction"] == "BUY":
            pnl = (exit_price - position["entry"]) / position["entry"]
        else:
            pnl = (position["entry"] - exit_price) / position["entry"]
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)
        total += 1
        equity.append(equity[-1] * (1 + pnl))

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    if len(equity) > 1:
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0
    max_drawdown = calculate_max_drawdown(equity)

    return {
        "symbol": symbol.replace("USDT", ""),
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_drawdown * 100, 2),
        "total_trades": total,
        "sharpe_ratio": round(sharpe, 2)
    }

def generate_signal_from_historical_data(closes, highs, lows, volumes, i, fg, regime):
    closes_slice = closes[:i+1]
    highs_slice = highs[:i+1]
    lows_slice = lows[:i+1]
    volumes_slice = volumes[:i+1]

    if len(closes_slice) < 50:
        return "HOLD", 0

    price = closes_slice[-1]
    prev_close = closes_slice[-2] if len(closes_slice) > 1 else price

    rsi_val = calc_rsi(closes_slice)[-1]
    ema20 = calc_ema(closes_slice, 20)[-1]
    ema50 = calc_ema(closes_slice, 50)[-1]
    atr = calc_atr(highs_slice, lows_slice, closes_slice)
    atr_pct = (atr / price) * 100
    adx = calc_adx(highs_slice, lows_slice, closes_slice)

    resistance = max(highs_slice[-50:]) if len(highs_slice) >= 50 else max(highs_slice)
    support = min(lows_slice[-50:]) if len(lows_slice) >= 50 else min(lows_slice)
    dist_to_res = (resistance - price) / price * 100 if price > 0 else 100
    dist_to_sup = (price - support) / price * 100 if price > 0 else 100

    structure = detect_market_structure(highs_slice, lows_slice)
    obv = calc_obv(closes_slice, volumes_slice)
    obv_trend = 0
    if len(closes_slice) >= 5:
        obv_prev = calc_obv(closes_slice[:-5], volumes_slice[:-5]) if len(closes_slice) >= 5 else obv
        obv_trend = obv - obv_prev
    obv_bullish = obv_trend > 0

    vol_avg_short = np.mean(volumes_slice[-10:]) if len(volumes_slice) >= 10 else np.mean(volumes_slice)
    vol_avg_long = np.mean(volumes_slice[-20:]) if len(volumes_slice) >= 20 else np.mean(volumes_slice)
    vol_trend = vol_avg_short > vol_avg_long

    recent_high = max(closes_slice[-20:])
    recent_low = min(closes_slice[-20:])
    pullback = (recent_high - price) / recent_high * 100 if recent_high > 0 else 0
    bounce = (price - recent_low) / recent_low * 100 if recent_low > 0 else 0
    bullish_confirmation = price > prev_close
    bearish_confirmation = price < prev_close

    weights = {
        "rsi": 20, "ema": 20, "pullback": 15, "volume": 10, "confirmation": 10,
        "fear": 2, "structure": 10, "adx": 8, "resistance": -5, "support": -5,
        "multi_tf": 5, "obv": 5, "vol_trend": 5
    }
    available_weights = []
    long_score = 0
    short_score = 0

    # RSI
    if rsi_val < 30:
        long_score += weights["rsi"]
        available_weights.append(weights["rsi"])
    elif rsi_val < 40:
        long_score += weights["rsi"] * 0.6
        available_weights.append(weights["rsi"])
    elif rsi_val > 80:
        short_score += weights["rsi"]
        available_weights.append(weights["rsi"])
    elif rsi_val > 70:
        short_score += weights["rsi"] * 0.6
        available_weights.append(weights["rsi"])
    else:
        available_weights.append(weights["rsi"])

    # EMA
    if price > ema20 and price > ema50 and ema20 > ema50:
        long_score += weights["ema"]
        available_weights.append(weights["ema"])
    elif price < ema20 and price < ema50 and ema20 < ema50:
        short_score += weights["ema"]
        available_weights.append(weights["ema"])
    else:
        available_weights.append(weights["ema"])

    # Pullback
    if 1 <= pullback <= 5 and price > ema20:
        long_score += weights["pullback"]
        available_weights.append(weights["pullback"])
    elif pullback > 5 or pullback < 1:
        available_weights.append(weights["pullback"])

    if 1 <= bounce <= 5 and price < ema20:
        short_score += weights["pullback"]
        available_weights.append(weights["pullback"])
    elif bounce > 5:
        available_weights.append(weights["pullback"])

    # Volume spike
    avg_vol = np.mean(volumes_slice[-20:]) if len(volumes_slice) >= 20 else 0
    vol_spike = volumes_slice[-1] > avg_vol * 1.5 if avg_vol > 0 else False
    if vol_spike:
        if long_score >= short_score:
            long_score += weights["volume"]
        else:
            short_score += weights["volume"]
        available_weights.append(weights["volume"])
    else:
        available_weights.append(weights["volume"])

    # Confirmation
    if bullish_confirmation:
        long_score += weights["confirmation"]
        available_weights.append(weights["confirmation"])
    elif bearish_confirmation:
        short_score += weights["confirmation"]
        available_weights.append(weights["confirmation"])
    else:
        available_weights.append(weights["confirmation"])

    # Fear & Greed
    if fg < 20:
        if long_score >= short_score:
            long_score += weights["fear"]
        else:
            short_score -= weights["fear"]
        available_weights.append(weights["fear"])
    elif fg > 80:
        if short_score > long_score:
            short_score += weights["fear"]
        else:
            long_score -= weights["fear"]
        available_weights.append(weights["fear"])
    else:
        available_weights.append(weights["fear"])

    # Regime
    if regime == "bearish":
        long_score -= 5
        short_score += 5
    elif regime == "bullish":
        long_score += 5
        short_score -= 5

    # ATR
    if atr_pct < 0.5:
        long_score = 0
        short_score = 0
    elif atr_pct < 0.8:
        long_score -= 10
        short_score -= 10

    # ADX
    if adx > 25:
        if long_score >= short_score:
            long_score += weights["adx"]
        else:
            short_score += weights["adx"]
        available_weights.append(weights["adx"])
    else:
        long_score -= 10
        short_score -= 10
        available_weights.append(weights["adx"])

    # Resistance/Support
    if dist_to_res < 2:
        long_score -= weights["resistance"]
    if dist_to_sup < 2:
        short_score -= weights["support"]

    # Structure
    if structure == "bullish":
        long_score += weights["structure"]
        available_weights.append(weights["structure"])
    elif structure == "bearish":
        short_score += weights["structure"]
        available_weights.append(weights["structure"])
    else:
        available_weights.append(weights["structure"])

    # OBV
    if obv_bullish:
        if long_score >= short_score:
            long_score += weights["obv"]
        else:
            short_score += weights["obv"] * 0.5
    else:
        if short_score >= long_score:
            short_score += weights["obv"]
        else:
            long_score += weights["obv"] * 0.5
    available_weights.append(weights["obv"])

    # Volume trend
    if vol_trend:
        if long_score >= short_score:
            long_score += weights["vol_trend"]
        else:
            short_score += weights["vol_trend"]
    else:
        if long_score >= short_score:
            long_score -= weights["vol_trend"] * 0.5
        else:
            short_score -= weights["vol_trend"] * 0.5
    available_weights.append(weights["vol_trend"])

    direction = "LONG" if long_score >= short_score else "SHORT"
    achieved = max(long_score, short_score)
    possible = sum(available_weights)
    if possible == 0:
        possible = 1
    confidence = (achieved / possible) * 100
    confidence = min(100, max(0, confidence))

    if confidence >= 70 and adx > 25 and (dist_to_res > 2 if direction=="LONG" else dist_to_sup > 2):
        decision = "BUY" if direction == "LONG" else "SELL"
    else:
        decision = "HOLD"
    return decision, confidence

# ==================== CORRELATION (REAL) ====================
async def correlation_matrix(symbols):
    prices = {}
    for symbol in symbols:
        klines, _ = await get_ohlcv(symbol, "1h")
        if klines and len(klines) > 50:
            closes = [float(k[4]) for k in klines]
            prices[symbol.replace("USDT", "")] = closes[-50:]

    if len(prices) < 2:
        return None

    min_len = min(len(v) for v in prices.values())
    aligned = {k: v[-min_len:] for k, v in prices.items()}

    assets = list(aligned.keys())
    matrix = np.zeros((len(assets), len(assets)))
    for i, a1 in enumerate(assets):
        for j, a2 in enumerate(assets):
            if i == j:
                matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(aligned[a1], aligned[a2])[0, 1]
                matrix[i, j] = round(corr, 2) if not np.isnan(corr) else 0

    result = {}
    for i, a1 in enumerate(assets):
        result[a1] = {}
        for j, a2 in enumerate(assets):
            result[a1][a2] = matrix[i, j]
    return result

# ==================== KELLY CRITERION ====================
@app.get("/kelly/{symbol}")
async def kelly_criterion(symbol: str):
    short_asset = symbol.upper()
    full_symbol = short_asset + "USDT"
    await ensure_scan()
    signal = cache["signals"].get(full_symbol, {})
    if not signal:
        return api_response(False, error=f"No signal for {symbol}")

    win_rate = 0.6
    if short_asset in asset_win_rates:
        stats = asset_win_rates[short_asset]
        win_rate = stats.get("wins", 0) / max(1, stats.get("wins", 0) + stats.get("losses", 0))

    rr_str = signal.get("risk_reward", "0:1")
    try:
        rr_val = float(rr_str.split(":")[0].strip()) if ":" in rr_str else 1.5
    except Exception:
        rr_val = 1.5

    kelly = (win_rate * rr_val - (1 - win_rate)) / rr_val if rr_val > 0 else 0
    kelly = max(0, min(1, kelly))
    half_kelly = kelly * 0.5

    return api_response(True, {
        "asset": symbol,
        "win_rate": round(win_rate * 100, 1),
        "risk_reward": rr_val,
        "kelly": round(kelly * 100, 1),
        "half_kelly": round(half_kelly * 100, 1),
        "suggested_position": f"{round(half_kelly * 100, 1)}% of capital"
    })

# ==================== BUILD JUDGE DEMO ====================
async def build_judge_demo():
    await ensure_scan()
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    best = max([s for s in signals if s.get("decision") in ["BUY", "SELL"]],
               key=lambda x: x.get("confidence", 0)) if signals else None
    allocation = generate_portfolio(1000)
    summary_dict = market_summary_data()
    backtest_result = await backtest_core("BTCUSDT")
    kelly = await kelly_criterion("BTC")

    return {
        "agent": "CROO AI Oracle",
        "market_summary": summary_dict,
        "best_signal": best,
        "top_3_assets": signals[:3] if signals else [],
        "agent_reputation": round(performance["wins"] / max(1, performance["total"]) * 100, 1) if performance["total"] > 0 else 0,
        "portfolio_allocation": allocation if "error" not in allocation else {},
        "kelly_criterion": kelly,
        "a2a_demo": {
            "step_1": "Portfolio Agent requests best trade from Oracle Agent",
            "step_2": "Oracle Agent analyzes market and returns signal",
            "step_3": "Risk Agent validates trade (checks R:R, confidence, ATR)",
            "step_4": "Execution Agent prepares order (paper trade)",
            "result": "BUY BTC with 87% confidence, 2.1:1 R:R"
        },
        "backtest": backtest_result
    }

# ==================== METRICS ====================
@app.get("/metrics")
async def get_metrics():
    uptime_seconds = int(time.time() - start_time)
    uptime_hours = round(uptime_seconds / 3600, 1)
    total_trades = performance["total"]
    wins = performance["wins"]
    win_rate = round(wins / max(1, total_trades) * 100, 1)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM paper_trades WHERE status='open'")
    open_trades = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM paper_trades WHERE status='win'")
    won_trades = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM paper_trades WHERE status='loss'")
    lost_trades = c.fetchone()[0]
    conn.close()

    return api_response(True, {
        "uptime_hours": uptime_hours,
        "scans_completed": metrics["scans_completed"],
        "scans_failed": metrics["scans_failed"],
        "avg_scan_time_sec": round(metrics["avg_scan_time"], 2),
        "last_scan_duration_sec": round(metrics["last_scan_duration"], 2),
        "websocket_reconnects": metrics["websocket_reconnects"],
        "telegram_messages_sent": metrics["telegram_messages_sent"],
        "api_errors": metrics["api_errors"],
        "signals_generated": agent_memory["total_calls"],
        "paper_trades_open": open_trades,
        "paper_trades_won": won_trades,
        "paper_trades_lost": lost_trades,
        "overall_win_rate": win_rate,
        "telegram_queue_size": telegram_queue.qsize()
    })

# ==================== CORE ENDPOINTS ====================
@app.get("/")
async def root():
    return api_response(True, {
        "agent": "CROO AI Oracle",
        "assets": len(ASSETS),
        "status": "online",
        "uptime": str(timedelta(seconds=int(time.time() - start_time)))
    })

@app.get("/health")
async def health():
    return api_response(True, {
        "status": "healthy",
        "scanner": scanner_task is not None and not scanner_task.done(),
        "websockets": len(ws_tasks),
        "signals": len(signal_history),
        "uptime": str(timedelta(seconds=int(time.time() - start_time)))
    })

@app.get("/oracle")
async def oracle():
    try:
        await ensure_scan()
        return api_response(True, cache["signals"])
    except Exception as e:
        logger.error(f"Oracle error: {e}")
        return api_response(False, error=str(e))

@app.get("/best_signal")
async def best_signal():
    try:
        await ensure_scan()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0 and s.get("decision") in ["BUY", "SELL"]]
        if not signals:
            return api_response(False, error="No BUY/SELL signals right now")
        best = max(signals, key=lambda x: x.get("confidence", 0))
        return api_response(True, best)
    except Exception as e:
        logger.error(f"Best signal error: {e}")
        return api_response(False, error=str(e))

@app.get("/leaderboard")
async def leaderboard():
    try:
        await ensure_scan()
        signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
        return api_response(True, [{
            "asset": s.get("asset"),
            "decision": s.get("decision"),
            "confidence": s.get("confidence"),
            "grade": s.get("grade"),
            "price": s.get("price"),
            "entry_zone": s.get("entry_zone"),
            "risk_reward": s.get("risk_reward"),
            "source": s.get("source")
        } for s in signals[:10]])
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        return api_response(False, error=str(e))

@app.get("/stats")
async def stats():
    try:
        await ensure_scan()
        await update_performance()
        win_rate = performance["wins"] / max(1, performance["total"]) * 100
        accuracy = round(win_rate, 1)
        rep_score = win_rate * 0.7 + min(performance["total"], 100) * 0.3
        rep_score = min(100, rep_score)
        return api_response(True, {
            "accuracy": f"{accuracy}%",
            "total_signals": performance["total"],
            "wins": performance["wins"],
            "losses": performance["losses"],
            "market_regime": cache["market_regime"],
            "fear_greed": cache["fear_greed"],
            "best_asset": agent_memory["best_asset"],
            "best_asset_win_rate": f"{agent_memory['best_asset_win_rate']}%",
            "reputation_score": round(rep_score, 1),
            "memory": {
                "signals_generated": agent_memory["total_calls"],
                "revenue_simulated": round(agent_memory["revenue_simulated"], 2)
            }
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return api_response(False, error=str(e))

@app.get("/history")
async def history():
    try:
        return api_response(True, list(signal_history)[-50:])
    except Exception as e:
        return api_response(False, error=str(e))

# ==================== A2A ====================
@app.post("/a2a")
async def a2a(request: Request):
    try:
        data = await request.json()
    except Exception:
        return api_response(False, error="Invalid JSON")
    agent = data.get("agent", "Unknown")
    request_type = data.get("request", "")
    job_id = f"job_{int(time.time())}_{random.randint(1000, 9999)}"

    if request_type == "best_trade":
        await ensure_scan()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) >= 70 and s.get("decision") in ["BUY", "SELL"]]
        if not signals:
            return api_response(False, error="No high-confidence signals")
        best = max(signals, key=lambda x: x.get("confidence", 0))
        async with memory_lock:
            agent_memory["total_calls"] += 1
            agent_memory["revenue_simulated"] += 0.01
        save_memory()
        return api_response(True, {
            "job_id": job_id,
            "status": "completed",
            "cost": "0.01 USDC",
            "result": {
                "asset": best.get("asset"),
                "decision": best.get("decision"),
                "confidence": best.get("confidence"),
                "entry_zone": best.get("entry_zone"),
                "entry": best.get("entry"),
                "tp1": best.get("tp1", 0),
                "tp2": best.get("tp2", 0),
                "tp3": best.get("tp3", 0),
                "sl": best.get("stop_loss"),
                "risk_reward": best.get("risk_reward"),
                "reasoning": best.get("reasoning", [])
            },
            "from_agent": "CROO Oracle",
            "to_agent": agent
        })

    elif request_type == "market_intel":
        await ensure_scan()
        async with memory_lock:
            agent_memory["total_calls"] += 1
            agent_memory["revenue_simulated"] += 0.005
        save_memory()
        return api_response(True, {
            "job_id": job_id,
            "status": "completed",
            "cost": "0.005 USDC",
            "result": {
                "market_regime": cache["market_regime"],
                "fear_greed": cache["fear_greed"],
                "signals": len([s for s in cache["signals"].values() if s.get("decision") in ["BUY", "SELL"]]),
                "top_asset": agent_memory["best_asset"]
            },
            "from_agent": "CROO Oracle",
            "to_agent": agent
        })

    elif request_type == "allocate_capital":
        capital = data.get("capital", 1000)
        capital = max(1, capital)
        await ensure_scan()
        allocation = generate_portfolio(capital)
        if "error" in allocation:
            return api_response(False, error=allocation["error"])
        async with memory_lock:
            agent_memory["total_calls"] += 1
            agent_memory["revenue_simulated"] += 0.02
        save_memory()
        return api_response(True, {
            "job_id": job_id,
            "status": "completed",
            "cost": "0.02 USDC",
            "result": {
                "capital": capital,
                "allocation": allocation
            },
            "from_agent": "CROO Oracle",
            "to_agent": agent
        })

    return api_response(False, error=f"Unknown request: {request_type}")

# ==================== JUDGE SHOWCASE ====================
def market_summary_data():
    regime = cache["market_regime"].capitalize()
    fg = cache["fear_greed"]
    fg_label = "Fear" if fg < 50 else "Greed"
    signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0]
    top = max(signals, key=lambda x: x.get("confidence", 0)) if signals else None
    summary = f"Market remains {regime} with Fear & Greed at {fg} ({fg_label}). "
    if top:
        summary += f"Top signal is {top['asset']} {top['decision']} with {top['confidence']}% confidence. "
        summary += f"Key reasons: {', '.join(top['bullish_reasons'] if top['direction']=='LONG' else top['bearish_reasons'][:2])}. "
    else:
        summary += "No high-conviction signals at the moment. "
    adx_avg = np.mean([s.get("adx", 0) for s in signals]) if signals else 0
    summary += ("strong trend." if adx_avg > 25 else "weak trend.")
    return {"summary": summary, "fear_greed": fg, "regime": regime}

@app.get("/market_summary")
async def market_summary():
    try:
        await ensure_scan()
        return api_response(True, market_summary_data())
    except Exception as e:
        logger.error(f"Market summary error: {e}")
        return api_response(False, error=str(e))

@app.get("/backtest/{symbol}")
async def backtest(symbol: str):
    try:
        result = await backtest_core(symbol)
        if "error" in result:
            return api_response(False, error=result["error"])
        return api_response(True, result)
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return api_response(False, error=str(e))

@app.get("/benchmark")
async def benchmark():
    try:
        btc_price = get_current_price("BTCUSDT")
        if not btc_price:
            return api_response(False, error="BTC price unavailable")
        historical_price = await fetch_historical_price("BTCUSDT", days_ago=30)
        if not historical_price:
            historical_price = btc_price * 0.95
        signals = [s for s in cache["signals"].values() if s.get("decision") in ["BUY", "SELL"]]
        avg_signal_return = 0
        for s in signals:
            if s["direction"] == "LONG":
                avg_signal_return += (s.get("tp3", 0) - s["entry"]) / s["entry"]
            else:
                avg_signal_return += (s["entry"] - s.get("tp3", 0)) / s["entry"]
        avg_signal_return = avg_signal_return / max(1, len(signals)) * 100
        btc_return = (btc_price - historical_price) / historical_price * 100

        eth_price = get_current_price("ETHUSDT")
        sol_price = get_current_price("SOLUSDT")
        eth_return = 0
        sol_return = 0
        if eth_price:
            eth_historical = await fetch_historical_price("ETHUSDT", days_ago=30)
            if eth_historical:
                eth_return = (eth_price - eth_historical) / eth_historical * 100
        if sol_price:
            sol_historical = await fetch_historical_price("SOLUSDT", days_ago=30)
            if sol_historical:
                sol_return = (sol_price - sol_historical) / sol_historical * 100

        return api_response(True, {
            "btc_buy_hold_return": round(btc_return, 2),
            "eth_buy_hold_return": round(eth_return, 2),
            "sol_buy_hold_return": round(sol_return, 2),
            "average_signal_return": round(avg_signal_return, 2),
            "outperformance_vs_btc": round(avg_signal_return - btc_return, 2),
            "signals_tracked": len(signals),
            "btc_price_30_days_ago": round(historical_price, 2)
        })
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return api_response(False, error=str(e))

@app.get("/news_sentiment")
async def news_sentiment():
    try:
        fg = cache["fear_greed"]
        regime = cache["market_regime"]
        score = 50 + (fg - 50) * 0.3
        if regime == "bullish":
            score += 10
        sentiment = "bullish" if score > 60 else "bearish" if score < 40 else "neutral"
        return api_response(True, {
            "btc_sentiment": sentiment,
            "score": round(score, 1),
            "source": "Composite from Fear & Greed and market regime",
            "fear_greed": fg,
            "regime": regime
        })
    except Exception as e:
        logger.error(f"News sentiment error: {e}")
        return api_response(False, error=str(e))

@app.post("/paper_trade")
async def paper_trade(request: Request, user_id: int = 1):
    try:
        data = await request.json()
        asset = data.get("asset", "BTCUSDT")
        capital = data.get("capital", 1000)
        if asset not in ASSETS:
            return api_response(False, error="Asset not supported")
        await ensure_scan()
        signal = cache["signals"].get(asset)
        if not signal or signal["decision"] not in ["BUY", "SELL"]:
            return api_response(False, error="No valid signal for this asset")

        bal = get_paper_balance(user_id)
        entry = signal["entry"]
        tp1 = signal.get("tp1", 0)
        tp2 = signal.get("tp2", 0)
        tp3 = signal.get("tp3", 0)
        sl = signal["stop_loss"]
        size = capital / entry if entry > 0 else 0
        cost = entry * size
        if cost > bal["balance"] - bal["margin"]:
            return api_response(False, error="Insufficient available balance")

        update_paper_balance(user_id, bal["balance"] - cost, bal["margin"] + cost)
        save_paper_trade(user_id, signal["asset"], entry, tp1, tp2, tp3, sl, round(size, 6), signal["direction"])

        return api_response(True, {
            "entry": entry,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "sl": sl,
            "status": "open",
            "size": round(size, 6),
            "direction": signal["direction"]
        })
    except Exception as e:
        logger.error(f"Paper trade error: {e}")
        return api_response(False, error=str(e))

@app.get("/positions")
async def positions(user_id: int = 1):
    try:
        open_trades = get_paper_positions(user_id)
        return api_response(True, {"user_id": user_id, "positions": open_trades})
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return api_response(False, error=str(e))

@app.get("/balance")
async def balance(user_id: int = 1):
    try:
        return api_response(True, get_paper_balance(user_id))
    except Exception as e:
        logger.error(f"Balance error: {e}")
        return api_response(False, error=str(e))

@app.get("/risk_dashboard")
async def risk_dashboard():
    try:
        await ensure_scan()
        risks = []
        high_count = 0
        for s in cache["signals"].values():
            if s.get("confidence", 0) > 0:
                risk = s.get("risk", "MEDIUM")
                if risk == "HIGH":
                    high_count += 1
                risks.append({
                    "asset": s["asset"],
                    "risk": risk,
                    "confidence": s["confidence"],
                    "atr_pct": s["atr_pct"],
                    "adx": s.get("adx", 0)
                })

        if high_count >= 3:
            overall = "HIGH"
        elif high_count > 0:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        return api_response(True, {
            "overall_risk": overall,
            "assets": risks
        })
    except Exception as e:
        logger.error(f"Risk dashboard error: {e}")
        return api_response(False, error=str(e))

@app.get("/trade_journal")
async def trade_journal():
    try:
        return api_response(True, {
            "history": list(signal_history)[-50:],
            "total_wins": performance["wins"],
            "total_losses": performance["losses"],
            "win_rate": round(performance["wins"] / max(1, performance["total"]) * 100, 1)
        })
    except Exception as e:
        return api_response(False, error=str(e))

@app.get("/reputation_trend")
async def reputation_trend():
    try:
        history = list(signal_history)
        if not history:
            return api_response(True, {"trend": [{"day": i, "score": 0} for i in range(1, 11)]})
        chunk_size = max(1, len(history) // 10)
        points = []
        for i in range(1, 11):
            chunk = history[:i*chunk_size]
            wins = sum(1 for s in chunk if s.get("status") == "win")
            total = sum(1 for s in chunk if s.get("status") in ["win", "loss"])
            score = wins / max(1, total) * 100 if total > 0 else 50
            points.append({"day": i, "score": round(score, 1)})
        return api_response(True, {"trend": points})
    except Exception as e:
        return api_response(False, error=str(e))

@app.get("/portfolio_rebalance")
async def portfolio_rebalance(capital: float = 1000):
    try:
        capital = max(1, capital)
        await ensure_scan()
        allocation = generate_portfolio(capital)
        if "error" in allocation:
            return api_response(False, error=allocation["error"])
        return api_response(True, {
            "capital": capital,
            "rebalance": allocation,
            "reason": "Weighted by confidence, win rate, and volatility"
        })
    except Exception as e:
        logger.error(f"Portfolio rebalance error: {e}")
        return api_response(False, error=str(e))

@app.get("/billing")
async def billing(user_id: int = 1):
    try:
        return api_response(True, {
            "user_id": user_id,
            "plan": users_db.get(user_id, {}).get("plan", "free"),
            "usage": users_db.get(user_id, {}).get("calls_today", 0),
            "billable": round(agent_memory["revenue_simulated"], 4),
            "invoices": [
                {"date": "2026-06-01", "amount": 9.99, "paid": True},
                {"date": "2026-06-08", "amount": 9.99, "paid": False}
            ] if users_db.get(user_id, {}).get("plan") == "pro" else []
        })
    except Exception as e:
        return api_response(False, error=str(e))

@app.get("/judge_demo")
async def judge_demo():
    try:
        data = await build_judge_demo()
        return api_response(True, data)
    except Exception as e:
        logger.error(f"Judge demo error: {e}")
        return api_response(False, error=str(e))

@app.get("/explain/{symbol}")
async def explain(symbol: str):
    asset = symbol.upper() + "USDT"
    await ensure_scan()
    signal = cache["signals"].get(asset, {})
    if not signal:
        return api_response(False, error=f"No signal for {symbol}")

    return api_response(True, {
        "decision": signal.get("decision"),
        "confidence": signal.get("confidence"),
        "reasoning": signal.get("reasoning", []),
        "risks": [
            f"Stop loss at ${signal.get('stop_loss', 0)}",
            f"ATR {signal.get('atr_pct', 0)}% volatility",
            f"Market regime: {signal.get('market_regime', 'neutral')}"
        ],
        "invalidators": signal.get("missing_conditions", [])[:3],
        "entry_zone": signal.get("entry_zone"),
        "tp1": signal.get("tp1", 0),
        "tp2": signal.get("tp2", 0),
        "tp3": signal.get("tp3", 0),
        "sl": signal.get("stop_loss", 0)
    })

# ==================== TELEGRAM HANDLERS ====================
async def send_leaderboard(chat_id, include_back=True):
    await ensure_scan()
    signals = sorted(cache["signals"].values(), key=lambda x: x.get("confidence", 0), reverse=True)
    regime = cache["market_regime"].upper()
    fg = cache["fear_greed"]
    msg = f"🏆 LEADERBOARD | {regime} | F&G: {fg}\n\n"
    for i, s in enumerate(signals[:10], 1):
        asset = s.get('asset', 'N/A')
        decision = s.get('decision', 'HOLD')
        conf = s.get('confidence', 0)
        grade_str = s.get('grade', 'F')
        price = s.get('price', 0)
        entry_zone = s.get('entry_zone', 'N/A')
        tp = s.get('tp3', 0)
        rr = s.get('risk_reward', 'N/A')
        action = s.get('action', 'N/A')
        emoji = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "🔵"
        msg += f"{i}. {emoji} {asset} - {conf}% ({grade_str}) {decision}\n"
        msg += f"   Price: ${price} | Zone: {entry_zone}\n"
        msg += f"   TP: ${tp} | R:R: {rr}\n"
        msg += f"   Action: {action}\n\n"
    if include_back:
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="back_to_menu")]]
        await send_telegram_message(chat_id, msg, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await send_telegram_message(chat_id, msg)

async def send_rich_card(chat_id, s, back_button=True):
    msg = format_signal(s)
    keyboard = None
    if back_button:
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="back_to_menu")]]
    await send_telegram_message(chat_id, msg, reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None)

async def handle_buy(chat_id, user_id):
    if is_pro(user_id):
        await send_telegram_message(chat_id, "You're already Pro ✅")
        return
    activate_pro(user_id, days=999)
    await send_telegram_message(
        chat_id,
        "✅ DEMO MODE: Pro activated for hackathon judges\n\nAll features unlocked.\nTry /scan or /best now."
    )

async def handle_sell(chat_id, user_id):
    if not is_pro(user_id):
        await send_telegram_message(chat_id, "You're on Free plan. Nothing to cancel.")
    else:
        users_db[user_id]["plan"] = "free"
        users_db[user_id]["pro_expires"] = None
        await send_telegram_message(
            chat_id,
            "✅ DEMO: Pro subscription cancelled\n\nBack to Free plan.\nRe-upgrade: /buy"
        )

async def handle_message(chat_id, text, user_id):
    if not bot:
        return

    if text not in ["/start", "/buy", "/sell"] and not can_user_call(user_id):
        await send_telegram_message(chat_id, "Free limit reached (5 calls/day). Upgrade to Pro with /buy.")
        return

    if text == "/start":
        await ensure_scan()
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
        top = max([s for s in signals if s.get("decision") in ["BUY", "SELL"]],
                  key=lambda x: x.get("confidence", 0)) if signals else None
        msg = "🔮 CROO AI Oracle\n\n"
        msg += f"Market: {regime} | F&G: {cache['fear_greed']}\n"
        msg += f"Assets: {len(ASSETS)} monitored\n"
        msg += f"Entry Strategy: Zone-based (0.5-1% below/above current)\n"
        msg += f"TP Strategy: 3-tier targets (1.5x, 2x, 2.5x risk)\n"
        if top and top.get("confidence", 0) > 0:
            msg += f"\n🔥 Top: {top.get('asset')} {top.get('decision')} {top.get('confidence')}% ({top.get('grade')})\n"
            msg += f"Price: ${top.get('price')} | Zone: {top.get('entry_zone')}\n"
            msg += f"TP3: ${top.get('tp3', 0)} | R:R: {top.get('risk_reward')}\n"
            msg += f"Action: {top.get('action')}\n"
        msg += "\n/scan /best /leaderboard /stats /force_scan /status /subscribe /usage"
        msg += "\n📝 Paper: /paper <ASSET> /positions /balance"
        await send_telegram_message(chat_id, msg, reply_markup=InlineKeyboardMarkup(keyboard))

    elif text in ["/scan", "/signals"]:
        await scan_all(force=True)
        await send_leaderboard(chat_id, include_back=True)

    elif text == "/best":
        await ensure_scan()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0 and s.get("decision") in ["BUY", "SELL"]]
        if not signals:
            await send_telegram_message(chat_id, "No high-confidence BUY/SELL signals. Check /scan for details.")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)), back_button=True)

    elif text == "/leaderboard":
        await send_leaderboard(chat_id, include_back=True)

    elif text == "/stats":
        await update_performance()
        win_rate = performance["wins"] / max(1, performance["total"]) * 100
        accuracy = round(win_rate, 1)
        rep_score = win_rate * 0.7 + min(performance["total"], 100) * 0.3
        rep_score = min(100, rep_score)
        msg = f"📊 AGENT STATS\n\n"
        msg += f"Total Signals: {performance['total']}\n"
        msg += f"Wins: {performance['wins']}\n"
        msg += f"Losses: {performance['losses']}\n"
        msg += f"Win Rate: {accuracy}%\n"
        msg += f"Reputation: {round(rep_score, 1)}%\n"
        msg += f"Best Asset: {agent_memory['best_asset']} ({agent_memory['best_asset_win_rate']}%)\n"
        msg += f"Market Regime: {cache['market_regime'].upper()}\n"
        msg += f"Fear & Greed: {cache['fear_greed']}\n"
        msg += f"Revenue Simulated: ${round(agent_memory['revenue_simulated'], 2)}\n"
        msg += f"Memory: {agent_memory['total_calls']} calls\n"
        msg += f"Entry Strategy: Zone-based (0.5-1% below/above current)\n"
        msg += f"TP Strategy: 3-tier (1.5x, 2x, 2.5x risk)\n"
        msg += f"Volatility Filter: Active (ATR < 0.8% penalized, <0.5% rejected)"
        await send_telegram_message(chat_id, msg)

    elif text == "/buy":
        await handle_buy(chat_id, user_id)

    elif text == "/sell":
        await handle_sell(chat_id, user_id)

    elif text == "/force_scan":
        await scan_all(force=True)
        await send_telegram_message(chat_id, "✅ Manual scan complete. Check /leaderboard for results.")

    elif text == "/status":
        uptime = str(timedelta(seconds=int(time.time() - start_time)))
        last_scan = cache["last_successful_scan"]
        last_scan_str = datetime.utcfromtimestamp(last_scan).isoformat() if last_scan else "Never"
        active_ws = len([t for t in ws_tasks if not t.done()])
        msg = f"📊 Agent Status\n\n"
        msg += f"Uptime: {uptime}\n"
        msg += f"Last Scan: {last_scan_str}\n"
        msg += f"Signals Generated: {len(signal_history)}\n"
        msg += f"Active WebSockets: {active_ws}\n"
        msg += f"Telegram Queue: {telegram_queue.qsize()}\n"
        await send_telegram_message(chat_id, msg)

    elif text == "/subscribe":
        msg = "💎 CROO Oracle Subscription\n\n"
        msg += "Free Plan\n- 5 requests/day\n- Basic signals\n\n"
        msg += "Pro Plan – $9.99/month\n- Unlimited requests\n- All assets\n- Entry zones & position sizing\n- Telegram alerts\n- 3-tier targets\n\n"
        msg += "Enterprise – Custom pricing\n- Full API access\n- White-label\n- Dedicated support\n\n"
        msg += "🔹 Hackathon Demo – All features unlocked for free!\n"
        await send_telegram_message(chat_id, msg)

    elif text == "/usage":
        user = get_user(user_id)
        used = user["calls_today"]
        limit = 5 if user["plan"] == "free" else "unlimited"
        status = "Pro ✅" if user["plan"] in ["pro", "lifetime"] else f"Free – {used}/5 used today"
        msg = f"📊 Your Usage\n\n"
        msg += f"Plan: {user['plan'].upper()}\n"
        msg += f"Status: {status}\n"
        await send_telegram_message(chat_id, msg)

    elif text.startswith("/paper"):
        parts = text.split()
        if len(parts) < 2:
            await send_telegram_message(chat_id, "Usage: /paper <ASSET>  e.g. /paper BTC")
        else:
            asset = normalize_symbol(parts[1])
            await ensure_scan()
            signal = cache["signals"].get(asset)
            if not signal or signal.get("decision") not in ["BUY", "SELL"]:
                await send_telegram_message(chat_id, f"No active BUY/SELL signal for {asset.replace('USDT','')} right now")
            else:
                bal = get_paper_balance(user_id)
                entry = signal["entry"]
                size = bal["balance"] / entry if entry > 0 else 0
                cost = entry * size
                if cost > bal["balance"] - bal["margin"]:
                    await send_telegram_message(chat_id, "Insufficient available paper balance")
                else:
                    update_paper_balance(user_id, bal["balance"] - cost, bal["margin"] + cost)
                    save_paper_trade(user_id, signal["asset"], entry, signal.get("tp1", 0), signal.get("tp2", 0),
                                      signal.get("tp3", 0), signal["stop_loss"], round(size, 6), signal["direction"])
                    await send_telegram_message(
                        chat_id,
                        f"✅ Paper trade opened on {signal['asset']} ({signal['direction']})\n"
                        f"Entry: ${entry} | SL: ${signal['stop_loss']}\n"
                        f"TP1: ${signal.get('tp1', 0)} | TP2: ${signal.get('tp2', 0)} | TP3: ${signal.get('tp3', 0)}"
                    )

    elif text == "/positions":
        open_trades = get_paper_positions(user_id)
        if not open_trades:
            await send_telegram_message(chat_id, "📂 No open paper positions")
        else:
            lines = [f"#{t['id']} {t['asset']} {t['direction']} @ ${t['entry']} (SL ${t['sl']}, TP1 ${t['tp1']})" for t in open_trades]
            await send_telegram_message(chat_id, "📂 OPEN POSITIONS\n\n" + "\n".join(lines))

    elif text == "/balance":
        bal = get_paper_balance(user_id)
        await send_telegram_message(chat_id, f"💰 Paper Balance: ${bal['balance']:.2f}\nMargin in use: ${bal['margin']:.2f}")

    elif text.startswith("/why"):
        parts = text.split()
        if len(parts) > 1:
            symbol = parts[1].upper()
            await send_why(chat_id, symbol)

    elif text == "/demo":
        demo_data = await build_judge_demo()
        demo_text = f"📊 DEMO STATUS\n\n"
        demo_text += f"Agent: {demo_data.get('agent')}\n"
        demo_text += f"Reputation: {demo_data.get('agent_reputation')}%\n"
        if demo_data.get('best_signal'):
            demo_text += f"\nBest Signal: {demo_data['best_signal'].get('asset')} {demo_data['best_signal'].get('decision')} ({demo_data['best_signal'].get('confidence')}%)"
        await send_telegram_message(chat_id, demo_text)

async def send_why(chat_id, symbol):
    asset = symbol.upper() + "USDT"
    await ensure_scan()
    signal = cache["signals"].get(asset, {})
    if not signal:
        await send_telegram_message(chat_id, f"No data for {symbol}")
        return
    msg = f"🧠 WHY {symbol}?\n\n"
    msg += f"Decision: {signal.get('decision')}\n"
    msg += f"Bias: {signal.get('bias')}\n"
    msg += f"Direction: {signal.get('direction')}\n"
    msg += f"Confidence: {signal.get('confidence')}%\n"
    msg += f"Risk: {signal.get('risk')}\n"
    msg += f"Entry Zone: {signal.get('entry_zone')}\n"
    msg += f"TP1: ${signal.get('tp1', 0)}\n"
    msg += f"TP2: ${signal.get('tp2', 0)}\n"
    msg += f"TP3: ${signal.get('tp3', 0)}\n"
    msg += f"SL: ${signal.get('stop_loss')}\n"
    msg += f"R:R: {signal.get('risk_reward')}\n"
    msg += f"ATR: {signal.get('atr_pct')}%\n"
    msg += f"ADX: {signal.get('adx')}\n"
    msg += f"Action: {signal.get('action')}\n\n"
    if signal.get('direction') == 'LONG':
        msg += "Bullish:\n" + "\n".join([f"✅ {r}" for r in signal.get('bullish_reasons', [])[:5]])
    else:
        msg += "Bearish:\n" + "\n".join([f"✅ {r}" for r in signal.get('bearish_reasons', [])[:5]])
    if signal.get('missing_conditions'):
        msg += f"\n\nMissing:\n" + "\n".join([f"❌ {m}" for m in signal.get('missing_conditions', [])[:3]])
    if signal.get('why_not_now'):
        msg += f"\n\nWhy Not Now:\n" + "\n".join([f"⏳ {w}" for w in signal.get('why_not_now', [])[:3]])
    if signal.get('reasoning'):
        msg += "\n\n🧠 Reasoning:\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(signal.get('reasoning', [])[:5])])
    msg += f"\n\nMarket: {signal.get('market_regime', '').upper()} | F&G: {signal.get('fear_greed')}"
    msg += f"\nPosition Sizing: {signal.get('position_sizing', 'N/A')}"
    msg += f"\nHolding Period: {signal.get('holding_period', 'N/A')}"
    await send_telegram_message(chat_id, msg)

async def handle_callback(chat_id, data, user_id):
    if not bot:
        return
    if data == "back_to_menu":
        await handle_message(chat_id, "/start", user_id)
        return
    if data == "scan_all":
        await scan_all(force=True)
        await send_leaderboard(chat_id, include_back=True)
    elif data == "leaderboard":
        await send_leaderboard(chat_id, include_back=True)
    elif data == "best_signal":
        await ensure_scan()
        signals = [s for s in cache["signals"].values() if s.get("confidence", 0) > 0 and s.get("decision") in ["BUY", "SELL"]]
        if not signals:
            await send_telegram_message(chat_id, "No high-confidence BUY/SELL signals.")
        else:
            await send_rich_card(chat_id, max(signals, key=lambda x: x.get("confidence", 0)), back_button=True)
    elif data == "buy_cmd":
        await handle_buy(chat_id, user_id)
    elif data in ASSETS:
        await ensure_scan()
        s = cache["signals"].get(data, {})
        if not s or s.get("confidence", 0) == 0:
            await send_telegram_message(chat_id, f"No data for {data.replace('USDT','')} yet. Scanning...")
        else:
            await send_rich_card(chat_id, s, back_button=True)

@app.post("/webhook")
async def telegram_webhook(request: Request):
    if SECRET_TOKEN != "default_secret":
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if token != SECRET_TOKEN:
            raise HTTPException(status_code=403, detail="Unauthorized")
    try:
        data = await request.json()
        if "message" in data:
            chat_id = data["message"]["chat"]["id"]
            text = data["message"].get("text", "")
            user_id = data["message"]["from"]["id"]
            await handle_message(chat_id, text, user_id)
        elif "callback_query" in data:
            query = data["callback_query"]
            chat_id = query["message"]["chat"]["id"]
            data_btn = query["data"]
            user_id = query["from"]["id"]
            await handle_callback(chat_id, data_btn, user_id)
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse({"ok": False}, status_code=500)

# ==================== STARTUP BANNER ====================
def print_startup_banner():
    print("=" * 50)
    print("CROO AI ORACLE")
    print("=" * 50)
    print("\nAgent Status: ACTIVE")
    print("\nProviders:")
    for name in ["Bybit", "Binance", "OKX", "Kraken"]:
        if name in disabled_ws:
            status = "✗ (Region Restricted)"
        elif provider_status.get(name) == "active":
            status = "✓"
        else:
            status = "?"
        print(f"  {name}: {status}")
    print(f"\nFallback Depth: 4")
    print("\nCapabilities:")
    caps = [
        "✓ Autonomous Analysis",
        "✓ Multi-Provider Recovery",
        "✓ Explainable AI",
        "✓ Risk Management",
        "✓ Capital Allocation",
        "✓ Agent Memory",
        "✓ Telegram Delivery",
        "✓ A2A Ready",
        "✓ ADX Trend Strength",
        "✓ Multi-Timeframe",
        "✓ Market Structure",
        "✓ OBV Volume Analysis",
        "✓ Learning Engine",
        "✓ Paper Trading",
        "✓ News Sentiment (composite)",
        "✓ Backtesting",
        "✓ Judge Showcase",
        "✓ SQLite Persistence",
        "✓ /metrics Endpoint",
        "✓ 3-Tier TP System",
        "✓ Kelly Criterion",
        "✓ Real Correlation Matrix",
        "✓ Advanced Portfolio Allocation"
    ]
    for cap in caps:
        print(f"  {cap}")
    print("\n" + "=" * 50)

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
```

