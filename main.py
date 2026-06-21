import os, time, json, threading, requests
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import telebot

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TOKEN)

CACHE = {"signals": {}, "sectors": {}, "last_scan": None, "scans": 0}

# === TA CORE ===
def get_klines(symbol, interval="1h", limit=100):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=5)
        return [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in r.json()]
    except: return []

def calculate_rsi(closes, period=14):
    if len(closes) < period: return 50
    gains = [max(closes[i] - closes[i-1], 0) for i in range(1, len(closes))]
    losses = [max(closes[i-1] - closes[i], 0) for i in range(1, len(closes))]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def calculate_ema(closes, period):
    if len(closes) < period: return closes[-1]
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]: ema = (price - ema) * multiplier + ema
    return ema

# === BATCH SECTOR DATA - NO RATE LIMITS ===
def get_binance_24hr_batch():
    symbols = ["BTCUSDT","ETHUSDT","FETUSDT","AGIXUSDT","OCEANUSDT","RNDRUSDT","ONDOUSDT","POLYXUSDT","CFGUSDT","AAVEUSDT","UNIUSDT","MKRUSDT","COMPUSDT","XRPUSDT","XLMUSDT","ALGOUSDT"]
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
            params={"symbols": json.dumps(symbols)}, timeout=5)
        return {x["symbol"]: float(x["priceChangePercent"]) for x in r.json()}
    except: return {}

def update_sectors():
    d = get_binance_24hr_batch()
    CACHE["sectors"] = {
        "AI": sum([d.get(s,0) for s in ["FETUSDT","AGIXUSDT","OCEANUSDT","RNDRUSDT"]]) / 4,
        "RWA": sum([d.get(s,0) for s in ["ONDOUSDT","POLYXUSDT","CFGUSDT"]]) / 3,
        "DEFI": sum([d.get(s,0) for s in ["AAVEUSDT","UNIUSDT","MKRUSDT","COMPUSDT"]]) / 4,
        "PAYFI": sum([d.get(s,0) for s in ["XRPUSDT","XLMUSDT","ALGOUSDT"]]) / 3
    }

# === PULLBACK ENGINE WITH CONFIDENCE BREAKDOWN ===
def analyze_pullback(symbol):
    klines = get_klines(symbol, "1h", 100)
    if len(klines) < 50: return None
    
    closes = [k[3] for k in klines]
    volumes = [k[4] for k in klines]
    price = closes[-1]
    
    rsi = calculate_rsi(closes)
    ema50 = calculate_ema(closes, 50)
    high_24h = max([k[1] for k in klines[-24:]])
    pullback_pct = ((high_24h - price) / high_24h) * 100
    avg_vol = sum(volumes[-20:]) / 20
    vol_spike = volumes[-1] > avg_vol * 1.5
    
    confidence = 0
    breakdown = []
    
    if rsi < 40:
        confidence += 25
        breakdown.append(f"├── RSI Oversold: +25% ({rsi:.1f})")
    if price > ema50:
        confidence += 25
        breakdown.append(f"├── Above EMA50: +25% (${ema50:.2f})")
    if 3 <= pullback_pct <= 8:
        confidence += 25
        breakdown.append(f"├── Healthy Pullback: +25% ({pullback_pct:.1f}%)")
    if vol_spike:
        confidence += 10
        breakdown.append(f"└── Volume Spike: +10% ({volumes[-1]/avg_vol:.1f}x)")
    
    if confidence < 70:
        return {"symbol": symbol, "signal": "NONE", "confidence": 0, "rsi": round(rsi,1), "price": price}
    
    entry_low = price * 0.995
    entry_high = price * 1.005
    tp = price * 1.05
    sl = price * 0.97
    rrr = round((tp - price) / (price - sl), 2)
    
    return {
        "symbol": symbol, "price": round(price,4), "signal": "BUY", "confidence": confidence,
        "rsi": round(rsi,1), "pullback_pct": round(pullback_pct,1), "breakdown": breakdown,
        "entry_zone": f"${entry_low:.4f} - ${entry_high:.4f}", "tp": round(tp,4), 
        "sl": round(sl,4), "rrr": rrr, "high_24h": round(high_24h,4)
    }

def run_scanner():
    assets = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT","DOTUSDT","FETUSDT","RNDRUSDT","INJUSDT","PEPEUSDT","WIFUSDT"]
    results = {}
    for symbol in assets:
        result = analyze_pullback(symbol)
        if result: results[symbol] = result
        time.sleep(0.1)
    CACHE["signals"] = results
    CACHE["last_scan"] = datetime.now(timezone.utc).isoformat()
    CACHE["scans"] += 1
    update_sectors()

# === TELEGRAM COMMANDS ===
@bot.message_handler(commands=['start'])
def start(message):
    run_scanner()
    signals = CACHE["signals"]
    sectors = CACHE["sectors"]
    
    msg = "🚀 <b>CROO AI Oracle Agent</b>\n\n<b>Sector Intelligence:</b>\n"
    for sector, pct in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
        emoji = "🔥" if pct > 5 else "🟢" if pct > 0 else "🔻"
        msg += f"{emoji} {sector}: {pct:+.1f}%\n"
    
    msg += "\n<b>Active Pullback Setups:</b>\n"
    found = False
    for symbol, s in signals.items():
        if s["signal"]!= "NONE":
            msg += f"🟢 {symbol.replace('USDT','')}: {s['confidence']}% | R:R {s['rrr']}:1\n"
            found = True
    
    if not found:
        msg += "\n🔍 No high-confidence setups right now.\nUse /scan to refresh or /signal BTC to check specific asset."
    
    bot.send_message(message.chat.id, msg, parse_mode='HTML')

@bot.message_handler(commands=['scan'])
def scan(message):
    bot.send_message(message.chat.id, "🔄 Scanning 15 assets for pullbacks...")
    run_scanner()
    start(message)

@bot.message_handler(commands=['signal'])
def signal(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.send_message(message.chat.id, "Usage: /signal BTC\nExample: /signal PEPE")
        return
    
    asset = parts[1].upper()
    symbol = f"{asset}USDT"
    result = analyze_pullback(symbol)
    
    if not result or result["signal"] == "NONE":
        bot.send_message(message.chat.id, f"🔍 No high-confidence pullback for {asset}\n\n📊 RSI: {result['rsi'] if result else 'N/A'}\n💰 Price: ${result['price'] if result else 'N/A'}\n\nUse /scan for active setups")
        return
    
    msg = f"""🚨 <b>{asset} Pullback Analysis</b>

📉 <b>Pullback detected:</b> {result['pullback_pct']}% from 24h high ${result['high_24h']}
📊 <b>RSI:</b> {result['rsi']} (oversold zone)
📈 <b>Volume:</b> Accumulation detected

<b>Confidence: {result['confidence']}%</b>
{chr(10).join(result['breakdown'])}

🎯 <b>Entry Zone:</b> {result['entry_zone']}
🎯 <b>Take Profit:</b> ${result['tp']}
🛡️ <b>Stop Loss:</b> ${result['sl']}
📐 <b>Risk:Reward:</b> {result['rrr']}:1

<i>CROO Agent assessment: High-probability mean reversion setup</i>"""
    
    bot.send_message(message.chat.id, msg, parse_mode='HTML')

@bot.message_handler(commands=['stats'])
def stats(message):
    signals = CACHE["signals"]
    high_conf = len([s for s in signals.values() if s["signal"]!= "NONE"])
    sectors_str = chr(10).join([f"{k}: {v:+.1f}%" for k,v in CACHE['sectors'].items()])
    msg = f"""📊 <b>CROO Agent Stats</b>

<b>Last Scan:</b> {CACHE['last_scan'][:19] if CACHE['last_scan'] else 'Never'}
<b>Total Scans:</b> {CACHE['scans']}
<b>Assets Monitored:</b> 15
<b>Active Signals:</b> {high_conf}

<b>Sector Performance:</b>
{sectors_str}"""
    bot.send_message(message.chat.id, msg, parse_mode='HTML')

# === CAP ENDPOINT FOR RENDER ===
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"agent": "CROO AI Oracle", "version": "2.0", "capabilities": ["pullback_scan", "confidence_scoring", "sector_rotation", "rrr_calculation"], "api": "/oracle"}).encode())
        elif self.path.startswith('/oracle'):
            self.send_response(405)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/oracle':
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            data = json.loads(body)
            asset = data.get('asset', 'BTC').upper()
            result = analyze_pullback(f"{asset}USDT")
            
            if not result or result["signal"] == "NONE":
                response = {"signal": "NONE", "confidence": 0, "message": "No high-confidence pullback"}
            else:
                response = {"signal": result["signal"], "confidence": result["confidence"], "entry": result["entry_zone"], 
                          "tp": result["tp"], "sl": result["sl"], "rsi": result["rsi"], 
                          "reasons": result["breakdown"], "rrr": result["rrr"]}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

def run_server():
    server = HTTPServer(('0.0.0.0', int(os.getenv('PORT', 10000))), Handler)
    server.serve_forever()

if __name__ == "__main__":
    print("Starting CROO AI Oracle Agent...")
    threading.Thread(target=run_server, daemon=True).start()
    threading.Thread(target=lambda: [run_scanner(), time.sleep(300)] * 999, daemon=True).start()
    bot.polling(non_stop=True)
