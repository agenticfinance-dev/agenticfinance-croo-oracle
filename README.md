🔮 CROO AI Oracle

Autonomous AI-powered crypto intelligence agent built for the CROO Network.

CROO AI Oracle provides explainable trading intelligence using live multi-exchange market data. It delivers **zone-based entries**, **fixed risk management**, and **agent-to-agent (A2A)** communication with CAP payment support.

Unlike traditional signal bots, CROO AI Oracle **never recommends entering at the current market price**. Every trade includes a calculated entry zone, stop loss, take-profit targets, confidence score, and reasoning.

Features

- Multi-provider market data
  - Bybit
  - Binance
  - OKX
  - Kraken
  - CoinGecko fallback

- Explainable AI
  - Confidence score
  - Bullish/Bearish reasoning
  - Why Not Now analysis
  - Missing conditions

- Smart Risk Management
  - Zone-based entries
  - ATR volatility filtering
  - ADX trend confirmation
  - Position sizing
  - Fixed 2.5:1 Risk/Reward

- Agent Features
  - A2A communication
  - CAP payment ready
  - Agent memory
  - Reputation scoring
  - Performance tracking

- Telegram Bot
  - Market scan
  - Best signal
  - Leaderboard
  - Explain signal
  - Portfolio statistics

Architecture

                   ┌──────────────┐
                   │ Telegram Bot │
                   └──────┬───────┘
                          │
                    FastAPI Backend
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
     Scanner         A2A API         REST API
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                   Trading Engine
                          │
      ┌──────────┬──────────┬──────────┬──────────┐
      │ Bybit    │ Binance  │ OKX      │ Kraken   │
      └──────────┴──────────┴──────────┴──────────┘
                          │
                     CoinGecko
                      (Fallback)

Supported Assets

- BTCUSDT
- ETHUSDT
- SOLUSDT
- XRPUSDT
- BNBUSDT
- AVAXUSDT
- DOGEUSDT
- TRXUSDT
- ADAUSDT
- LINKUSDT

REST Endpoints

Market

GET /scan
GET /oracle
GET /best_signal
GET /leaderboard
GET /market_summary

Analysis

GET /explain/{symbol}
GET /why/{symbol}
GET /backtest/{symbol}

Agent

POST /a2a
GET /judge_demo
GET /stats
GET /history
GET /billing

A2A Requests

Example:

json
POST /a2a

{
  "agent":"quant_bot",
  "request":"best_trade"
}


Response

json
{
  "status":"completed",
  "job_id":"...",
  "result":{
      "asset":"BTCUSDT",
      "decision":"BUY",
      "entry_zone":"...",
      "tp1":"...",
      "sl":"...",
      "confidence":84
  }
}


# CAP Pricing

| Service | Price |
|----------|-------|
| best_trade | Free (Judging) |
| market_intel | 0.005 USDC |
| premium_scan | 0.01 USDC |
| allocate_capital | 0.02 USDC |


Technology Stack

- Python
- FastAPI
- asyncio
- WebSockets
- SQLite
- Telegram Bot API
- NumPy
- aiohttp
- Render
- CROO Network
- CAP Protocol

Running Locally

bash
git clone https://github.com/agenticfinance-dev/agenticfinance-croo-oracle.git

cd agenticfinance-croo-oracle

pip install -r requirements.txt

python app.py


Live Demo

API

https://agenticfinance-croo-oracle.onrender.com

Telegram

https://t.me/CROOOracleBot

GitHub

https://github.com/agenticfinance-dev/agenticfinance-croo-oracle

YouTube

https://www.youtube.com/watch?v=58IqIJ_GVyI

CROO Store

https://agent.croo.network/agents/08add3b9-6a9c-4b83-8ca7-582a106a735d


Why CROO AI Oracle?

- Autonomous trading intelligence
- Explainable AI
- Agent-to-Agent communication
- CAP payment ready
- Multi-provider failover
- Production-ready architecture
- Zone-based execution
- Fixed risk management


License

MIT License


Built for the **CROO Network Hackathon 2026**.
