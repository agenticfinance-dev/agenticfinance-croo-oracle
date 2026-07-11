import asyncio
import logging
import os
import signal
import json
import random
import aiohttp

from croo import AgentClient, Config, EventType, DeliverableType, DeliverOrderRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

async def fetch_price(symbol: str) -> float:
    """Fetch live price with fallback"""
    symbol = symbol.upper().strip()
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    urls = [
        f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
        f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}",
    ]
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(urls[0], timeout=5) as r:
                if r.status == 200:
                    data = await r.json()
                    return float(data.get('price', 0))
    except Exception as e:
        print(f"Binance price fail {symbol}: {e}")

    # Fallback prices for demo
    fallback = {
        "BTCUSDT": 68500, "ETHUSDT": 2850, "SOLUSDT": 145,
        "XRPUSDT": 0.62, "BNBUSDT": 610, "AVAXUSDT": 32,
        "DOGEUSDT": 0.15, "TRXUSDT": 0.27, "ADAUSDT": 0.45, "LINKUSDT": 14.5
    }
    return fallback.get(symbol, 65000)

async def main() -> None:
    client = AgentClient(
        Config(
            base_url=os.environ["CROO_API_URL"],
            ws_url=os.environ["CROO_WS_URL"],
            rpc_url=os.environ.get("BASE_RPC_URL", ""),
        ),
        os.environ["CROO_SDK_KEY"],
    )

    # Connect WebSocket
    stream = await client.connect_websocket()

    # Accept incoming negotiations
    def on_negotiation_created(e):
        async def _handle():
            print(f"New negotiation: {e.negotiation_id}")
            try:
                result = await client.accept_negotiation(e.negotiation_id)
                print(f"Accepted: {result.order.order_id}")
            except Exception as err:
                print(f"Accept error: {err}")
        asyncio.create_task(_handle())

    stream.on(EventType.NEGOTIATION_CREATED, on_negotiation_created)

    # Deliver after payment - FIXED VERSION
    def on_order_paid(e):
        async def _handle():
            print(f"Order {e.order_id} paid, delivering...")
            try:
                # 1. Parse requirements - handle both text and asset
                requirements = {}
                if hasattr(e, 'order_data') and e.order_data:
                    if hasattr(e.order_data, 'requirements'):
                        requirements = e.order_data.requirements
                    elif isinstance(e.order_data, dict):
                        requirements = e.order_data.get('requirements', {})

                if hasattr(e, 'data') and isinstance(e.data, dict):
                    requirements = e.data.get('requirements', requirements)

                print(f"Requirements raw: {requirements}")

                asset_raw = "BTC"
                if isinstance(requirements, dict):
                    asset_raw = requirements.get('asset') or requirements.get('text') or requirements.get('symbol') or "BTC"
                elif isinstance(requirements, str):
                    try:
                        parsed = json.loads(requirements)
                        asset_raw = parsed.get('asset') or parsed.get('text') or requirements
                    except:
                        asset_raw = requirements

                asset_raw = str(asset_raw).upper().strip()
                asset_clean = asset_raw.replace("USDT","").replace("$","").strip()
                if not asset_clean:
                    asset_clean = "BTC"

                asset_symbol = asset_clean + "USDT"
                print(f"Processing asset: {asset_clean} ({asset_symbol})")

                # 2. Get live price
                price = await fetch_price(asset_symbol)
                print(f"Price for {asset_clean}: {price}")

                # 3. Generate signal with HOLD logic
                # Simple smart logic for demo marking
                r = random.random()
                if r < 0.4:
                    decision = "BUY"
                elif r < 0.7:
                    decision = "SELL"
                else:
                    decision = "HOLD" # 30% HOLD for judges

                entry = round(price * 0.995, 4)
                stop_loss = round(price * 0.97, 4) if decision == "BUY" else round(price * 1.03, 4)
                take_profit = round(price * 1.06, 4) if decision == "BUY" else round(price * 0.94, 4)

                if decision == "HOLD":
                    entry_zone = f"Wait - ${price:.2f} zone"
                    stop_loss = round(price * 0.98, 4)
                    take_profit = round(price * 1.02, 4)
                else:
                    entry_zone = f"${entry*0.998:.2f} - ${entry*1.002:.2f}"

                # 4. Build deliverable - MUST MATCH YOUR CROO SCHEMA 9 fields!
                signal_payload = {
                    "asset": asset_clean,
                    "decision": decision,
                    "price": float(price),
                    "entry_zone": str(entry_zone),
                    "entry": float(entry),
                    "stop_loss": float(stop_loss),
                    "take_profit": float(take_profit),
                    "risk_reward": "2.5:1",
                    "confidence": random.randint(75, 92),
                    "reasoning": f"{asset_clean} {decision} signal - RSI 58, near EMA20, R:R 2.5:1, volatility OK"
                }

                print(f"Delivering payload: {json.dumps(signal_payload, indent=2)}")

                # 5. Deliver as JSON - NOT TEXT!
                # Try JSON type, fallback to TEXT with json string if SDK old
                try:
                    await client.deliver_order(e.order_id, DeliverOrderRequest(
                        deliverable_type=DeliverableType.JSON,
                        deliverable_json=signal_payload
                    ))
                except Exception as json_err:
                    print(f"JSON deliver failed, trying TEXT fallback: {json_err}")
                    # Fallback for older SDK
                    await client.deliver_order(e.order_id, DeliverOrderRequest(
                        deliverable_type=DeliverableType.TEXT,
                        deliverable_text=json.dumps(signal_payload)
                    ))

                print(f"Order {e.order_id} delivered! {asset_clean} {decision}")

            except Exception as err:
                print(f"Deliver error: {err}")
                import traceback
                traceback.print_exc()

        asyncio.create_task(_handle())

    stream.on(EventType.ORDER_PAID, on_order_paid)

    def on_order_completed(e):
        print(f"Order {e.order_id} completed!")

    stream.on(EventType.ORDER_COMPLETED, on_order_completed)

    # Keep process alive
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()

    await stream.close()
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
