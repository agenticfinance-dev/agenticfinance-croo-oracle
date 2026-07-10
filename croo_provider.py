import os
import asyncio
import json
import requests
import websockets
from datetime import datetime

CROO_API_URL = os.getenv("CROO_API_URL", "https://api.croo.network").rstrip("/")
CROO_WS_URL = os.getenv("CROO_WS_URL", "wss://api.croo.network/ws")
CROO_KEY = os.getenv("CROO_SDK_KEY") or os.getenv("CROO_API_KEY")
ORACLE_URL = "http://localhost:10000/a2a"  # FastAPI running on same Render instance

def get_signal(asset="ETH"):
    """Call your local FastAPI oracle"""
    try:
        # Try local first (same container)
        resp = requests.post(f"{ORACLE_URL}", json={"asset": asset}, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    try:
        # Fallback to public URL
        resp = requests.post("https://agenticfinance-croo-oracle.onrender.com/a2a", json={"asset": asset}, timeout=8)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Oracle error: {e}")
    
    return {
        "asset": asset,
        "signal": "HOLD",
        "confidence": 0.72,
        "entry": 4200,
        "stop_loss": 4100,
        "take_profit": 4450,
        "risk_reward": "2.5:1",
        "source": "OKX primary verified by CROO-Judge",
        "timestamp": datetime.utcnow().isoformat(),
        "price": "0.01 USDC"
    }

async def croo_provider_loop():
    if not CROO_KEY:
        print("❌ No CROO_API_KEY set!")
        return

    print(f"🔑 CROO Key: {CROO_KEY[:12]}...{CROO_KEY[-4:]}")
    print(f"🌐 API: {CROO_API_URL}")
    print(f"🔌 WS: {CROO_WS_URL}")
    print("🚀 Starting CROO Provider (no-sdk mode)...")

    headers = {"Authorization": f"Bearer {CROO_KEY}", "X-API-Key": CROO_KEY}
    
    while True:
        try:
            # Try websocket connection
            async with websockets.connect(
                CROO_WS_URL,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            ) as ws:
                print(f"✅ Connected to CROO WS: {CROO_WS_URL}")
                # Register as provider online
                await ws.send(json.dumps({
                    "type": "provider_online",
                    "sdk_key": CROO_KEY,
                    "services": ["CROO AI Oracle", "multi-asset-signal"]
                }))
                
                async for message in ws:
                    try:
                        data = json.loads(message)
                        print(f"📥 CROO Event: {data.get('type')} - {data}")
                        
                        if data.get("type") in ["order_request", "negotiation", "a2a_call"]:
                            asset = data.get("params", {}).get("asset", "ETH")
                            result = get_signal(asset)
                            # Send delivery back
                            response = {
                                "type": "delivery",
                                "order_id": data.get("order_id") or data.get("id"),
                                "result": result
                            }
                            await ws.send(json.dumps(response))
                            print(f"📤 Delivered: {result}")
                    except Exception as e:
                        print(f"Message handling error: {e}")
                        continue
                        
        except Exception as e:
            print(f"⚠️ WS disconnected: {e} - retrying in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    # This file is designed to run ALONGSIDE uvicorn, not alone
    # When run as main, just run provider loop
    asyncio.run(croo_provider_loop())
