import os, asyncio, requests
from croo_sdk import CrooProvider, ServiceHandler

# Your existing oracle endpoint
ORACLE_URL = "https://agenticfinance-croo-oracle.onrender.com/a2a"

async def handle_signal(params):
    # Forward CROO request to your oracle
    try:
        r = requests.post(ORACLE_URL, json=params, timeout=20)
        return r.json()
    except Exception as e:
        return {"signal": "HOLD", "price": 0, "reason": str(e)}

async def main():
    provider = CrooProvider(
        api_url=os.getenv("CROO_API_URL", "https://api.croo.network"),
        ws_url=os.getenv("CROO_WS_URL", "wss://api.croo.network/ws"),
        sdk_key=os.getenv("CROO_SDK_KEY") or os.getenv("CROO_API_KEY")
    )
    
    # Register handlers for your 2 services
    provider.add_handler("croo-ai-oracle-multi-asset", handle_signal)
    
    print("🚀 CROO Provider Online - listening for orders...")
    await provider.start()

if __name__ == "__main__":
    asyncio.run(main())
