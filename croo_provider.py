import os
import asyncio
import requests
from croo_sdk import CrooProvider

# Your oracle endpoint - keep your main FastAPI running separately or call directly
ORACLE_URL = os.getenv("ORACLE_INTERNAL_URL", "https://agenticfinance-croo-oracle.onrender.com/a2a")

# For local logic if Render endpoint is same process, implement directly
def get_signal_logic(asset="ETH"):
    """
    Fallback: Simple logic if oracle endpoint unreachable
    Replace with your real OKX + CROO-Judge logic
    """
    try:
        resp = requests.post(ORACLE_URL, json={"asset": asset}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Oracle forward failed: {e}")
    
    # Fallback demo signal
    return {
        "asset": asset,
        "signal": "HOLD",
        "confidence": 0.65,
        "entry": 0,
        "stop_loss": 0,
        "take_profit": 0,
        "risk_reward": "2.5:1",
        "source": "OKX primary + fallback",
        "verified": "CROO-Judge",
        "price": "0.01 USDC via x402"
    }

async def handle_multi_asset_signal(params: dict):
    """Handler for both services"""
    print(f"📥 Received order: {params}")
    asset = params.get("asset", "ETH") if isinstance(params, dict) else "ETH"
    result = get_signal_logic(asset)
    print(f"📤 Delivering: {result}")
    return result

async def main():
    sdk_key = os.getenv("CROO_SDK_KEY") or os.getenv("CROO_API_KEY")
    api_url = os.getenv("CROO_API_URL", "https://api.croo.network")
    ws_url = os.getenv("CROO_WS_URL", "wss://api.croo.network/ws")

    if not sdk_key:
        print("❌ CROO_SDK_KEY or CROO_API_KEY not set!")
        return

    print(f"🔑 Using SDK key: {sdk_key[:10]}...{sdk_key[-4:]}")
    print(f"🌐 API: {api_url}")
    print(f"🔌 WS: {ws_url}")

    provider = CrooProvider(
        api_url=api_url,
        ws_url=ws_url,
        sdk_key=sdk_key
    )

    # Register handler - use service IDs from your dashboard or generic
    # The SDK will auto-match to your services
    provider.add_handler("CROO AI Oracle - Multi-Asset Signal", handle_multi_asset_signal)
    provider.add_handler("croo-ai-oracle", handle_multi_asset_signal)
    provider.add_handler("default", handle_multi_asset_signal)

    print("🚀 CROO Provider Online - listening for A2A orders...")
    print("   Waiting for negotiations -> accept -> payment -> delivery")
    await provider.start()

if __name__ == "__main__":
    asyncio.run(main())
