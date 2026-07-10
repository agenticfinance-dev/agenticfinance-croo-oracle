import asyncio
import logging
import os
import signal

from croo import AgentClient, Config, EventType, DeliverableType, DeliverOrderRequest, Event

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


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
    def on_negotiation_created(e: Event) -> None:
        async def _handle() -> None:
            print(f"New negotiation: {e.negotiation_id}")
            try:
                result = await client.accept_negotiation(e.negotiation_id)
                print(f"Order created: {result.order.order_id}")
            except Exception as err:
                print(f"accept error: {err}")
        asyncio.create_task(_handle())

    stream.on(EventType.NEGOTIATION_CREATED, on_negotiation_created)

    # Deliver after payment
    def on_order_paid(e: Event) -> None:
        async def _handle() -> None:
            print(f"Order {e.order_id} paid, delivering...")
            try:
                await client.deliver_order(e.order_id, DeliverOrderRequest(
                    deliverable_type=DeliverableType.TEXT,
                    deliverable_text='{"analysis": "done", "score": 95}',
                ))
                print(f"Order {e.order_id} delivered!")
            except Exception as err:
                print(f"deliver error: {err}")
        asyncio.create_task(_handle())

    stream.on(EventType.ORDER_PAID, on_order_paid)

    def on_order_completed(e: Event) -> None:
        print(f"Order {e.order_id} completed!")

    stream.on(EventType.ORDER_COMPLETED, on_order_completed)

    # Keep process alive
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    await stop.wait()

    await stream.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
