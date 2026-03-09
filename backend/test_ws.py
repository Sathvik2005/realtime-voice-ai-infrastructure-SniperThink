"""WebSocket handshake test — run from project root."""
import asyncio
import json
import sys

async def test_ws():
    try:
        import websockets
    except ImportError:
        print("websockets not found — skipping WS test")
        return

    uri = "ws://localhost:8000/ws/test-manual-01"
    try:
        async with websockets.connect(uri) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            print(f"  Server sent: {msg}")
            assert msg["type"] == "ready", f"Expected 'ready', got: {msg}"
            assert msg["session_id"] == "test-manual-01"

            # Send interrupt — should not error
            await ws.send(json.dumps({"type": "interrupt"}))
            await asyncio.sleep(0.3)
            print("  WebSocket handshake:  PASSED")
    except Exception as exc:
        print(f"  WebSocket test FAILED: {exc}")
        sys.exit(1)

async def test_multi_session():
    try:
        import websockets
    except ImportError:
        return

    sids = ["sess-A", "sess-B", "sess-C"]
    conns = []
    for sid in sids:
        ws = await websockets.connect(f"ws://localhost:8000/ws/{sid}")
        raw = await asyncio.wait_for(ws.recv(), timeout=5)
        msg = json.loads(raw)
        assert msg["type"] == "ready"
        assert msg["session_id"] == sid
        conns.append(ws)

    for ws in conns:
        await ws.close()

    print("  Multi-session concurrency: PASSED")

async def main():
    print("\n== WebSocket Integration Tests ==\n")
    await test_ws()
    await test_multi_session()
    print("\nAll WebSocket tests PASSED\n")

asyncio.run(main())
