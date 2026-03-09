"""Quick diagnostic: connect WS, send transcript_direct, print all replies."""
import asyncio
import json
import sys
sys.path.insert(0, ".")

import websockets


async def main():
    uri = "ws://localhost:8000/ws/test-diag-001"
    print(f"Connecting → {uri}")
    try:
        async with websockets.connect(uri, open_timeout=5) as ws:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            print(f"  ← ready: {msg}")

            payload = json.dumps({"type": "transcript_direct", "text": "Say exactly: Hi there"})
            await ws.send(payload)
            print(f"  → sent transcript_direct")

            for _ in range(40):
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=6)
                    if isinstance(raw, bytes):
                        print(f"  ← [binary {len(raw)} bytes]")
                        continue
                    m = json.loads(raw)
                    print(f"  ← {m.get('type'):25s} {str(m)[:100]}")
                    if m.get("type") in ("response_complete", "error"):
                        break
                except asyncio.TimeoutError:
                    print("  Timeout — no more messages")
                    break
    except Exception as exc:
        print(f"ERROR: {exc}")


asyncio.run(main())
