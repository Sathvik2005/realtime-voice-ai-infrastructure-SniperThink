"""Test OpenAI API key and print the real error."""
import asyncio
import os
import sys
sys.path.insert(0, ".")

from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import openai

async def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    print(f"Key: {key[:12]}...{key[-6:]} (len={len(key)})")
    
    client = openai.AsyncOpenAI(api_key=key)
    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            stream=True,
            max_tokens=20,
        )
        result = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                result += delta
        print(f"SUCCESS: '{result}'")
    except Exception as exc:
        print(f"ERROR TYPE : {type(exc).__name__}")
        print(f"ERROR MSG  : {exc}")

asyncio.run(main())
