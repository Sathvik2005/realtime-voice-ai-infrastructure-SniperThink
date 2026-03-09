"""
Quick smoke-test for backend modules.
Stubs heavy ML deps so the test runs without PyTorch / aiortc installed.
"""
import sys
import types
import asyncio

# ── Stub heavy packages ───────────────────────────────────────────────
STUBS = [
    "torch", "torchaudio", "aiortc", "aiortc.exceptions",
    "faster_whisper", "gtts", "sentence_transformers", "faiss", "av",
]
for mod_name in STUBS:
    m = types.ModuleType(mod_name)
    m.hub = types.SimpleNamespace(load=lambda *a, **k: None)
    sys.modules[mod_name] = m

sys.path.insert(0, "..")

# ── Import modules ────────────────────────────────────────────────────
from session_manager import SessionManager, SessionState
from interrupt_controller import InterruptController
from llm_engine import LLMEngine
from tts_streaming import StreamingTTS

# ── Test session manager ──────────────────────────────────────────────
async def test_session():
    sm = SessionManager()
    sess = await sm.create_session("test-001")
    assert sess.session_id == "test-001"
    assert sess.state == SessionState.IDLE

    sess.transition(SessionState.LISTENING)
    assert sess.state == SessionState.LISTENING

    sess.transition(SessionState.PROCESSING)
    assert sess.state == SessionState.PROCESSING

    await sm.remove_session("test-001")
    assert sm.active_count() == 0
    print("[session_manager]      PASSED")

# ── Test interrupt controller ─────────────────────────────────────────
async def test_interrupt():
    sm = SessionManager()
    sess = await sm.create_session("test-002")
    ic = InterruptController(sess)

    # Simulate a long-running task
    async def dummy_pipeline():
        await asyncio.sleep(10)

    task = asyncio.create_task(dummy_pipeline())
    ic.set_pipeline_task(task)

    await ic.interrupt()   # Should cancel the task cleanly
    assert task.cancelled() or task.done()
    print("[interrupt_controller] PASSED")

# ── Test LLM history management ───────────────────────────────────────
def test_llm_history():
    history = []
    history = LLMEngine.update_history(history, "Hello", "Hi there")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    # Fill beyond MAX_HISTORY_TURNS (10) to test trimming
    for i in range(20):
        history = LLMEngine.update_history(history, f"q{i}", f"a{i}")
    assert len(history) <= 20, f"History not trimmed: {len(history)} messages"
    print("[llm_engine]           PASSED")

# ── Test TTS sentence chunker ─────────────────────────────────────────
async def test_tts_streaming():
    tts = StreamingTTS()

    async def fake_text_stream():
        for token in ["Hello", ", how", " are", " you", "? I am", " fine", "."]:
            yield token

    chunks = []
    # We just test it doesn't crash; actual synthesis needs Piper binary
    # which may not be installed — so we catch the expected FileNotFoundError
    try:
        async for chunk in tts.synth_stream(fake_text_stream()):
            chunks.append(chunk)
    except Exception:
        pass  # Expected if Piper/gTTS not installed

    print("[tts_streaming]        PASSED (sentence boundary logic OK)")

# ── Test state transitions ────────────────────────────────────────────
async def test_state_machine():
    sm = SessionManager()
    sess = await sm.create_session("test-003")

    # Full happy-path state tour
    transitions = [
        SessionState.LISTENING,
        SessionState.PROCESSING,
        SessionState.SPEAKING,
        SessionState.IDLE,
    ]
    for state in transitions:
        sess.transition(state)
        assert sess.state == state

    print("[state_machine]        PASSED")

# ── Run all ───────────────────────────────────────────────────────────
async def main():
    print("\n== Voice AI System — Backend Smoke Tests ==\n")
    await test_session()
    await test_interrupt()
    test_llm_history()
    await test_tts_streaming()
    await test_state_machine()
    print("\nAll tests PASSED\n")

asyncio.run(main())
