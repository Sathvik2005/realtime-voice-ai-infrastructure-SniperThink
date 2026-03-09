# Architecture — Voice AI System

## 1. System Overview

This system implements a **real-time, bidirectional voice conversation pipeline** running across a browser frontend and a Python backend. The architecture is designed around three principles:

- **Streaming everywhere**: no stage waits for a previous stage to complete before beginning
- **Pipeline parallelism**: LLM generation and TTS synthesis run concurrently
- **Clean separation of concerns**: each module owns exactly one responsibility

---

## 2. High-Level Architecture Diagram

```
┌────────────────────────── BROWSER ──────────────────────────────────┐
│                                                                      │
│  Microphone                                                          │
│     │                                                                │
│  AudioWorklet (20 ms frames, float32→int16)                         │
│     │                           ▲                                    │
│     │  WebRTC (DTLS/SRTP/UDP)   │  WebSocket (binary — TTS audio)   │
│     │                           │                                    │
│  RTCPeerConnection           AudioPlayback (Web Audio API)          │
│                                  ▲                                   │
│  WebSocket ─────────── control events (JSON)                        │
└─────────────────────────────────────────────────────────────────────┘
         │ (WebRTC audio track)          │ (WS binary + JSON)
         ▼                               │
┌────────────────────────── BACKEND ──────────────────────────────────┐
│                                                                      │
│  WebRTC Server (aiortc)                                              │
│     │  48 kHz → 16 kHz resample                                     │
│     ▼                                                                │
│  AudioRouter.process_audio_chunk()                                   │
│     │                                                                │
│  ┌──▼──────────────────────────────────────────────────────────┐    │
│  │  VADEngine  (Silero VAD, per-session state machine)          │    │
│  │     ├── speech_start → transition to LISTENING              │    │
│  │     │                  reset ASR buffer                      │    │
│  │     └── speech_end   → transition to PROCESSING             │    │
│  │                        launch _run_pipeline()               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────── _run_pipeline() ────────────────────────┐    │
│  │                                                              │    │
│  │  1. StreamingASR.final()    ←  Faster-Whisper               │    │
│  │          │ transcript                                        │    │
│  │          ▼                                                   │    │
│  │  2. RAG retrieval (optional)  ←  FAISS + sentence-xformers  │    │
│  │          │ context string                                    │    │
│  │          ▼                                                   │    │
│  │  3. LLMEngine.stream_response()  ←  OpenAI streaming        │    │
│  │          │ token stream ──────────────────────────────┐     │    │
│  │          ▼                                            │     │    │
│  │  4. StreamingTTS.synth_stream()  ←  Piper TTS         │     │    │
│  │          │ audio chunks  ◄──────────────────────────  │     │    │
│  │          ▼                                                   │    │
│  │  5. _send_audio() → WebSocket binary frames                  │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Responsibilities

### 3.1 Frontend

| File | Responsibility |
|---|---|
| `audio_processor_worklet.js` | AudioWorklet processor — converts float32 PCM to int16, emits 20 ms frames |
| `audio_capture.js` | Acquires microphone, builds AudioContext chain, exposes `onFrame` callback |
| `webrtc_client.js` | Manages RTCPeerConnection lifetime, SDP/ICE exchange, binary audio receive |
| `audio_playback.js` | Schedules AudioBufferSourceNodes back-to-back; instant stop on interrupt |
| `app.js` | Wires all modules; owns UI state machine; detect interruptions via RMS |

### 3.2 Backend

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, lifespan (model loading), WebSocket endpoint, health check |
| `session_manager.py` | Per-session state machine, conversation history, async coordination |
| `websocket_signaling.py` | JSON message router — SDP, ICE, interrupt, direct audio fallback |
| `webrtc_server.py` | aiortc RTCPeerConnection, audio track consumer, 48→16 kHz resample |
| `audio_router.py` | Central orchestrator — routes chunks through VAD→ASR→LLM→TTS |
| `vad.py` | Silero VAD per-session state (shared model); emits speech_start/end events |
| `asr_streaming.py` | Faster-Whisper; per-session audio buffer; partial + final transcription |
| `llm_engine.py` | OpenAI client; streaming token generator; conversation history management |
| `tts_streaming.py` | Piper TTS subprocess; sentence-boundary batch synthesis; gTTS fallback |
| `interrupt_controller.py` | asyncio.Task cancellation; clears TTS queue on interrupt |

### 3.3 RAG (optional)

| File | Responsibility |
|---|---|
| `embeddings.py` | sentence-transformers embedding engine (shared model) |
| `vector_store.py` | FAISS IndexFlatIP; add/search/persist |
| `retrieval.py` | Query → embed → search → format context string; auto-ingest on startup |

---

## 4. Turn-Taking State Machine

```
         ┌──────┐
    ┌───►│ IDLE │◄─────────────────────────────────────────┐
    │    └──┬───┘                                           │
    │       │ VAD: speech_start                             │
    │       ▼                                               │
    │    ┌──────────┐                                       │
    │    │ LISTENING│  ← audio buffered to ASR              │
    │    └──┬───────┘                                       │
    │       │ VAD: speech_end                               │
    │       ▼                                               │
    │    ┌────────────┐                                     │
    │    │ PROCESSING │  ← ASR running                      │
    │    └──┬─────────┘                                     │
    │       │ transcript ready                              │
    │       ▼                                               │
    │    ┌──────────┐                                       │
    │    │ SPEAKING │  ← LLM streaming + TTS streaming      │
    │    └──┬───────┘                                       │
    │       │                             interrupt?        │
    │       │ response_complete ─────────────────────────►  │
    └───────┘
```

Interruption path:
- User's mic RMS exceeds threshold while state == SPEAKING
- Frontend sends `{ type: "interrupt" }` over WebSocket
- Backend: `InterruptController.interrupt()` → sets `interrupt_event`, cancels pipeline task
- Frontend: `AudioPlayback.stop()` immediately silences speakers

---

## 5. Streaming Pipeline Design

### Why sentence-boundary TTS?

The LLM streams tokens. Feeding one token at a time to TTS is too slow to synthesise. Feeding the full response is too slow to start playback. The solution: buffer tokens until a sentence boundary (`. ! ?`) is detected, then immediately synthesise that sentence. This achieves:

- Playback starts within ~200 ms of first sentence completion
- Subsequent sentences are synthesised while the first plays

### LLM + TTS Concurrency

Both `llm_worker` and `tts_worker` run as concurrent asyncio tasks connected by an `asyncio.Queue`. This means:

```
LLM: token token token . ──────────────────────────────────►
                       │
                       └─► TTS: synthesise sentence 1
                                │
                                └─► audio stream ─► browser
LLM:          token token token ! ──────────────────────────►
                               │
                               └─► TTS: synthesise sentence 2 ...
```

---

## 6. Latency Analysis

### Target: < 800 ms end-to-end

| Stage | Budget | Strategy |
|---|---|---|
| Audio capture → WebRTC | ~20 ms | 20 ms frames; no buffering |
| Network (WebRTC) | ~10–50 ms | UDP; DTLS/SRTP; no TCP head-of-line blocking |
| VAD detection | ~30–100 ms | 32 ms Silero frames; 3-frame confirmation (~96 ms) |
| Silence detection | ~480 ms | Configurable; trade-off vs false triggers |
| ASR (Faster-Whisper base) | ~100–300 ms | Runs on full utterance; beam_size=5 |
| LLM first token | ~100–400 ms | GPT-4o-mini; streaming |
| TTS first sentence | ~100–200 ms | Piper: ~50 ms per sentence on CPU |
| Audio buffering (playback) | ~20 ms | Gapless scheduled AudioBufferSourceNodes |

**Dominant costs**: VAD silence trailing (480 ms) + ASR (~200 ms) + LLM first token (~200 ms). 

Optimisation levers:
- Reduce `MIN_SILENCE_FRAMES` (risk: premature speech-end detection)
- Use `whisper-tiny` or `distil-whisper` (lower accuracy)
- Use local LLM with speculative decoding
- Run ASR/VAD on GPU

---

## 7. Design Decisions

### WebRTC for input, WebSocket for TTS output

WebRTC provides low-jitter, UDP-based transport with automatic echo cancellation. For **outbound TTS audio**, a WebSocket binary stream is simpler to control: we can cancel it instantly, apply chunking aligned to synthesis, and avoid the complexity of injecting synthesised audio into a WebRTC MediaStreamTrack.

### Shared models, per-session state

Loading Silero VAD and Faster-Whisper once and sharing the model tensors across sessions avoids memory duplication. Each session maintains its own state wrappers (VADEngine, StreamingASR) so there is no cross-session data leakage.

### asyncio throughout

The entire backend is single-threaded asyncio. Blocking operations (model inference) are offloaded to a thread pool via `loop.run_in_executor()`. This keeps the event loop free for WebSocket I/O without requiring multi-process coordination.

---

## 8. Known Trade-offs

| Trade-off | Choice | Rationale |
|---|---|---|
| ASR accuracy vs latency | `base` model, beam_size=5 | Good balance for demo; tunable |
| TTS naturalness | Piper (fast/local) vs cloud | Piper avoids API costs and network latency |
| Silence detection | 480 ms trailing silence | Below this users feel "cut off" |
| WebSocket TTS vs WebRTC track | WebSocket | Simpler interrupt control |
| Local LLM vs API | OpenAI API | Availability; replaceable with Ollama |
| Single-server | Yes | Horizontal scaling requires sticky sessions or shared state (Redis) |

---

## 9. Scalability Notes

For production scale:

- Move session state to Redis for multi-node deployments
- Use a message queue (NATS/Kafka) between the WebSocket gateway and pipeline workers
- Run VAD/ASR on GPU workers behind a gRPC service
- Use a CDN / edge WebRTC relay (Cloudflare Calls or Janus) for global low-latency
