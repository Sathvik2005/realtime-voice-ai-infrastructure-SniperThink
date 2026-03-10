# Voice AI System

A **real-time, bidirectional voice conversation system** built from first principles.  
The user speaks; the AI listens, reasons, and responds — all with sub-800 ms latency.

The system is fully resilient: if OpenAI is unavailable or quota is exhausted, it falls back to a local Ollama model, and if that too is unavailable it falls back to a fast rule-based engine — so the system always responds.

---

## Architecture at a Glance

```
Browser                                  Backend (Python / FastAPI)
──────────────────────────────────────   ────────────────────────────────────────────
Microphone
  │
AudioWorklet (20 ms PCM frames)
  │                                        ┌─ WebRTC Server (aiortc)
  ├──── WebRTC (DTLS/SRTP/UDP) ──────────► │    48→16 kHz resample
  │                                        │
  │                                        ▼
  │                                      AudioRouter
  │                                        │
  │                                      VAD (Silero)
  │                                        │  speech_start / speech_end
  │                                        ▼
  │                                      ASR (Faster-Whisper)
  │                                        │  transcript
  │                                        ▼
  │                                      [RAG retrieval — optional]
  │                                        │  context
  │                                        ▼
  │                                      LLM (fallback chain)
  │                                        │  1. OpenAI API (cloud)
  │                                        │  2. Ollama (local)
  │                                        │  3. Rule-based (always works)
  │                                        │  token stream ──────────┐
  │                                        ▼                         │
  │                                      TTS (Piper streaming) ◄─────┘
  │                                        │  audio chunks
  │◄──── WebSocket binary ◄───────────────┘
  │
AudioPlayback (Web Audio API)
Speakers
```

---

## Features

- **Real-time WebRTC audio transport** — microphone audio travels over UDP with DTLS/SRTP encryption
- **Streaming VAD** — Silero VAD detects speech boundaries at 32 ms resolution
- **Streaming ASR** — Faster-Whisper with partial and final transcription modes
- **Streaming LLM with automatic fallback** — three-tier chain: OpenAI → Ollama (local) → Rule-based engine; the system always produces a response even without internet or API quota
- **Sentence-pipeline TTS** — Piper TTS begins synthesising before the full response is available
- **Interruption handling** — user speech cancels the current AI turn instantly
- **Conversational memory** — full multi-turn history maintained per session
- **RAG (optional)** — FAISS vector search augments LLM context with domain knowledge

---

## Project Structure

```
voice-ai-system/
├── backend/
│   ├── main.py                  FastAPI app + lifespan
│   ├── session_manager.py       Per-session state machine
│   ├── websocket_signaling.py   SDP/ICE/control message routing
│   ├── webrtc_server.py         aiortc peer connection management
│   ├── audio_router.py          Pipeline orchestrator
│   ├── vad.py                   Silero VAD wrapper
│   ├── asr_streaming.py         ASR providers: local Faster-Whisper + HF Gradio Whisper
│   ├── llm_engine.py            LLM fallback chain (OpenAI → Ollama → Rule-based)
│   ├── tts_streaming.py         Piper TTS streaming wrapper
│   └── interrupt_controller.py  asyncio task cancellation
├── frontend/
│   ├── index.html               Minimal single-page UI
│   ├── app.js                   Application controller
│   ├── webrtc_client.js         RTCPeerConnection + WebSocket
│   ├── audio_capture.js         Microphone + AudioWorklet
│   ├── audio_playback.js        Gapless TTS playback
│   └── audio_processor_worklet.js  AudioWorklet processor (audio thread)
├── rag/
│   ├── embeddings.py            sentence-transformers wrapper
│   ├── vector_store.py          FAISS index with persistence
│   └── retrieval.py             Query pipeline + directory ingestion
├── docs/
│   └── architecture.md          Detailed system design document
├── .env.example
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| Node.js | Not required (pure browser JS) |
| CUDA | Optional — CPU works for base Whisper model |

---

## Installation

### 1. Clone and create virtual environment

```bash
git clone <repo-url> voice-ai-system
cd voice-ai-system
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Piper TTS (optional — gTTS fallback is used otherwise)

```bash
# Linux x86-64 example
mkdir -p backend/models
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz
tar -xzf piper_linux_x86_64.tar.gz -C backend/

# Download a voice model
wget -O backend/models/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O backend/models/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 5. Silero VAD model (auto-downloaded on first run via torch.hub)

No manual step required — the model is fetched from GitHub on first startup.

### 6. (Optional) Populate the RAG knowledge base

```
mkdir -p rag/knowledge_base
# Copy .txt or .md files into rag/knowledge_base/
# They will be ingested automatically on first startup when RAG_ENABLED=true
```

---

## Running

```bash
cd backend
python main.py
```

Open your browser at: **http://localhost:8000**

Click **Start Conversation** and begin speaking.

---

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (optional — system works without it via fallback) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_MODEL` | `llama3` | Ollama model name (used if OpenAI is unavailable) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `ASR_MODE` | `local` | `local` (Faster-Whisper) or `hf` (HF Gradio Whisper) |
| `WHISPER_MODEL_SIZE` | `base` | Local Whisper model size: `tiny`, `base`, `small`, `medium` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `HF_WHISPER_SPACE` | `openai/whisper` | HF Space used when `ASR_MODE=hf` |
| `HF_TOKEN` | — | HuggingFace token for private spaces or higher rate limits |
| `TTS_VOICE` | `en_US-lessac-medium` | Piper voice model name |
| `PIPER_PATH` | `piper` | Path to piper executable |
| `PORT` | `8000` | HTTP/WebSocket port |
| `RAG_ENABLED` | `false` | Enable RAG context retrieval |
| `KNOWLEDGE_BASE_PATH` | `rag/knowledge_base` | Directory of .txt / .md files to ingest |

---

## ASR Modes

Two speech recognition backends are available, controlled by `ASR_MODE`:

| Mode | Value | How it works |
|---|---|---|
| **Local** (default) | `ASR_MODE=local` | Runs Faster-Whisper on your machine (CPU/GPU). No internet required after model download. |
| **Hugging Face** | `ASR_MODE=hf` | Calls the [openai/whisper](https://huggingface.co/spaces/openai/whisper) Gradio Space over HTTPS. No local model needed — requires internet. |

### Using HF Whisper mode

```bash
# 1. Install the client
pip install gradio_client

# 2. Set in .env
ASR_MODE=hf
HF_WHISPER_SPACE=openai/whisper   # default
# HF_TOKEN=hf_...                 # optional, for private spaces or higher rate limits
```

The HF Space accepts WAV audio and returns a transcript in one round-trip. Because it is request/response (not streaming), `partial()` is a no-op — the live typing indicator is driven by the browser's `SpeechRecognition` API instead, which still works in parallel.

---

## LLM Fallback Chain

The system tries three providers in order and falls back automatically on any error:

```
1. OpenAI API          — cloud GPT-4o-mini (or any model)
      │  400/401/429/timeout → fallback
      ▼
2. Ollama (local)      — llama3 (or any pulled model) at localhost:11434
      │  connection refused / timeout → fallback
      ▼
3. Rule-based engine   — keyword regex, hardcoded replies, always responds
```

The active provider name is sent with every response (`msg.provider`) and displayed as a badge in the UI when OpenAI is **not** the provider (so normal users see nothing extra, but you can tell when a fallback is active).

### Setting up Ollama (optional but recommended)

```bash
# Install Ollama — https://ollama.com
# Then pull a model:
ollama pull llama3        # ~4 GB one-time download
# Ollama runs on http://localhost:11434 by default
```

Set `OLLAMA_MODEL` and `OLLAMA_URL` in your `.env` (see `.env.example`).

---



The system targets **< 800 ms end-to-end** response latency.

### Breakdown

| Stage | Typical Latency | Technique |
|---|---|---|
| Audio capture | ~20 ms | 20 ms AudioWorklet frames |
| WebRTC transport | ~10–50 ms | UDP, no TCP head-of-line blocking |
| VAD speech detection | ~96 ms | 3× 32 ms frame confirmation |
| VAD silence detection | ~480 ms | Configurable trailing silence |
| ASR (Whisper base) | ~150–300 ms | CPU int8 quantisation |
| LLM first token | ~100–400 ms | Streaming response |
| TTS first sentence | ~50–150 ms | Piper local synthesis |
| Audio scheduling | ~20 ms | Gapless AudioBufferSourceNode |

**Total budget**: ~930 ms worst-case → reduced to ~700 ms by overlapping silence detection with connection setup time and by starting TTS on the first sentence before LLM finishes.

### Parallelism

```
time ──────────────────────────────────────────────────────►
     [VAD silence 480ms][ASR][LLM token 1][TTS sentence 1]
                                          [LLM token 2 3…]
                                                      [TTS sentence 2]
```

LLM and TTS run concurrently via two asyncio tasks connected by a Queue. Playback of sentence 1 begins while sentence 2 is still being synthesised.

---

## Interruption Handling

1. User speaks while AI audio is playing
2. Browser AudioWorklet detects RMS > threshold
3. `app.js` calls `AudioPlayback.stop()` — playback silenced instantly
4. `app.js` sends `{ type: "interrupt" }` over WebSocket
5. Backend `InterruptController.interrupt()`:
   - Sets `session.interrupt_event`
   - Cancels the running `asyncio.Task` (pipeline)
   - Both LLM and TTS generators check the event and exit cleanly
6. Session state returns to IDLE; next utterance begins immediately

---

## RAG Integration

When `RAG_ENABLED=true`:

1. On startup, all `.txt` / `.md` files in `KNOWLEDGE_BASE_PATH` are chunked (300 words, 30-word overlap) and embedded using `all-MiniLM-L6-v2`
2. Embeddings are stored in a FAISS flat index (cosine similarity)
3. At query time, the user transcript is embedded and the top-4 chunks above similarity threshold 0.35 are retrieved
4. The retrieved text is prepended to the LLM system prompt as context

The focus is on **architecture correctness** over dataset quality.

---

## Known Limitations

- **Single-server only** — session state is in-process; horizontal scaling requires Redis + sticky routing
- **English only** — Whisper can transcribe other languages; change `language="en"` in `asr_streaming.py`
- **Fully offline capable** — Ollama + Piper + Whisper (with CUDA) means no internet connection is needed at all after initial model downloads
- **Piper model download** — manual step; gTTS fallback works but adds network latency
- **No TURN server** — WebRTC may fail through strict NAT; add Coturn for production

---

## Architecture Document

See [docs/architecture.md](docs/architecture.md) for the full design document including state machine diagrams, latency analysis, and scalability notes.

---

## License

MIT
