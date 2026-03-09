/**
 * audio_playback.js
 *
 * Plays streaming TTS audio received from the backend over WebSocket.
 *
 * Design:
 *  • Incoming binary frames (after stripping the 1-byte header) are
 *    pushed into a queue.
 *  • A continuous "pump" loop drains the queue, decodes each chunk via
 *    AudioContext.decodeAudioData (MP3 from gTTS) or interprets it as
 *    raw s16le PCM (Piper), then schedules AudioBufferSourceNodes
 *    back-to-back for seamless gapless playback.
 *  • Calling stop() cancels all pending sources immediately, providing
 *    instant interruption.
 *
 * Format detection:
 *  Piper outputs raw PCM (little-endian int16, 22 050 Hz, mono).
 *  gTTS outputs MP3.  We probe the first two bytes: 0xFF 0xFB / 0xFF 0xF3
 *  indicates MP3; otherwise treat as raw PCM.
 */

const PIPER_SAMPLE_RATE = 22_050;
const GTTS_SAMPLE_RATE  = 24_000; // gTTS default
const PCM_CHANNELS      = 1;

class AudioPlayback {
  constructor() {
    this._ctx = null;
    this._queue = [];        // Pending audio chunks (Uint8Array)
    this._sources = [];      // Active AudioBufferSourceNodes
    this._nextStart = 0;     // Scheduled start time (AudioContext clock)
    this._pumping = false;
    this._stopped = false;

    /** Called when playback finishes the last queued chunk */
    this.onComplete = null;
    /** Called when playback starts */
    this.onStart = null;
  }

  // ── Public API ─────────────────────────────────────────────────────

  /** Push a raw audio chunk from the WebSocket binary frame payload. */
  push(uint8Array) {
    if (this._stopped) return;
    this._queue.push(uint8Array);
    if (!this._pumping) {
      this._pump();
    }
  }

  /** Immediately cancel all pending and playing audio (used for interruptions). */
  stop() {
    this._stopped = true;
    this._queue = [];
    this._pumping = false;

    for (const src of this._sources) {
      try { src.stop(); } catch (_) {}
    }
    this._sources = [];

    if (this._ctx) {
      this._nextStart = this._ctx.currentTime;
    }
  }

  /** Resume after a stop (ready for next turn). */
  reset() {
    this._stopped = false;
    this._queue = [];
  }

  // ── Internal pump ──────────────────────────────────────────────────

  async _pump() {
    if (this._pumping || this._stopped) return;
    this._pumping = true;

    this._ensureContext();
    if (this._nextStart < this._ctx.currentTime) {
      this._nextStart = this._ctx.currentTime;
    }

    while (this._queue.length > 0 && !this._stopped) {
      const chunk = this._queue.shift();
      try {
        const buffer = await this._decode(chunk);
        if (this._stopped) break;

        const source = this._ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(this._ctx.destination);

        // Clean up finished sources
        source.onended = () => {
          this._sources = this._sources.filter((s) => s !== source);
          if (this._sources.length === 0 && this._queue.length === 0 && !this._stopped) {
            if (this.onComplete) this.onComplete();
          }
        };

        source.start(this._nextStart);
        this._nextStart += buffer.duration;
        this._sources.push(source);

        if (this._sources.length === 1 && this.onStart) {
          this.onStart();
        }
      } catch (err) {
        console.warn("[AudioPlayback] decode error:", err);
      }
    }

    this._pumping = false;
  }

  async _decode(uint8Array) {
    this._ensureContext();
    const isMp3 = uint8Array[0] === 0xff && (uint8Array[1] & 0xe0) === 0xe0;

    if (isMp3) {
      // Use native browser MP3 decoder
      const arrayBuf = uint8Array.buffer.slice(
        uint8Array.byteOffset,
        uint8Array.byteOffset + uint8Array.byteLength
      );
      return await this._ctx.decodeAudioData(arrayBuf);
    } else {
      // Treat as raw int16 LE PCM from Piper
      return this._decodePcm(uint8Array);
    }
  }

  _decodePcm(uint8Array) {
    // uint8Array: raw int16 LE mono PCM at PIPER_SAMPLE_RATE
    const samples = uint8Array.byteLength / 2;
    const buffer = this._ctx.createBuffer(
      PCM_CHANNELS,
      samples,
      PIPER_SAMPLE_RATE
    );
    const channelData = buffer.getChannelData(0);
    const view = new DataView(
      uint8Array.buffer,
      uint8Array.byteOffset,
      uint8Array.byteLength
    );
    for (let i = 0; i < samples; i++) {
      channelData[i] = view.getInt16(i * 2, true) / 32_768;
    }
    return buffer;
  }

  _ensureContext() {
    if (!this._ctx) {
      this._ctx = new AudioContext({ latencyHint: "playback" });
      this._nextStart = this._ctx.currentTime;
    }
  }
}
