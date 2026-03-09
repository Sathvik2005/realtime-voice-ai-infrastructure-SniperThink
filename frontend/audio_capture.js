/**
 * audio_capture.js
 *
 * Microphone capture via MediaStream + AudioWorklet.
 *
 * Responsibilities:
 *  • Request microphone permission
 *  • Build an AudioContext → MediaStreamSource → AudioWorkletNode chain
 *  • Emit 20 ms int16 PCM frames via onFrame(Int16Array, rms)
 *  • Expose RMS level so the UI can animate the meter
 *
 * The caller (app.js) decides what to do with each frame —
 * send via WebRTC DataChannel or WebSocket fallback.
 */

class AudioCapture {
  constructor() {
    this._ctx = null;
    this._workletNode = null;
    this._stream = null;
    this._running = false;

    /** @type {(pcm: Int16Array, rms: number) => void} */
    this.onFrame = null;
  }

  // ── Public API ─────────────────────────────────────────────────────

  /**
   * Start capturing microphone audio.
   * Registers the AudioWorklet processor and begins emitting frames.
   */
  async start() {
    if (this._running) return;

    // Request mic access — no video, echo cancellation on
    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 48000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    this._ctx = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });

    // Load the worklet processor from the backend's static route
    await this._ctx.audioWorklet.addModule("/static/audio_processor_worklet.js");

    const source = this._ctx.createMediaStreamSource(this._stream);
    this._workletNode = new AudioWorkletNode(this._ctx, "voice-processor", {
      processorOptions: {},
    });

    // Receive frames from the audio thread
    this._workletNode.port.onmessage = (evt) => {
      if (evt.data.type === "frame" && this.onFrame) {
        const pcm = new Int16Array(evt.data.pcm);
        this.onFrame(pcm, evt.data.rms);
      }
    };

    source.connect(this._workletNode);
    // Do NOT connect worklet to destination — we don't want local playback

    this._running = true;
  }

  /**
   * Stop capturing and release hardware resources.
   */
  stop() {
    if (!this._running) return;
    this._running = false;

    if (this._workletNode) {
      this._workletNode.disconnect();
      this._workletNode = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
    }
    if (this._ctx) {
      this._ctx.close();
      this._ctx = null;
    }
  }

  /** Expose the active MediaStream for WebRTC track registration. */
  get stream() {
    return this._stream;
  }

  get isRunning() {
    return this._running;
  }
}
