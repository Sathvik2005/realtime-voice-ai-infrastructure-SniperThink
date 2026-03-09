/**
 * audio_processor_worklet.js
 *
 * AudioWorkletProcessor — runs in the dedicated audio thread.
 *
 * Accumulates input samples into 20 ms frames (960 samples @ 48 kHz),
 * converts float32 → int16, then posts binary frames to the main thread.
 * The main thread feeds these frames to the WebRTC peer connection (or
 * directly to the WebSocket fallback).
 *
 * A secondary output: the RMS level of each frame is posted back so the
 * UI can animate the microphone level meter.
 */

const FRAME_SAMPLES = 960; // 20 ms @ 48 kHz

class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(FRAME_SAMPLES);
    this._pos = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) {
      this._buf[this._pos++] = channel[i];

      if (this._pos === FRAME_SAMPLES) {
        // Convert float32 PCM → int16
        const pcm16 = new Int16Array(FRAME_SAMPLES);
        let sumSq = 0;
        for (let j = 0; j < FRAME_SAMPLES; j++) {
          const s = Math.max(-1, Math.min(1, this._buf[j]));
          pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
          sumSq += this._buf[j] * this._buf[j];
        }

        const rms = Math.sqrt(sumSq / FRAME_SAMPLES);

        // Transfer ownership of the ArrayBuffer (zero-copy)
        this.port.postMessage(
          { type: "frame", pcm: pcm16.buffer, rms },
          [pcm16.buffer]
        );

        this._pos = 0;
      }
    }

    return true; // Keep processor alive
  }
}

registerProcessor("voice-processor", VoiceProcessor);
