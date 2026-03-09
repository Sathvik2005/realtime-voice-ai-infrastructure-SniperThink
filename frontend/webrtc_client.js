/**
 * webrtc_client.js
 *
 * Manages one RTCPeerConnection per voice session.
 *
 * Flow:
 *  1. Connect to the backend signaling WebSocket.
 *  2. Add the local microphone MediaStream track.
 *  3. Create SDP offer → send via WebSocket.
 *  4. Receive SDP answer → setRemoteDescription.
 *  5. Exchange ICE candidates bidirectionally.
 *
 * The WebSocket also carries:
 *  • Control events     (JSON text frames)   → onEvent(msg)
 *  • TTS audio chunks   (binary frames)      → onAudioChunk(Uint8Array)
 *
 * Re-connection / fallback:
 *  If WebRTC negotiation fails the caller can still transmit audio by
 *  posting raw PCM binary frames over the same WebSocket (direct path).
 */

class WebRTCClient {
  /**
   * @param {string} wsUrl  - Full WebSocket URL, e.g. "ws://localhost:8000/ws/xyz"
   */
  constructor(wsUrl) {
    this._wsUrl = wsUrl;
    this._ws = null;
    this._pc = null;

    /** @type {(msg: object) => void} JSON control event handler */
    this.onEvent = null;
    /** @type {(chunk: Uint8Array) => void} TTS audio chunk handler */
    this.onAudioChunk = null;
    /** @type {() => void} Called when WebSocket opens */
    this.onConnected = null;
    /** @type {() => void} Called when WebSocket closes */
    this.onDisconnected = null;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────

  /**
   * Open the WebSocket and, once the server sends "ready",
   * begin WebRTC negotiation using the provided MediaStream.
   *
   * @param {MediaStream} stream  - Local microphone stream
   */
  async connect(stream) {
    this._ws = new WebSocket(this._wsUrl);
    this._ws.binaryType = "arraybuffer";

    this._ws.onopen = () => {
      console.debug("[WebRTCClient] WebSocket open");
    };

    this._ws.onclose = () => {
      console.debug("[WebRTCClient] WebSocket closed");
      if (this.onDisconnected) this.onDisconnected();
    };

    this._ws.onerror = (err) => {
      console.error("[WebRTCClient] WebSocket error:", err);
    };

    this._ws.onmessage = async (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        // Binary frame = TTS audio chunk
        if (this.onAudioChunk) {
          const bytes = new Uint8Array(evt.data);
          // Strip the 1-byte type header (0x01 = audio)
          if (bytes[0] === 0x01) {
            this.onAudioChunk(bytes.slice(1));
          }
        }
        return;
      }

      // JSON control/signaling message
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }

      await this._handleSignalingMessage(msg, stream);
    };

    // Wait for "ready" signal from the server
    await this._waitForReady();
    if (this.onConnected) this.onConnected();

    // Start WebRTC negotiation
    await this._negotiate(stream);
  }

  /** Send a raw JSON control message to the backend. */
  sendControl(msg) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(msg));
    }
  }

  /** Send raw PCM int16 bytes over WebSocket (non-WebRTC fallback). */
  sendAudioFallback(pcm16) {
    if (this._ws?.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: "audio_data", data: Array.from(pcm16) }));
    }
  }

  disconnect() {
    if (this._pc) {
      this._pc.close();
      this._pc = null;
    }
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }

  // ── WebRTC Negotiation ─────────────────────────────────────────────

  async _negotiate(stream) {
    const config = {
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        { urls: "stun:stun1.l.google.com:19302" },
      ],
    };

    this._pc = new RTCPeerConnection(config);

    // Add microphone tracks to the peer connection
    stream.getAudioTracks().forEach((track) => {
      this._pc.addTrack(track, stream);
    });

    // Trickle ICE
    this._pc.onicecandidate = (evt) => {
      if (evt.candidate) {
        this.sendControl({
          type: "ice_candidate",
          candidate: evt.candidate.toJSON(),
        });
      }
    };

    this._pc.onconnectionstatechange = () => {
      console.debug("[WebRTCClient] connection state:", this._pc.connectionState);
    };

    // Create and send offer
    const offer = await this._pc.createOffer({ offerToReceiveAudio: false });
    await this._pc.setLocalDescription(offer);

    this.sendControl({
      type: "offer",
      sdp: offer.sdp,
      type_: offer.type,
    });
  }

  async _handleSignalingMessage(msg, stream) {
    switch (msg.type) {
      case "ready":
        // Already handled by _waitForReady
        break;

      case "answer":
        if (this._pc) {
          await this._pc.setRemoteDescription(
            new RTCSessionDescription({ sdp: msg.sdp, type: msg.type_ })
          );
          console.debug("[WebRTCClient] SDP answer applied");
        }
        break;

      case "ice_candidate":
        if (this._pc && msg.candidate) {
          try {
            await this._pc.addIceCandidate(new RTCIceCandidate(msg.candidate));
          } catch (e) {
            console.warn("[WebRTCClient] ICE candidate error:", e);
          }
        }
        break;

      default:
        // Delegate control events (speech_start, transcript, etc.) to caller
        if (this.onEvent) this.onEvent(msg);
        break;
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────

  _waitForReady() {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("Server ready timeout")), 8000);
      const origOnMessage = this._ws.onmessage;

      this._ws.onmessage = async (evt) => {
        let msg;
        try { msg = JSON.parse(evt.data); } catch { return; }
        if (msg.type === "ready") {
          clearTimeout(timeout);
          // Restore the proper handler
          this._ws.onmessage = origOnMessage;
          resolve();
        }
      };
    });
  }
}
