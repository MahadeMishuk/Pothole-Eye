/**
 * WebRTC WHEP client for AI Eyes on the Road live stream.
 *
 * Connects to MediaMTX via the WHEP (WebRTC-HTTP Egress Protocol) endpoint.
 * Falls back to the SocketIO JPEG stream if WebRTC is unavailable or fails.
 *
 * Usage (from index.html or dashboard.js):
 *   import { WebRTCStream } from '/static/js/webrtc_stream.js';
 *   const stream = new WebRTCStream('#video-element', onFallback);
 *   await stream.start();
 *   stream.stop();
 */

export class WebRTCStream {
  /**
   * @param {string|HTMLVideoElement} videoTarget  CSS selector or <video> element
   * @param {Function} [onFallback]  Called with (reason) when falling back to SocketIO
   */
  constructor(videoTarget, onFallback) {
    this._video = typeof videoTarget === 'string'
      ? document.querySelector(videoTarget)
      : videoTarget;
    this._onFallback = onFallback || null;
    this._pc     = null;
    this._active = false;
  }

  async start() {
    try {
      const info = await this._fetchStreamInfo();
      if (!info.enabled) {
        this._fallback('rtsp_disabled');
        return;
      }
      await this._connect(info.whep_url);
    } catch (err) {
      console.warn('[WebRTC] start failed:', err);
      this._fallback(String(err));
    }
  }

  stop() {
    this._active = false;
    if (this._pc) {
      this._pc.close();
      this._pc = null;
    }
    if (this._video) {
      this._video.srcObject = null;
    }
  }

  get isActive() { return this._active; }

  // Private─

  async _fetchStreamInfo() {
    const resp = await fetch('/api/stream/webrtc');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  }

  async _connect(whepUrl) {
    const pc = new RTCPeerConnection({
      iceServers: [],           // MediaMTX is on same host; STUN not needed
      bundlePolicy: 'max-bundle',
    });
    this._pc = pc;

    // Request video-only track from MediaMTX
    pc.addTransceiver('video', { direction: 'recvonly' });

    pc.ontrack = (evt) => {
      if (evt.streams && evt.streams[0]) {
        this._video.srcObject = evt.streams[0];
        this._video.play().catch(() => {});
        this._active = true;
        console.info('[WebRTC] stream attached');
      }
    };

    pc.oniceconnectionstatechange = () => {
      const s = pc.iceConnectionState;
      console.debug('[WebRTC] ICE state:', s);
      if (s === 'failed' || s === 'disconnected') {
        this._fallback('ice_' + s);
      }
    };

    // WHEP: create offer, POST to /live/whep, parse answer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Wait for ICE gathering to complete (or 3s timeout)
    await this._waitForIce(pc);

    const resp = await fetch(whepUrl, {
      method:  'POST',
      headers: { 'Content-Type': 'application/sdp' },
      body:    pc.localDescription.sdp,
    });

    if (!resp.ok) {
      throw new Error(`WHEP POST failed: HTTP ${resp.status}`);
    }

    const answerSdp = await resp.text();
    await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
    console.info('[WebRTC] WHEP exchange complete');
  }

  _waitForIce(pc) {
    return new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') { resolve(); return; }
      const timeout = setTimeout(resolve, 3000);
      pc.onicegatheringstatechange = () => {
        if (pc.iceGatheringState === 'complete') {
          clearTimeout(timeout);
          resolve();
        }
      };
    });
  }

  _fallback(reason) {
    console.warn('[WebRTC] falling back to SocketIO —', reason);
    this.stop();
    if (this._onFallback) this._onFallback(reason);
  }
}

// Standalone upload progress poller

/**
 * Poll /api/jobs/<jobId>/progress until status is 'done' or 'error'.
 *
 * @param {string}   jobId
 * @param {Function} onProgress  Called with (pct, status) each tick
 * @param {Function} onDone      Called with (outputFilename) on completion
 * @param {Function} onError     Called with (statusString) on error
 * @param {number}   [intervalMs=1500]
 */
export function pollJobProgress(jobId, onProgress, onDone, onError, intervalMs = 1500) {
  const timer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/jobs/${jobId}/progress`);
      const data = await resp.json();

      onProgress(data.progress || 0, data.status || 'processing');

      if (data.status === 'done') {
        clearInterval(timer);
        onDone(data.output || null);
      } else if ((data.status || '').startsWith('error')) {
        clearInterval(timer);
        onError(data.status);
      }
    } catch (err) {
      console.warn('[pollJobProgress] fetch error:', err);
    }
  }, intervalMs);

  return { cancel: () => clearInterval(timer) };
}
