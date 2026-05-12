
'use strict';

// Configuration────────────
const CAM_CONFIG = {
  IDEAL_WIDTH:    640,
  IDEAL_HEIGHT:   480,
  FRAME_INTERVAL: 120,   // ms between captured frames (~8 FPS to CPU server)
  JPEG_QUALITY:   0.72,  // canvas toBlob quality
  GPS_INTERVAL:   3000,  // ms between GPS pushes to server
  IP_GEO_URL:     'https://ipapi.co/json/',  // IP-based fallback
};


// BrowserCamera────────────

class BrowserCamera {
  /**
   * @param {object} socket   socket.io client instance
   * @param {HTMLVideoElement}  videoEl   preview <video> element
   * @param {HTMLCanvasElement} canvasEl  offscreen <canvas> for frame capture
   */
  constructor(socket, videoEl, canvasEl) {
    this.socket   = socket;
    this.videoEl  = videoEl;
    this.canvasEl = canvasEl;
    this.ctx      = canvasEl.getContext('2d');

    this._stream        = null;
    this._frameTimer    = null;
    this._active        = false;
    this._facingMode    = 'user';   // 'user' = front/laptop cam; 'environment' = rear (mobile only)

    this.onError   = null;   // callback(message)
    this.onStarted = null;   // callback()
    this.onStopped = null;   // callback()
  }

  // Public API─────────────

  get active() { return this._active; }

  async start(facingMode = null) {
    if (this._active) return;
    this._facingMode = facingMode || this._facingMode;

    // Prefer a real built-in camera device over virtual cameras (e.g. OBS).
    // Falls back to facingMode if no suitable deviceId is found.
    const preferredDeviceId = await BrowserCamera._pickBuiltinCameraId();

    const videoConstraints = preferredDeviceId
      ? { deviceId: { exact: preferredDeviceId },
          width: { ideal: CAM_CONFIG.IDEAL_WIDTH },
          height: { ideal: CAM_CONFIG.IDEAL_HEIGHT },
          frameRate: { ideal: 15, max: 30 } }
      : { width: { ideal: CAM_CONFIG.IDEAL_WIDTH },
          height: { ideal: CAM_CONFIG.IDEAL_HEIGHT },
          facingMode: this._facingMode,
          frameRate: { ideal: 15, max: 30 } };

    const constraints = { video: videoConstraints, audio: false };

    try {
      this._stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      const msg = this._friendlyError(err);
      this._fire('onError', msg);
      return;
    }

    // Attach stream to preview element
    this.videoEl.srcObject = this._stream;
    this.videoEl.setAttribute('playsinline', '');  // required on iOS
    await this.videoEl.play().catch(() => {});

    // Sync canvas size to actual video resolution
    this.videoEl.addEventListener('loadedmetadata', () => {
      this.canvasEl.width  = this.videoEl.videoWidth  || CAM_CONFIG.IDEAL_WIDTH;
      this.canvasEl.height = this.videoEl.videoHeight || CAM_CONFIG.IDEAL_HEIGHT;
    }, { once: true });

    this._active = true;
    this.socket.emit('start_browser_camera');
    this._startCapture();
    this._fire('onStarted');
  }

  stop() {
    if (!this._active) return;
    this._active = false;

    clearInterval(this._frameTimer);
    this._frameTimer = null;

    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    this.videoEl.srcObject = null;
    this.socket.emit('stop_browser_camera');
    this._fire('onStopped');
  }

  /** Flip between front/rear camera (mobile). */
  async flipCamera() {
    const wasActive = this._active;
    this.stop();
    this._facingMode = this._facingMode === 'environment' ? 'user' : 'environment';
    if (wasActive) await this.start(this._facingMode);
  }

  /** List available video input devices. */
  static async getDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(d => d.kind === 'videoinput');
  }

  /**
   * Pick the best built-in camera, avoiding virtual cameras (OBS, etc.).
   * Returns a deviceId string, or null to fall back to facingMode.
   */
  static async _pickBuiltinCameraId() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter(d => d.kind === 'videoinput');
      if (cameras.length === 0) return null;

      // Skip known virtual camera labels
      const VIRTUAL = ['obs', 'virtual', 'snap camera', 'mmhmm', 'camo'];
      const real = cameras.filter(d => {
        const label = d.label.toLowerCase();
        return !VIRTUAL.some(v => label.includes(v));
      });

      // Prefer a real camera; fall back to first available
      const chosen = real.length > 0 ? real[0] : cameras[0];
      return chosen.deviceId || null;
    } catch {
      return null;
    }
  }

  // Frame Capture───────────

  _startCapture() {
    this._frameTimer = setInterval(() => {
      if (!this._active || this.videoEl.readyState < 2) return;
      this._captureFrame();
    }, CAM_CONFIG.FRAME_INTERVAL);
  }

  _captureFrame() {
    const w = this.canvasEl.width  || CAM_CONFIG.IDEAL_WIDTH;
    const h = this.canvasEl.height || CAM_CONFIG.IDEAL_HEIGHT;

    // Draw current video frame to canvas
    this.ctx.drawImage(this.videoEl, 0, 0, w, h);

    // Export as JPEG blob → base64 → emit
    this.canvasEl.toBlob(
      (blob) => {
        if (!blob || !this._active) return;
        const reader = new FileReader();
        reader.onloadend = () => {
          const b64 = reader.result.split(',')[1];  // strip data:image/jpeg;base64,
          this.socket.emit('browser_frame', { image: b64, width: w, height: h });
        };
        reader.readAsDataURL(blob);
      },
      'image/jpeg',
      CAM_CONFIG.JPEG_QUALITY
    );
  }

  // Helpers─

  _friendlyError(err) {
    const map = {
      NotAllowedError:    'Camera permission denied. Please allow camera access in your browser.',
      NotFoundError:      'No camera found on this device.',
      NotReadableError:   'Camera is already in use by another application.',
      OverconstrainedError: 'Camera does not support the requested resolution.',
      SecurityError:      'Camera blocked by browser security policy (HTTPS required on some browsers).',
    };
    return map[err.name] || `Camera error: ${err.message}`;
  }

  _fire(cb, ...args) {
    if (typeof this[cb] === 'function') this[cb](...args);
  }
}


// GPS Manager 

class GPSManager {
  /**
   * Manages GPS acquisition with three-tier fallback:
   *   1. Browser Geolocation API (high accuracy, requires permission)
   *   2. IP-based geolocation (low accuracy, no permission needed)
   *   3. No location (map centres on server default)
   */
  constructor() {
    this._watchId     = null;
    this._ipFallback  = false;
    this._lastUpdate  = 0;
    this.currentPos   = null;   // { lat, lng, accuracy, source }
    this.onUpdate     = null;   // callback(pos)
    this.onError      = null;   // callback(message)
  }

  start() {
    if (!navigator.geolocation) {
      this._tryIPFallback();
      return;
    }

    this._watchId = navigator.geolocation.watchPosition(
      (pos) => this._handleBrowserGPS(pos),
      (err) => {
        const msgs = {
          1: 'GPS permission denied.',
          2: 'GPS position unavailable.',
          3: 'GPS request timed out.',
        };
        this._fire('onError', msgs[err.code] || 'GPS error.');
        if (err.code === 1) this._tryIPFallback();  // permission denied → IP fallback
      },
      { enableHighAccuracy: true, maximumAge: 2000, timeout: 10000 }
    );
  }

  stop() {
    if (this._watchId !== null) {
      navigator.geolocation.clearWatch(this._watchId);
      this._watchId = null;
    }
  }

  // Browser GPS──────────

  _handleBrowserGPS(pos) {
    const now = Date.now();
    if (now - this._lastUpdate < CAM_CONFIG.GPS_INTERVAL) return;
    this._lastUpdate = now;

    const location = {
      lat:      pos.coords.latitude,
      lng:      pos.coords.longitude,
      accuracy: pos.coords.accuracy,
      source:   'browser',
    };
    this.currentPos = location;
    this._pushToServer(location);
    this._fire('onUpdate', location);
  }

  // IP Fallback──────────

  async _tryIPFallback() {
    if (this._ipFallback) return;
    this._ipFallback = true;

    try {
      const res  = await fetch(CAM_CONFIG.IP_GEO_URL);
      const data = await res.json();
      if (!data.latitude || !data.longitude) throw new Error('No coordinates');

      const location = {
        lat:      data.latitude,
        lng:      data.longitude,
        accuracy: null,
        source:   'ip',
        city:     data.city,
        country:  data.country_name,
      };
      this.currentPos = location;
      this._pushToServer(location);
      this._fire('onUpdate', location);
    } catch (e) {
      this._fire('onError', 'IP geolocation unavailable — location features disabled.');
    }
  }

  // Push to server───────

  _pushToServer(location) {
    fetch('/api/location', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        lat:    location.lat,
        lng:    location.lng,
        source: location.source,
      }),
    }).catch(() => {});
  }

  _fire(cb, ...args) {
    if (typeof this[cb] === 'function') this[cb](...args);
  }
}


// Module exports────────────

window.BrowserCamera = BrowserCamera;
window.GPSManager    = GPSManager;
