/**
 * AI Eyes on the Road — Dashboard Controller
 * Handles SocketIO events, UI state, audio alerts, and API calls.
 */

// SocketIO Connection───
const socket = io({ transports: ['websocket'], reconnection: true });

// State
const state = {
  mode: 'camera',          // 'camera' | 'upload'
  cameraActive: false,
  uploadProcessing: false,
  pipelineReady: false,
  totalPotholes: 0,
  // Snapshot of DB total when a camera session starts — used to compute session count
  dbTotalAtStart: null,
  alertActive: false,
  alertTimeout: null,
  audioCtx: null,
  lastAlertTime: 0,
};

// DOM Elements─────────
const el = {
  videoFeed:         document.getElementById('videoFeed'),
  videoPlaceholder:  document.getElementById('videoPlaceholder'),
  videoContainer:    document.getElementById('videoContainer'),
  uploadZone:        document.getElementById('uploadZone'),
  uploadInput:       document.getElementById('uploadInput'),
  uploadPanel:       document.getElementById('uploadPanel'),

  startCameraBtn:    document.getElementById('startCameraBtn'),
  stopCameraBtn:     document.getElementById('stopCameraBtn'),
  uploadBtn:         document.getElementById('uploadBtn'),
  stopUploadBtn:     document.getElementById('stopUploadBtn'),

  modeCameraTab:     document.getElementById('modeCameraTab'),
  modeUploadTab:     document.getElementById('modeUploadTab'),

  // Metrics
  metFPS:            document.getElementById('metFPS'),
  metPotholes:       document.getElementById('metPotholes'),
  metLatency:        document.getElementById('metLatency'),
  metSession:        document.getElementById('metSession'),

  // Alert
  alertBanner:       document.getElementById('alertBanner'),
  alertMessage:      document.getElementById('alertMessage'),

  // Status
  statusDot:         document.getElementById('statusDot'),
  statusText:        document.getElementById('statusText'),

  // Progress
  progressContainer: document.getElementById('progressContainer'),
  progressFill:      document.getElementById('progressFill'),
  progressLabel:     document.getElementById('progressLabel'),
  progressFrames:    document.getElementById('progressFrames'),

  // Right panel tabs
  tabBtns:           document.querySelectorAll('.right-tab'),
  tabPanels:         document.querySelectorAll('.tab-panel'),

  // Pothole list
  potholeList:       document.getElementById('potholeList'),

  // Alert log
  alertLog:          document.getElementById('alertLog'),

  // Download link
  downloadLink:      document.getElementById('downloadLink'),

  // Toast container
  toastContainer:    document.getElementById('toastContainer'),
};

// Initialisation────────

document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupModes();
  setupUploadZone();
  setupConfidenceSlider();
  loadPotholeList();

  // Request pipeline warm-up
  socket.emit('init_pipeline');

  setInterval(loadPotholeList, 10_000);
});

// SocketIO Handlers─────

socket.on('connect', () => {
  setStatus('connected', 'Connected');
  toast('Connected to server', 'success');
});

socket.on('disconnect', () => {
  setStatus('disconnected', 'Disconnected');
  state.cameraActive = false;
  updateCameraButtons();
  window.onCameraSessionStop?.();   // clear stale real-time metrics
});

socket.on('pipeline_ready', (data) => {
  if (data.status === 'ready') {
    state.pipelineReady = true;
    toast('AI Pipeline ready', 'success');
  } else {
    toast('Pipeline error: ' + data.message, 'danger');
  }
});

// Live camera frame — update only real-time metrics here
// metPotholes (DB total) is updated exclusively by loadPotholeList() to avoid flicker
socket.on('frame', (data) => {
  if (!data.image) return;
  if (!state.cameraActive) return;

  el.videoFeed.src = 'data:image/jpeg;base64,' + data.image;
  el.videoFeed.style.display = 'block';
  el.videoPlaceholder.style.display = 'none';

  el.metFPS.textContent     = data.fps         != null ? String(data.fps)           : '—';
  el.metLatency.textContent = data.processing_ms != null ? data.processing_ms + 'ms' : '—';

  if (data.alert && data.alert_message) {
    triggerAlert(data.alert_message, 'near');
  }
});

// Upload frame preview — metPotholes is NOT updated here; DB total comes from loadPotholeList
socket.on('upload_frame', (data) => {
  if (data.image) {
    el.videoFeed.src = 'data:image/jpeg;base64,' + data.image;
    el.videoFeed.style.display = 'block';
    el.videoPlaceholder.style.display = 'none';
  }
  if (data.progress !== undefined) {
    updateProgress(data.progress, data.frame_number, data.total_frames);
  }
  el.metFPS.textContent = data.fps != null ? String(data.fps) : '—';
});

socket.on('upload_complete', (data) => {
  state.uploadProcessing = false;
  updateUploadButtons();
  updateProgress(100);
  toast('Video processing complete!', 'success');

  if (data.download_url && el.downloadLink) {
    el.downloadLink.href = data.download_url;
    el.downloadLink.style.display = 'inline-flex';
  }
  loadPotholeList();
});

socket.on('upload_error', (data) => {
  state.uploadProcessing = false;
  updateUploadButtons();
  toast('Upload error: ' + data.message, 'danger');
});

socket.on('camera_error', (data) => {
  state.cameraActive = false;
  updateCameraButtons();
  toast('Camera error: ' + data.message, 'danger');
});

socket.on('alert', (data) => {
  triggerAlert(data.message, data.risk_level);
  logAlert(data);
});

socket.on('status', (data) => {
  // Only overwrite state for fields the server actually included.
  // browser_camera_active status events omit camera_active, so assigning
  // data.camera_active (undefined) would make the frame guard drop all frames.
  if ('camera_active'      in data) state.cameraActive      = data.camera_active;
  if ('upload_processing'  in data) state.uploadProcessing  = data.upload_processing;
  updateCameraButtons();
  updateUploadButtons();
});

socket.on('map_data', (data) => {
  if (window.mapController) {
    window.mapController.updateMarkers(data.potholes || []);
  }
});

// Immediately refresh map markers AND detections list when the backend confirms a new pothole.
socket.on('new_pothole', () => {
  if (window.mapController) {
    window.mapController._fetchAndUpdate();
  }
  loadPotholeList();
});

// Mode Switching────────

function setupModes() {
  el.modeCameraTab.addEventListener('click', () => switchMode('camera'));
  el.modeUploadTab.addEventListener('click', () => switchMode('upload'));
}

function switchMode(mode) {
  state.mode = mode;

  el.modeCameraTab.classList.toggle('active', mode === 'camera');
  el.modeUploadTab.classList.toggle('active', mode === 'upload');

  const cameraControls = document.getElementById('cameraControls');
  const uploadControls = document.getElementById('uploadControls');

  if (mode === 'camera') {
    cameraControls?.classList.remove('hidden');
    uploadControls?.classList.add('hidden');
    el.uploadPanel.classList.add('hidden');
    el.videoContainer.style.display = 'flex';
  } else {
    cameraControls?.classList.add('hidden');
    uploadControls?.classList.remove('hidden');
    el.uploadPanel.classList.remove('hidden');
    if (!state.uploadProcessing) {
      el.videoContainer.style.display = 'none';
    }
  }
}

// Camera Controls───────
// Camera start/stop is handled by the inline script in index.html (which manages
// the browser-vs-server source selection and RTSP URL). That script uses
// stopImmediatePropagation() so these handlers are intentionally absent here.
// State sync happens via the SocketIO 'status' event below.

function updateCameraButtons() {
  if (!el.startCameraBtn) return;
  el.startCameraBtn.disabled = state.cameraActive;
  el.stopCameraBtn.disabled  = false;   // Reset is always available
}

// Upload

function setupUploadZone() {
  el.uploadZone?.addEventListener('click', () => el.uploadInput?.click());

  el.uploadZone?.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.uploadZone.classList.add('drag-over');
  });

  el.uploadZone?.addEventListener('dragleave', () => {
    el.uploadZone.classList.remove('drag-over');
  });

  el.uploadZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    el.uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) uploadFile(file);
  });

  el.uploadInput?.addEventListener('change', () => {
    const file = el.uploadInput.files[0];
    if (file) uploadFile(file);
  });

  el.uploadBtn?.addEventListener('click', () => el.uploadInput?.click());

  el.stopUploadBtn?.addEventListener('click', () => {
    // Tell server to cancel the active job, then clean up client state
    socket.emit('stop_upload');
    if (state._pollTimer) { clearInterval(state._pollTimer); state._pollTimer = null; }
    state.uploadProcessing = false;
    updateUploadButtons();
    toast('Upload processing stopped.', 'warning');
  });
}

function uploadFile(file) {
  const allowed = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    toast(`Unsupported file type: .${ext}`, 'danger');
    return;
  }

  const formData = new FormData();
  formData.append('video', file);

  state.uploadProcessing = true;
  state._pollTimer = null;
  updateUploadButtons();
  setStatus('active', 'Processing Video');
  el.uploadPanel.classList.add('hidden');
  el.videoContainer.style.display = 'flex';
  el.progressContainer.classList.add('visible');
  if (el.downloadLink) el.downloadLink.style.display = 'none';
  updateProgress(0);

  toast(`Uploading ${file.name}…`);

  fetch('/upload', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        toast(data.error, 'danger');
        state.uploadProcessing = false;
        updateUploadButtons();
        return;
      }
      // Backend now returns {status:'queued', job_id:'abc...'} — poll for progress
      if (data.job_id) {
        toast(`Video queued (job ${data.job_id.slice(0,6)}…) — processing…`);
        _pollJobProgress(data.job_id);
      }
    })
    .catch(err => {
      toast('Upload failed: ' + err.message, 'danger');
      state.uploadProcessing = false;
      updateUploadButtons();
    });
}

function _pollJobProgress(jobId) {
  state._pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/jobs/${jobId}/progress`);
      const data = await resp.json();

      const pct = data.progress || 0;
      updateProgress(pct);

      if ((data.status || '').startsWith('error')) {
        clearInterval(state._pollTimer);
        state.uploadProcessing = false;
        updateUploadButtons();
        toast('Processing error: ' + data.status, 'danger');
        return;
      }

      if (data.status === 'done') {
        clearInterval(state._pollTimer);
        state.uploadProcessing = false;
        updateUploadButtons();
        updateProgress(100);
        toast('Video processing complete!', 'success');
        if (data.output && el.downloadLink) {
          el.downloadLink.href = `/outputs/${data.output}`;
          el.downloadLink.style.display = 'inline-flex';
        }
        loadPotholeList();
      }
    } catch (e) {
      // network blip — keep polling
    }
  }, 1500);
}

function updateUploadButtons() {
  if (!el.uploadBtn) return;
  el.uploadBtn.disabled = state.uploadProcessing;
  el.stopUploadBtn.disabled = !state.uploadProcessing;
}

function updateProgress(pct, frame, total) {
  el.progressFill.style.width = pct + '%';
  el.progressLabel.textContent = `${Math.round(pct)}%`;
  if (frame && total) {
    el.progressFrames.textContent = `Frame ${frame} / ${total}`;
  }
}

// Alert System───────────

function triggerAlert(message, riskLevel) {
  const now = Date.now();
  if (now - state.lastAlertTime < 3000) return;  // client-side cooldown
  state.lastAlertTime = now;

  el.alertBanner.classList.add('visible');
  el.alertMessage.textContent = message;
  el.statusDot?.classList.add('danger');

  if (riskLevel === 'near') {
    playAlertSound(880, 0.6);
  } else if (riskLevel === 'medium') {
    playAlertSound(440, 0.3);
  }

  clearTimeout(state.alertTimeout);
  state.alertTimeout = setTimeout(() => {
    el.alertBanner.classList.remove('visible');
    el.statusDot?.classList.remove('danger');
  }, 4000);
}

function logAlert(data) {
  const item = document.createElement('div');
  item.className = `alert-log-item ${data.risk_level || 'far'}`;
  item.innerHTML = `
    <strong>${data.message}</strong>
    <div class="alert-time">${new Date(data.timestamp * 1000).toLocaleTimeString()}</div>
  `;
  el.alertLog?.prepend(item);

  // Keep only 50 entries
  while (el.alertLog?.children.length > 50) {
    el.alertLog.lastChild?.remove();
  }
}

// Web Audio API Alert Sound─

function playAlertSound(frequency = 880, volume = 0.5) {
  try {
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const ctx = state.audioCtx;

    // Double beep
    [0, 0.25].forEach((delay) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'square';
      osc.frequency.setValueAtTime(frequency, ctx.currentTime + delay);
      gain.gain.setValueAtTime(volume, ctx.currentTime + delay);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + 0.18);
      osc.start(ctx.currentTime + delay);
      osc.stop(ctx.currentTime + delay + 0.18);
    });
  } catch (e) {
    // Audio not available
  }
}

// Tab System────────────

function setupTabs() {
  el.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      el.tabBtns.forEach(b => b.classList.remove('active'));
      el.tabPanels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(target)?.classList.add('active');

      if (target === 'mapPanel') {
        // Request map data update
        socket.emit('get_map_data');
      }
    });
  });
}

// Pothole List───────────

async function loadPotholeList() {
  try {
    const res = await fetch('/api/potholes?limit=100');
    const data = await res.json();
    renderPotholeList(data.potholes || []);
    state.totalPotholes = data.total || 0;

    // DB total — the single source of truth for this metric
    el.metPotholes.textContent = state.totalPotholes;

    // Session count: potholes added to DB since this camera session started
    if (state.dbTotalAtStart !== null) {
      el.metSession.textContent = Math.max(0, state.totalPotholes - state.dbTotalAtStart);
    }
  } catch (e) { /* silent */ }
}

function renderPotholeList(potholes) {
  if (!el.potholeList) return;

  if (!potholes.length) {
    el.potholeList.innerHTML = '<div class="text-muted" style="padding:20px;text-align:center">No potholes detected yet</div>';
    return;
  }

  el.potholeList.innerHTML = potholes.map(ph => {
    const dist = ph.estimated_distance_m ? `${ph.estimated_distance_m.toFixed(1)}m` : 'N/A';
    const conf = (ph.confidence * 100).toFixed(0);
    const snap = ph.snapshot_path ? `<img class="pothole-snapshot" src="/static/${ph.snapshot_path}" alt="snapshot">` : '';
    const lat  = ph.latitude  ? ph.latitude.toFixed(5) : '—';
    const lon  = ph.longitude ? ph.longitude.toFixed(5) : '—';
    const risk = ph.risk_level || 'far';

    return `
      <div class="pothole-card" data-id="${ph.id}">
        <div class="risk-badge ${risk}"></div>
        <div class="pothole-info">
          <div class="ph-id">ID #${ph.id} · Track #${ph.track_id ?? '—'}</div>
          <div class="ph-conf">Conf: ${conf}% · Dist: ${dist} · Risk: ${risk.toUpperCase()}</div>
          <div class="ph-meta">GPS: ${lat}, ${lon} · Seen: ${ph.detection_count}× · ${formatTime(ph.last_seen)}</div>
        </div>
        ${snap}
        <div class="pothole-actions">
          <button class="btn btn-sm" onclick="reportPotholeEmail(${ph.id})">Report</button>
          <button class="btn btn-sm"
                  onclick="pinToMap(${ph.id}, ${ph.latitude ?? 'null'}, ${ph.longitude ?? 'null'})"
                  title="${ph.latitude == null ? 'No GPS stored — will use current location' : 'Show on map'}">📍 Map</button>
          <button class="btn btn-sm btn-danger" onclick="deletePothole(${ph.id})">✕</button>
        </div>
      </div>
    `;
  }).join('');
}

// loadStats() removed — metSession is now computed from dbTotalAtStart in loadPotholeList()
// Report panel stats are refreshed by loadReportStats() in index.html inline script

// Map Pin Helpers────────

function switchToMapPanel() {
  // Simulate clicking the Map tab so setupTabs() logic fires (active classes + get_map_data)
  const mapBtn = document.querySelector('.right-tab[data-tab="mapPanel"]');
  if (mapBtn) mapBtn.click();
}

async function pinToMap(id, lat, lng) {
  // If no stored coordinates, try to use the server's current GPS
  if (lat == null || lng == null) {
    try {
      const r = await fetch('/api/location');
      const g = await r.json();
      if (g.lat && g.lng) {
        // Assign current GPS to this pothole so it appears on the map
        await fetch(`/api/potholes/${id}/location`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat: g.lat, lng: g.lng }),
        });
        lat = g.lat;
        lng = g.lng;
        toast('GPS assigned from current location', 'success');
        loadPotholeList();
      } else {
        toast('No GPS available — enable GPS and start camera first', 'warning');
        return;
      }
    } catch (e) {
      toast('No GPS coordinates available', 'warning');
      return;
    }
  }
  switchToMapPanel();
  setTimeout(() => window.mapController?.focusPothole(id, lat, lng), 150);
}
window.pinToMap = pinToMap;

// Report

async function reportPothole(id) {
  toast('Sending MDOT report…');
  try {
    const res = await fetch(`/api/potholes/${id}/report`, { method: 'POST' });
    const data = await res.json();
    if (data.status === 'sent') {
      toast(`Report sent! Ref: ${data.reference_number}`, 'success');
    } else if (data.status === 'dry_run') {
      toast('Dry run (SMTP not configured). Report logged.', 'warning');
    } else {
      toast('Report failed: ' + (data.error || 'unknown'), 'danger');
    }
  } catch (e) {
    toast('Report error: ' + e.message, 'danger');
  }
}

async function deletePothole(id) {
  try {
    await fetch(`/api/potholes/${id}`, { method: 'DELETE' });
    toast('Pothole deleted', 'success');
    loadPotholeList();
    window.mapController?._fetchAndUpdate();
  } catch (e) {
    toast('Delete failed', 'danger');
  }
}

// GPS is managed by window.gpsManager (GPSManager class in camera.js).
// It is started/stopped by the camera start/stop handlers in index.html inline script
// so GPS only runs during live camera sessions, never during video uploads.

// Status Indicator───────

function setStatus(type, text) {
  if (!el.statusDot) return;
  el.statusDot.className = 'status-dot';
  if (type === 'active')       el.statusDot.classList.add('active');
  else if (type === 'danger')  el.statusDot.classList.add('danger');
  if (el.statusText) el.statusText.textContent = text;
}

// Toast Notifications────

function toast(message, type = '') {
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = message;
  el.toastContainer?.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// Utilities 

function formatTime(isoStr) {
  if (!isoStr) return '—';
  try {
    return new Date(isoStr).toLocaleTimeString();
  } catch { return isoStr; }
}

// Confidence Threshold Slider

function setupConfidenceSlider() {
  const slider  = document.getElementById('confSlider');
  const display = document.getElementById('confValue');
  if (!slider) return;

  slider.addEventListener('input', () => {
    const pct = parseInt(slider.value, 10);
    if (display) display.textContent = pct + '%';
    socket.emit('set_confidence', { threshold: pct / 100 });
  });
}

// Camera session helpers (called by index.html inline script)───────

/** Called when a camera session starts — snapshots DB total for session counter. */
function onCameraSessionStart() {
  state.dbTotalAtStart = state.totalPotholes;
  el.metSession.textContent = '0';
}

/** Called when a camera session ends — clears real-time metrics. */
function onCameraSessionStop() {
  el.metFPS.textContent     = '—';
  el.metLatency.textContent = '—';
  // Keep metSession showing the session total so the user can see what was found
}

// Expose for inline onclick handlers and inline camera script
window.reportPothole       = reportPothole;
window.deletePothole       = deletePothole;
window.onCameraSessionStart = onCameraSessionStart;
window.onCameraSessionStop  = onCameraSessionStop;
