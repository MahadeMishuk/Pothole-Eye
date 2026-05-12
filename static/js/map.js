/**
 * AI Eyes on the Road — Mapbox GL JS Map Controller
 * ===================================================
 * Real-time geospatial intelligence layer for pothole visualization.
 *
 * Architecture:
 *   - Single GeoJSON source ("potholes-source") powers all layers
 *   - Layers: clusters → cluster-count → unclustered stars → heatmap
 *   - Polling loop updates the source data every POLL_INTERVAL ms
 *   - Vehicle position tracked via Geolocation API
 *   - Path/route drawn as a LineString layer
 *   - Deduplication handled server-side (spatial clustering in DB)
 *     + client-side guard using a Set of known IDs
 *
 * Token security:
 *   mapboxgl.accessToken is set from a Flask template variable (server-injected),
 *   NOT hardcoded in this file. The /api/mapbox-token endpoint provides the same
 *   value for any runtime bootstrap that can't use template rendering.
 */

'use strict';

// Constants ─
const POLL_INTERVAL_MS  = 2000;    // how often to refresh pothole markers
const PATH_MAX_POINTS   = 500;     // cap on route history points
const CLUSTER_RADIUS    = 50;      // pixels within which markers cluster
const CLUSTER_MAX_ZOOM  = 14;      // stop clustering above this zoom
const SOURCE_ID         = 'potholes-source';
const HEATMAP_SOURCE_ID = 'heatmap-source';

// Risk → colour mapping (matches dashboard.css variables)
const RISK_COLOURS = {
  near:    '#f85149',
  medium:  '#e3a11c',
  far:     '#3fb950',
  unknown: '#58a6ff',
};


// PotholeMapController────

class PotholeMapController {
  /**
   * @param {string}  containerId  DOM element id for the map canvas
   * @param {object}  opts
   * @param {string}  opts.token       Mapbox access token
   * @param {string}  opts.style       Mapbox style URL
   * @param {number}  opts.defaultLat
   * @param {number}  opts.defaultLng
   * @param {number}  opts.defaultZoom
   */
  constructor(containerId, opts = {}) {
    this.containerId = containerId;
    this.token       = opts.token       || '';
    this.style       = opts.style       || 'mapbox://styles/mapbox/dark-v11';
    this.defaultLat  = opts.defaultLat  ?? 39.2904;
    this.defaultLng  = opts.defaultLng  ?? -76.6122;
    this.defaultZoom = opts.defaultZoom ?? 13;

    this.map             = null;
    this.popup           = null;
    this.vehicleMarker   = null;
    this.pathCoords      = [];          // [[lng, lat], ...]
    this._pollTimer      = null;
    this._knownIds       = new Set();   // client-side duplicate guard
    this._heatmapVisible = false;
    this._ready          = false;
    this._isDark         = true;        // current style: dark=true, light=false
    this._is3D           = false;       // 3D pitch mode

    // Active route state — used for auto re-routing on deviation
    this._routeCoords    = null;   // [[lng,lat],...] of the current route
    this._routeDestC     = null;   // [lng, lat] of the destination
    this._reRouting      = false;  // debounce guard
  }

  // Bootstrap────────────

  async init() {
    if (!this.token) {
      console.error('[MapController] No Mapbox token provided.');
      this._showNoTokenPlaceholder();
      return;
    }

    mapboxgl.accessToken = this.token;

    this.map = new mapboxgl.Map({
      container:   this.containerId,
      style:       this.style,
      center:      [this.defaultLng, this.defaultLat],
      zoom:        this.defaultZoom,
      attributionControl: true,
      antialias:   true,
    });

    // Navigation controls
    this.map.addControl(new mapboxgl.NavigationControl(), 'top-right');
    this.map.addControl(new mapboxgl.ScaleControl({ unit: 'metric' }), 'bottom-left');
    this.map.addControl(
      new mapboxgl.GeolocateControl({
        positionOptions: { enableHighAccuracy: true },
        trackUserLocation: true,
        showUserHeading: true,
      }),
      'top-right'
    );

    this.popup = new mapboxgl.Popup({
      closeButton: true,
      closeOnClick: false,
      maxWidth: '280px',
      className: 'pothole-popup',
    });

    this.map.on('load', () => {
      this._addStarImage();
      this._addSources();
      this._addLayers();
      this._bindEvents();
      this._startPolling();
      this._trackVehicle();
      this._ready = true;
      console.log('[MapController] Map ready.');
    });
  }

  // Star Icon────────────

  _addStarImage() {
    // We register one star SVG per risk level so we can style-switch per feature.
    ['near', 'medium', 'far', 'unknown'].forEach(risk => {
      const colour = RISK_COLOURS[risk];
      const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24">
          <polygon
            points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02
                    12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"
            fill="${colour}"
            stroke="rgba(255,255,255,0.85)"
            stroke-width="1.2"
          />
        </svg>`;
      const blob = new Blob([svg], { type: 'image/svg+xml' });
      const url  = URL.createObjectURL(blob);
      const img  = new Image(32, 32);
      img.onload = () => {
        if (!this.map.hasImage(`star-${risk}`)) {
          this.map.addImage(`star-${risk}`, img);
        }
        URL.revokeObjectURL(url);
      };
      img.src = url;
    });
  }

  // Sources 

  _addSources() {
    // Directions route source
    this.map.addSource('directions-route', {
      type: 'geojson',
      data: { type: 'Feature', geometry: { type: 'LineString', coordinates: [] } },
    });

    // Main pothole source — Mapbox handles clustering internally
    this.map.addSource(SOURCE_ID, {
      type:                 'geojson',
      data:                 this._emptyFC(),
      cluster:              true,
      clusterMaxZoom:       CLUSTER_MAX_ZOOM,
      clusterRadius:        CLUSTER_RADIUS,
      clusterProperties: {
        // Aggregate max confidence in each cluster
        max_confidence: ['max', ['get', 'confidence']],
      },
    });

    // Separate un-clustered source for heatmap (no cluster flag)
    this.map.addSource(HEATMAP_SOURCE_ID, {
      type: 'geojson',
      data: this._emptyFC(),
    });

    // Vehicle path source
    this.map.addSource('vehicle-path', {
      type: 'geojson',
      data: { type: 'Feature', geometry: { type: 'LineString', coordinates: [] } },
    });
  }

  // Layers ─

  _addLayers() {
    // 0. Directions route───
    this.map.addLayer({
      id:     'directions-route',
      type:   'line',
      source: 'directions-route',
      layout: { 'line-join': 'round', 'line-cap': 'round' },
      paint: {
        'line-color':   '#a371f7',
        'line-width':   5,
        'line-opacity': 0.85,
      },
    });

    // 1. Heatmap (hidden by default, toggleable)─
    this.map.addLayer({
      id:     'pothole-heatmap',
      type:   'heatmap',
      source: HEATMAP_SOURCE_ID,
      layout: { visibility: 'none' },
      paint: {
        'heatmap-weight':    ['interpolate', ['linear'], ['get', 'confidence'], 0, 0, 1, 1],
        'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 15, 3],
        'heatmap-color': [
          'interpolate', ['linear'], ['heatmap-density'],
          0,    'rgba(33,102,172,0)',
          0.2,  'rgb(103,169,207)',
          0.4,  'rgb(209,229,240)',
          0.6,  'rgb(253,219,199)',
          0.8,  'rgb(239,138,98)',
          1,    'rgb(178,24,43)',
        ],
        'heatmap-radius':   ['interpolate', ['linear'], ['zoom'], 0, 2, 15, 20],
        'heatmap-opacity':  0.75,
      },
    });

    // 2. Cluster circles
    this.map.addLayer({
      id:     'clusters',
      type:   'circle',
      source: SOURCE_ID,
      filter: ['has', 'point_count'],
      paint: {
        'circle-color': [
          'step', ['get', 'point_count'],
          '#3fb950',   5,   // ≤ 5  green
          '#e3a11c',  20,   // ≤ 20 orange
          '#f85149',        // > 20 red
        ],
        'circle-radius': [
          'step', ['get', 'point_count'],
          18,  10,
          22,  30,
          28,
        ],
        'circle-stroke-width': 2,
        'circle-stroke-color': 'rgba(255,255,255,0.5)',
        'circle-opacity': 0.85,
      },
    });

    // 3. Cluster count labels
    this.map.addLayer({
      id:     'cluster-count',
      type:   'symbol',
      source: SOURCE_ID,
      filter: ['has', 'point_count'],
      layout: {
        'text-field':  '{point_count_abbreviated}',
        'text-font':   ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
        'text-size':   13,
      },
      paint: {
        'text-color': '#ffffff',
      },
    });

    // 4. Unclustered star markers
    this.map.addLayer({
      id:     'unclustered-pothole',
      type:   'symbol',
      source: SOURCE_ID,
      filter: ['!', ['has', 'point_count']],
      layout: {
        'icon-image': [
          'match', ['get', 'risk_level'],
          'near',   'star-near',
          'medium', 'star-medium',
          'far',    'star-far',
          'star-unknown',
        ],
        'icon-size':          0.9,
        'icon-allow-overlap': true,
        'icon-anchor':        'center',
      },
    });

    // 5. Vehicle path───
    this.map.addLayer({
      id:     'vehicle-path',
      type:   'line',
      source: 'vehicle-path',
      layout: {
        'line-join': 'round',
        'line-cap':  'round',
      },
      paint: {
        'line-color':   '#58a6ff',
        'line-width':   2.5,
        'line-opacity': 0.7,
        'line-dasharray': [2, 1],
      },
    });
  }

  // Events ─

  _bindEvents() {
    // Expand cluster on click
    this.map.on('click', 'clusters', (e) => {
      const features = this.map.queryRenderedFeatures(e.point, { layers: ['clusters'] });
      const clusterId = features[0].properties.cluster_id;
      this.map.getSource(SOURCE_ID).getClusterExpansionZoom(clusterId, (err, zoom) => {
        if (err) return;
        this.map.easeTo({ center: features[0].geometry.coordinates, zoom });
      });
    });

    // Popup on unclustered marker click
    this.map.on('click', 'unclustered-pothole', (e) => {
      const coords = e.features[0].geometry.coordinates.slice();
      const props  = e.features[0].properties;
      this._showPopup(coords, props);
    });

    // Pointer cursor on hover
    ['clusters', 'unclustered-pothole'].forEach(layer => {
      this.map.on('mouseenter', layer, () => {
        this.map.getCanvas().style.cursor = 'pointer';
      });
      this.map.on('mouseleave', layer, () => {
        this.map.getCanvas().style.cursor = '';
      });
    });
  }

  // Popup

  _showPopup(coords, props) {
    const conf   = typeof props.confidence === 'number'
      ? (props.confidence * 100).toFixed(0) + '%'
      : 'N/A';
    const dist   = props.estimated_distance_m != null
      ? parseFloat(props.estimated_distance_m).toFixed(1) + ' m'
      : 'N/A';
    const risk   = (props.risk_level || 'unknown').toUpperCase();
    const time   = props.last_seen
      ? new Date(props.last_seen).toLocaleString()
      : (props.first_seen ? new Date(props.first_seen).toLocaleString() : '—');
    const count  = props.detection_count || 1;
    const riskColour = RISK_COLOURS[props.risk_level] || RISK_COLOURS.unknown;

    const snapHTML = props.snapshot_path
      ? `<img src="/static/${props.snapshot_path}"
              style="width:100%;border-radius:6px;margin-bottom:10px;display:block"
              onerror="this.style.display='none'">`
      : '';

    const mapsURL = `https://www.openstreetmap.org/?mlat=${coords[1]}&mlon=${coords[0]}&zoom=17`;

    const html = `
      <div class="popup-inner">
        ${snapHTML}
        <div class="popup-title">
          <span class="popup-dot" style="background:${riskColour}"></span>
          Pothole #${props.id}
        </div>
        <table class="popup-table">
          <tr><td>Risk</td><td style="color:${riskColour};font-weight:700">${risk}</td></tr>
          <tr><td>Confidence</td><td>${conf}</td></tr>
          <tr><td>Distance</td><td>${dist}</td></tr>
          <tr><td>Detections</td><td>${count}×</td></tr>
          <tr><td>Last seen</td><td>${time}</td></tr>
        </table>
        <div class="popup-actions">
          <button onclick="window.reportPothole(${props.id})" class="popup-btn popup-btn-report">
            Report SHA
          </button>
          <a href="${mapsURL}" target="_blank" class="popup-btn popup-btn-map">
            OSM
          </a>
        </div>
      </div>`;

    this.popup
      .setLngLat(coords)
      .setHTML(html)
      .addTo(this.map);
  }

  // Polling 

  _startPolling() {
    this._fetchAndUpdate();  // immediate first load
    this._pollTimer = setInterval(() => this._fetchAndUpdate(), POLL_INTERVAL_MS);
  }

  stopPolling() {
    clearInterval(this._pollTimer);
    this._pollTimer = null;
  }

  async _fetchAndUpdate() {
    try {
      const res  = await fetch('/api/map-data');
      const fc   = await res.json();
      this._updateSources(fc);
    } catch (e) {
      console.warn('[MapController] Poll error:', e.message);
    }
  }

  _updateSources(fc) {
    if (!this._ready) return;

    // Cache the last known FeatureCollection for client-side filtering
    this._lastFC = fc;

    // Update main clustered source
    const src = this.map.getSource(SOURCE_ID);
    if (src) src.setData(fc);

    // Update heatmap source (same data, different source)
    const hmSrc = this.map.getSource(HEATMAP_SOURCE_ID);
    if (hmSrc) hmSrc.setData(fc);

    // Track newly added IDs for client-side awareness
    (fc.features || []).forEach(f => this._knownIds.add(f.properties.id));
  }

  // Immediate client-side marker removal (no network round-trip)────

  removeMarkerById(id) {
    if (!this._ready) return;
    this._knownIds.delete(id);
    const fc = {
      type: 'FeatureCollection',
      features: (this._lastFC?.features || []).filter(f => f.properties.id !== id),
    };
    this._lastFC = fc;
    const src   = this.map.getSource(SOURCE_ID);
    const hmSrc = this.map.getSource(HEATMAP_SOURCE_ID);
    if (src)   src.setData(fc);
    if (hmSrc) hmSrc.setData(fc);
  }

  clearAllMapMarkers() {
    if (!this._ready) return;
    this._knownIds.clear();
    const empty = this._emptyFC();
    this._lastFC = empty;
    const src   = this.map.getSource(SOURCE_ID);
    const hmSrc = this.map.getSource(HEATMAP_SOURCE_ID);
    if (src)   src.setData(empty);
    if (hmSrc) hmSrc.setData(empty);
  }

  // Public: update from SocketIO

  /**
   * Called by dashboard.js when the server emits 'map_data' via SocketIO.
   * Converts flat array to GeoJSON and patches the source.
   *
   * @param {Array} potholes  [{id, latitude, longitude, confidence, ...}]
   */
  updateMarkers(potholes) {
    const features = potholes
      .filter(p => p.latitude != null && p.longitude != null)
      .map(p => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [p.longitude, p.latitude] },
        properties: {
          id:                   p.id,
          confidence:           p.confidence,
          risk_level:           p.risk_level || 'far',
          first_seen:           p.first_seen,
          last_seen:            p.last_seen,
          detection_count:      p.detection_count || 1,
          estimated_distance_m: p.estimated_distance_m,
          snapshot_path:        p.snapshot_path,
          crossed:              p.crossed,
          reported:             p.reported,
        },
      }));

    this._updateSources({ type: 'FeatureCollection', features });
  }

  // Vehicle Tracking─────

  _trackVehicle() {
    if (!navigator.geolocation) return;

    // Custom pulsing dot element
    const el = document.createElement('div');
    el.className = 'vehicle-marker';

    // 100 m off-route threshold before triggering a re-route
    const REROUTE_THRESHOLD_M = 100;

    navigator.geolocation.watchPosition(
      (pos) => {
        const { latitude: lat, longitude: lng } = pos.coords;

        if (!this.vehicleMarker) {
          this.vehicleMarker = new mapboxgl.Marker({ element: el, anchor: 'center' })
            .setLngLat([lng, lat])
            .addTo(this.map);
        } else {
          this.vehicleMarker.setLngLat([lng, lat]);
        }

        // Append to breadcrumb path
        this.pathCoords.push([lng, lat]);
        if (this.pathCoords.length > PATH_MAX_POINTS) this.pathCoords.shift();
        const pathSrc = this.map.getSource('vehicle-path');
        if (pathSrc) {
          pathSrc.setData({
            type: 'Feature',
            geometry: { type: 'LineString', coordinates: this.pathCoords },
          });
        }

        // Auto re-route when the vehicle strays more than REROUTE_THRESHOLD_M from
        // the active route (e.g. turn missed, road closed).  The check is free
        // client-side; _autoReroute() debounces the actual API call.
        if (this._routeCoords && this._routeDestC) {
          const deviance = this._distanceToRoute(lat, lng);
          if (deviance > REROUTE_THRESHOLD_M) {
            this._autoReroute(lat, lng);
          }
        }
      },
      (err) => console.warn('[MapController] GPS:', err.message),
      { enableHighAccuracy: true, maximumAge: 2000, timeout: 10000 }
    );
  }

  // Heatmap Toggle───────

  toggleHeatmap(visible) {
    if (!this._ready) return;
    this._heatmapVisible = visible ?? !this._heatmapVisible;
    this.map.setLayoutProperty(
      'pothole-heatmap',
      'visibility',
      this._heatmapVisible ? 'visible' : 'none'
    );
  }

  // Fly to pothole───────

  flyTo(lat, lng, zoom = 16) {
    this.map.flyTo({ center: [lng, lat], zoom, speed: 1.5, curve: 1 });
  }

  // Focus a specific pothole (called from Detections panel "📍 Map" button) ─

  focusPothole(id, lat, lng) {
    if (!this._ready) return;
    this.map.flyTo({ center: [lng, lat], zoom: 17, speed: 1.5, curve: 1 });
    this.map.once('moveend', () => {
      // Find the feature in the cached source so we can open its popup
      const feature = (this._lastFC?.features || []).find(f => f.properties.id === id);
      if (feature) {
        this._showPopup(feature.geometry.coordinates, feature.properties);
      }
    });
  }

  // Style Toggle (Light / Dark)

  toggleStyle() {
    if (!this._ready) return;
    this._isDark  = !this._isDark;
    this._ready   = false;  // block source updates during reload
    const newStyle = this._isDark
      ? 'mapbox://styles/mapbox/dark-v11'
      : 'mapbox://styles/mapbox/streets-v12';
    this.map.setStyle(newStyle);
    this.map.once('style.load', () => {
      this._addStarImage();
      this._addSources();
      this._addLayers();
      this._bindEvents();
      this._ready = true;
      this._fetchAndUpdate();
    });
  }

  // 3D View Toggle───────

  toggle3D() {
    if (!this._ready) return;
    this._is3D = !this._is3D;
    this.map.easeTo({
      pitch:    this._is3D ? 50  : 0,
      bearing:  this._is3D ? -20 : 0,
      duration: 800,
    });
  }

  // Directions───────────

  async _geocode(query) {
    const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json` +
                `?access_token=${this.token}&limit=1`;
    const res  = await fetch(url);
    const data = await res.json();
    if (!data.features?.length) throw new Error(`No results for "${query}"`);
    return data.features[0].center;  // [lng, lat]
  }

  async getDirections(fromQuery, toQuery) {
    const [fromC, toC] = await Promise.all([
      this._geocode(fromQuery),
      this._geocode(toQuery),
    ]);

    return this._fetchAndDrawRoute(fromC, toC);
  }

  /**
   * Core routing method.  Accepts resolved [lng,lat] pairs so it can be called
   * both from the user-facing getDirections() and from _autoReroute().
   *
   * Key: overview=full returns every intermediate road waypoint.
   * Without it, Mapbox returns overview=simplified — a coarse 3–5 point
   * approximation that produces the angular, straight-line-looking segments
   * visible in the UI.
   */
  async _fetchAndDrawRoute(fromC, toC) {
    const coords = `${fromC[0]},${fromC[1]};${toC[0]},${toC[1]}`;
    const url = `https://api.mapbox.com/directions/v5/mapbox/driving/${coords}` +
                `?geometries=geojson&overview=full&steps=false&access_token=${this.token}`;
    const res  = await fetch(url);
    const data = await res.json();

    if (!data.routes?.length) throw new Error('No route found between those locations');

    const route       = data.routes[0];
    const routeCoords = route.geometry.coordinates;

    // Store for auto re-routing deviation checks
    this._routeCoords = routeCoords;
    this._routeDestC  = toC;

    const src = this.map.getSource('directions-route');
    if (src) src.setData({ type: 'Feature', geometry: route.geometry });

    // Fit viewport to show the full route with padding
    const bounds = routeCoords.reduce(
      (b, c) => b.extend(c),
      new mapboxgl.LngLatBounds(routeCoords[0], routeCoords[0])
    );
    this.map.fitBounds(bounds, { padding: 60, duration: 1000 });

    return {
      distance: (route.distance / 1000).toFixed(1) + ' km',
      duration: Math.round(route.duration / 60) + ' min',
    };
  }

  clearDirections() {
    const src = this.map.getSource('directions-route');
    if (src) src.setData({ type: 'Feature', geometry: { type: 'LineString', coordinates: [] } });
    this._routeCoords = null;
    this._routeDestC  = null;
  }

  // Haversine distance (metres)

  _haversine(lat1, lon1, lat2, lon2) {
    const R  = 6_371_000;
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lon2 - lon1) * Math.PI / 180;
    const a  = Math.sin(Δφ / 2) ** 2 + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /** Return the shortest distance (metres) from a point to any coordinate on the active route. */
  _distanceToRoute(lat, lng) {
    if (!this._routeCoords?.length) return Infinity;
    let min = Infinity;
    for (const [rLng, rLat] of this._routeCoords) {
      const d = this._haversine(lat, lng, rLat, rLng);
      if (d < min) min = d;
    }
    return min;
  }

  /** Re-route from current GPS position to the stored destination. */
  async _autoReroute(lat, lng) {
    if (this._reRouting || !this._routeDestC) return;
    this._reRouting = true;
    try {
      await this._fetchAndDrawRoute([lng, lat], this._routeDestC);
    } catch (e) {
      console.warn('[MapController] Auto re-route failed:', e.message);
    } finally {
      this._reRouting = false;
    }
  }

  // Helpers 

  _emptyFC() {
    return { type: 'FeatureCollection', features: [] };
  }

  _showNoTokenPlaceholder() {
    const el = document.getElementById(this.containerId);
    if (!el) return;
    el.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;
                  justify-content:center;height:100%;color:#8b949e;
                  font-family:Inter,sans-serif;gap:12px;background:#0d1117">
        <div style="font-size:2.5rem">🗺️</div>
        <div style="font-size:0.95rem;font-weight:600">Mapbox token not configured</div>
        <div style="font-size:0.78rem;color:#484f58;text-align:center;max-width:220px">
          Add <code style="color:#58a6ff">MAPBOX_ACCESS_TOKEN</code> to your
          <code style="color:#58a6ff">.env</code> file and restart the server.
        </div>
        <a href="https://account.mapbox.com/" target="_blank"
           style="font-size:0.78rem;color:#58a6ff;margin-top:4px">
          Get a free token →
        </a>
      </div>`;
  }
}


// Popup Styles (injected at runtime)

(function injectPopupStyles() {
  if (document.getElementById('mapbox-popup-styles')) return;
  const style = document.createElement('style');
  style.id = 'mapbox-popup-styles';
  style.textContent = `
    .mapboxgl-popup-content {
      background: #1c2128;
      color: #e6edf3;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 0;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6);
      font-family: Inter, sans-serif;
      min-width: 220px;
    }
    .mapboxgl-popup-close-button {
      color: #8b949e;
      font-size: 1rem;
      padding: 6px 8px;
    }
    .mapboxgl-popup-tip { border-top-color: #1c2128; }

    .popup-inner { padding: 14px; }

    .popup-title {
      font-size: 0.92rem;
      font-weight: 700;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .popup-dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      display: inline-block;
      flex-shrink: 0;
    }
    .popup-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      margin-bottom: 12px;
    }
    .popup-table td {
      padding: 3px 6px 3px 0;
      vertical-align: top;
    }
    .popup-table td:first-child { color: #8b949e; padding-right: 10px; }

    .popup-actions { display: flex; gap: 8px; }
    .popup-btn {
      flex: 1;
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 500;
      cursor: pointer;
      text-align: center;
      text-decoration: none;
      border: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 4px;
    }
    .popup-btn-report { background: #58a6ff; color: #fff; }
    .popup-btn-report:hover { background: #79b8ff; }
    .popup-btn-map    { background: #3fb950; color: #fff; }
    .popup-btn-map:hover { background: #56d364; }

    /* Pulsing vehicle dot */
    .vehicle-marker {
      width: 18px; height: 18px;
      border-radius: 50%;
      background: #58a6ff;
      border: 3px solid #fff;
      box-shadow: 0 0 0 0 rgba(88,166,255,0.6);
      animation: vehicle-pulse 2s infinite;
    }
    @keyframes vehicle-pulse {
      0%   { box-shadow: 0 0 0 0   rgba(88,166,255,0.6); }
      70%  { box-shadow: 0 0 0 10px rgba(88,166,255,0);   }
      100% { box-shadow: 0 0 0 0   rgba(88,166,255,0);   }
    }
  `;
  document.head.appendChild(style);
})();


// Module initialisation────

/**
 * Reads map config from data-* attributes on <body> — values are injected
 * there by Flask's render_template(), keeping Jinja2 expressions out of JS
 * source files entirely (avoids IDE linter false-positives).
 *
 * <body
 *   data-mapbox-token="pk.xxx"
 *   data-mapbox-style="mapbox://styles/mapbox/dark-v11"
 *   data-default-lat="39.2904"
 *   data-default-lng="-76.6122"
 *   data-default-zoom="13"
 * >
 */
function initMapbox() {
  const ds = document.body.dataset;
  window.mapController = new PotholeMapController('map', {
    token:       ds.mapboxToken  || '',
    style:       ds.mapboxStyle  || 'mapbox://styles/mapbox/dark-v11',
    defaultLat:  parseFloat(ds.defaultLat)  || 39.2904,
    defaultLng:  parseFloat(ds.defaultLng)  || -76.6122,
    defaultZoom: parseFloat(ds.defaultZoom) || 13,
  });
  window.mapController.init();
}

// Auto-init once the DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initMapbox);
} else {
  initMapbox();
}
