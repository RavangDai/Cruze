/* CRUZE dashboard client.
   One WebSocket: binary messages are JPEG frames, text messages are JSON
   with a "type" discriminator (scene | event | utterance). */

import { drawOverlay } from "./render.js";

const MPH = 2.237;
const TICKER_MAX = 6;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 8000;
// Edge alert bars stay lit this long after their event — roughly a human
// reaction window, matching the cv2 HUD banner spirit.
const ALERT_TTL_MS = 2500;

const $ = (id) => document.getElementById(id);

const videoCanvas = $("video");
const overlayCanvas = $("overlay");
const videoCtx = videoCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

const state = {
  scene: null,
  lastFrameAt: 0,     // performance.now() of last JPEG
  frameTimes: [],     // rolling window for FPS
  connected: false,
  events: [],
  alerts: { left: 0, right: 0, bottom: 0 },  // edge-bar expiry timestamps
};

/* ---------- video frames ---------- */

async function onFrame(buffer) {
  const blob = new Blob([buffer], { type: "image/jpeg" });
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch {
    return; // corrupt frame — skip
  }
  if (videoCanvas.width !== bitmap.width || videoCanvas.height !== bitmap.height) {
    // Both canvases share the frame's pixel grid so overlay coords need no
    // scaling; CSS scales the pair together.
    videoCanvas.width = overlayCanvas.width = bitmap.width;
    videoCanvas.height = overlayCanvas.height = bitmap.height;
  }
  videoCtx.drawImage(bitmap, 0, 0);
  bitmap.close();

  const now = performance.now();
  state.lastFrameAt = now;
  state.frameTimes.push(now);
  while (state.frameTimes.length && now - state.frameTimes[0] > 2000) {
    state.frameTimes.shift();
  }
  $("nosignal").classList.add("hidden");
}

/* ---------- scene / instruments ---------- */

function fmtSpeed(mps) {
  return mps == null ? "--" : String(Math.round(mps * MPH));
}

function onScene(scene) {
  state.scene = scene;
  const s = scene.state || {};

  const ego = $("ego-speed");
  ego.textContent = fmtSpeed(s.speed_mps);
  const over = s.speed_mps != null && s.limit_mps != null && s.speed_mps > s.limit_mps;
  ego.classList.toggle("over-limit", over);

  $("speed-src").textContent = "SRC " + (s.source || "----").toUpperCase().replace("_", "-");
  $("limit").textContent = fmtSpeed(s.limit_mps);
  $("limit-sign").classList.toggle("over", over);
  $("heading").textContent =
    s.heading_deg == null ? "---°" : Math.round(s.heading_deg) + "°";

  setLamp("ann-gps", s.lat != null && s.lon != null);
  setLamp("ann-obd", s.source === "obd");

  renderLead(scene);
  updateNav(s.lat, s.lon, s.heading_deg);
}

function renderLead(scene) {
  const body = $("lead-body");
  const lead = (scene.tracks || []).find((t) => t.id === scene.lead_id);
  body.replaceChildren();
  if (!lead) {
    const none = document.createElement("span");
    none.className = "muted";
    none.textContent = "no lead vehicle";
    body.appendChild(none);
    return;
  }
  const id = document.createElement("span");
  id.className = "lead-id";
  id.textContent = `${lead.cls.toUpperCase()}-${lead.id}`;
  body.appendChild(id);

  const rows = [];
  if (lead.distance_m != null) rows.push(`${Math.round(lead.distance_m)} M AHEAD`);
  if (lead.speed_mps != null) rows.push(`${fmtSpeed(lead.speed_mps)} MPH`);
  if (lead.closing_mps != null && lead.closing_mps > 0.5) {
    rows.push(`CLOSING ${lead.closing_mps.toFixed(1)} M/S`);
  }
  for (const row of rows) {
    body.appendChild(document.createElement("br"));
    body.appendChild(document.createTextNode(row));
  }
}

function setLamp(id, on) {
  const el = $(id);
  el.classList.toggle("on", !!on);
  el.classList.toggle("warn", false);
}

/* ---------- events / ticker ---------- */

function onEvent(ev) {
  state.events.unshift({ ...ev, at: new Date() });
  state.events.length = Math.min(state.events.length, TICKER_MAX);
  armEdgeAlert(ev);

  const ticker = $("ticker");
  ticker.replaceChildren(
    ...state.events.map((e) => {
      const li = document.createElement("li");
      li.className = `level-${e.level}`;
      const ts = document.createElement("span");
      ts.className = "ts";
      ts.textContent = e.at.toTimeString().slice(0, 8);
      li.appendChild(ts);
      const kind = e.kind.replace(/_/g, " ").toUpperCase();
      const extra = e.context && e.context.ttc_s != null ? ` · TTC ${e.context.ttc_s}S` : "";
      li.appendChild(document.createTextNode(kind + extra));
      return li;
    })
  );
}

function onUtterance(u) {
  const line = $("voice-line");
  line.textContent = "▸ " + u.text;
  line.classList.remove("muted");
}

/* ---------- edge alert bars ---------- */

function armEdgeAlert(ev) {
  const until = performance.now() + ALERT_TTL_MS;
  if (ev.kind === "lane_departure" && ev.context && ev.context.side) {
    state.alerts[ev.context.side === "right" ? "right" : "left"] = until;
  } else if (ev.kind === "fcw" || ev.kind === "brake_hard") {
    state.alerts.bottom = until;
  }
  applyEdgeAlerts();
}

function applyEdgeAlerts() {
  const now = performance.now();
  $("edge-left").classList.toggle("active", state.alerts.left > now);
  $("edge-right").classList.toggle("active", state.alerts.right > now);
  $("edge-bottom").classList.toggle("active", state.alerts.bottom > now);
}

/* ---------- nav panel (Leaflet with numeric fallback) ---------- */

let map = null;
let marker = null;
let trail = null;
const trailPoints = [];

function navFallback(lat, lon) {
  $("map").hidden = true;
  const fb = $("nav-fallback");
  fb.hidden = false;
  $("nav-lat").textContent = lat != null ? lat.toFixed(5) : "---.-----";
  $("nav-lon").textContent = lon != null ? lon.toFixed(5) : "---.-----";
}

function updateNav(lat, lon, headingDeg) {
  if (lat == null || lon == null) {
    if (!map) navFallback(lat, lon);
    return;
  }
  if (window.__leafletJsFailed || window.__leafletCssFailed || typeof L === "undefined") {
    navFallback(lat, lon);
    return;
  }
  try {
    if (!map) {
      map = L.map("map", { zoomControl: false, attributionControl: false }).setView([lat, lon], 15);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19 }).addTo(map);
      const arrow = L.divIcon({ className: "", html: "➤", iconSize: [16, 16] });
      marker = L.marker([lat, lon], { icon: arrow }).addTo(map);
      trail = L.polyline([], { color: "#39ff6a", weight: 2, opacity: 0.7 }).addTo(map);
    }
    marker.setLatLng([lat, lon]);
    if (headingDeg != null && marker.getElement()) {
      // ➤ points east at 0° rotation; compass 0° is north → offset −90°.
      marker.getElement().style.transform += ` rotate(${headingDeg - 90}deg)`;
    }
    // Breadcrumb decimated to ~1 Hz, capped at 5 min of driving.
    const nowS = Date.now() / 1000;
    if (!trailPoints.length || nowS - trailPoints[trailPoints.length - 1].t >= 1) {
      trailPoints.push({ lat, lon, t: nowS });
      if (trailPoints.length > 300) trailPoints.shift();
      trail.setLatLngs(trailPoints.map((p) => [p.lat, p.lon]));
    }
    map.panTo([lat, lon], { animate: false });
  } catch {
    navFallback(lat, lon);
  }
}

/* ---------- websocket ---------- */

let reconnectDelay = RECONNECT_MIN_MS;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    state.connected = true;
    reconnectDelay = RECONNECT_MIN_MS;
    const link = $("ann-link");
    link.classList.add("on");
    link.classList.remove("warn");
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "scene") onScene(msg);
      else if (msg.type === "event") onEvent(msg);
      else if (msg.type === "utterance") onUtterance(msg);
    } else {
      onFrame(ev.data);
    }
  };

  ws.onclose = () => {
    state.connected = false;
    const link = $("ann-link");
    link.classList.remove("on");
    link.classList.add("warn");
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
  };

  ws.onerror = () => ws.close();
}

/* ---------- render + housekeeping loops ---------- */

function renderLoop() {
  drawOverlay(overlayCtx, state.scene);
  requestAnimationFrame(renderLoop);
}

setInterval(() => {
  // Clock (UTC per the status bar label).
  $("clock").textContent = new Date().toISOString().slice(11, 19) + " UTC";

  // FPS over the rolling window + frame age. Server timestamps are
  // time.monotonic — not comparable to the client clock — so "ms since the
  // frame arrived" is the honest latency readout.
  const n = state.frameTimes.length;
  const span = n > 1 ? state.frameTimes[n - 1] - state.frameTimes[0] : 0;
  $("cam-fps").textContent = span > 0 ? ((n - 1) / (span / 1000)).toFixed(1) : "--.-";

  const age = state.lastFrameAt ? Math.round(performance.now() - state.lastFrameAt) : null;
  $("frame-age").textContent = age == null ? "---" : String(Math.min(age, 9999));
  if (age != null && age > 3000) $("nosignal").classList.remove("hidden");

  applyEdgeAlerts(); // expire stale edge bars
}, 500);

connect();
requestAnimationFrame(renderLoop);
