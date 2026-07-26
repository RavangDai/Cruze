/* CRUZE dashboard client.

   One WebSocket. Binary messages are a uint32 little-endian frame_id followed
   by JPEG bytes; text messages are JSON with a "type" discriminator
   (scene | event | utterance).

   Frames and overlays are paired by frame_id. The previous client extrapolated
   each box along its measured pixel velocity because scenes arrived well
   behind the video; that guessing is what made the overlay wobble. Now a frame
   is held until the scene computed from it arrives, and the overlay is drawn
   against the image it actually describes. Video still paints at camera rate —
   only the overlay waits. */

import { drawOverlay } from "./render.js";
import { drawBev } from "./bev.js";

const MPH = 2.237;
const TICKER_MAX = 4;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 8000;
// Edge bars stay lit roughly a human reaction window past their event.
const ALERT_TTL_MS = 2500;
// A stage lamp stays lit this long after its stage last produced output.
const STAGE_TTL_MS = 1200;

const $ = (id) => document.getElementById(id);

const videoCanvas = $("video");
const overlayCanvas = $("overlay");
const bevCanvas = $("bev");
const videoCtx = videoCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");

const dpr = Math.min(window.devicePixelRatio || 1, 2);

const state = {
  scene: null,          // newest scene, drives both the overlay and the panels
  frameSize: null,      // { w, h, srcW, srcH } of the streamed bitmap
  lastFrameAt: 0,
  frameTimes: [],
  sceneTimes: [],
  events: [],
  alerts: { left: 0, right: 0, bottom: 0 },
  stages: { capture: 0, pre: 0, nets: 0, fusion: 0, plan: 0 },
};

/* ---------- video ---------- */

// uint32 frame_id, uint16 source width, uint16 source height, then JPEG.
const HEADER_BYTES = 8;

async function onBinary(buffer) {
  const view = new DataView(buffer);
  const frameId = view.getUint32(0, true);
  // Overlay coordinates are in camera-frame pixels; the JPEG is downscaled.
  // The source size travels with the frame rather than being inferred from
  // camera config, which a replayed file or a stubborn capture device ignores.
  const srcW = view.getUint16(4, true);
  const srcH = view.getUint16(6, true);
  const blob = new Blob([buffer.slice(HEADER_BYTES)], { type: "image/jpeg" });
  let bitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch {
    return; // corrupt frame — skip
  }

  sizeCanvases(bitmap.width, bitmap.height, srcW, srcH);

  // Painted once, here, in socket order — and never repainted afterwards.
  videoCtx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height);
  bitmap.close();

  const now = performance.now();
  state.lastFrameAt = now;
  state.frameTimes.push(now);
  while (state.frameTimes.length && now - state.frameTimes[0] > 2000) {
    state.frameTimes.shift();
  }
  markStage("capture");
  $("nosignal").classList.add("hidden");
}

function sizeCanvases(w, h, srcW, srcH) {
  if (state.frameSize && state.frameSize.w === w && state.frameSize.srcW === srcW) return;
  // w/h are the streamed bitmap; srcW/srcH are the space overlay coords use.
  state.frameSize = { w, h, srcW: srcW || w, srcH: srcH || h };
  videoCanvas.width = w;
  videoCanvas.height = h;
  // The overlay is drawn at device resolution rather than the frame's, so text
  // and hairlines stay crisp on a HiDPI screen instead of being upscaled.
  overlayCanvas.width = Math.round(w * dpr);
  overlayCanvas.height = Math.round(h * dpr);
  overlayCanvas.style.width = "100%";
  overlayCanvas.style.height = "100%";
}

/* ---------- scene ---------- */

function onScene(scene) {
  state.scene = scene;
  repaintOverlay();

  const now = performance.now();
  state.sceneTimes.push(now);
  while (state.sceneTimes.length && now - state.sceneTimes[0] > 2000) {
    state.sceneTimes.shift();
  }

  markStage("fusion");
  if (scene.ego_path || scene.curvature != null || scene.cipo_dist != null) markStage("nets");
  if (scene.lanes) markStage("pre");
  if (scene.accel != null) markStage("plan");

  updateInstruments(scene);
  drawBev(bevCanvas, scene, dpr);
}

function fmtSpeed(mps) {
  return mps == null ? "--" : String(Math.round(mps * MPH));
}

// Bicycle-model steer angle from path curvature: δ = atan(L·κ). Display-only
// advisory — a real Planning-stage target tyre angle is not computed yet.
// WHEELBASE_M is a nominal passenger-car value.
const WHEELBASE_M = 2.7;
function steerAngleFromCurvature(kappa) {
  if (kappa == null) return null;
  return (Math.atan(WHEELBASE_M * kappa) * 180) / Math.PI;
}

function accelClass(a) {
  if (a == null || a >= -1) return "value";
  if (a < -5) return "value emergency";
  if (a < -3) return "value brake";
  return "value firm";
}

function updateInstruments(scene) {
  const s = scene.state || {};

  const speed = $("ego-speed");
  speed.textContent = fmtSpeed(s.speed_mps);
  const over = s.speed_mps != null && s.limit_mps != null && s.speed_mps > s.limit_mps;
  speed.className = over ? "value over" : "value";
  $("limit").textContent = fmtSpeed(s.limit_mps);
  $("limit-block").className = over ? "limit over" : "limit";

  $("heading").textContent = s.heading_deg == null ? "---°" : `${Math.round(s.heading_deg)}°`;
  setLamp("lamp-gps", s.lat != null && s.lon != null);
  setLamp("lamp-obd", s.source === "obd");

  // Lead: prefer the tracked lead's own range, fall back to AutoDrive's CIPO
  // distance so the readout still says something when only the net has an
  // opinion. Both are metres to the same object.
  const lead = (scene.tracks || []).find((t) => t.id === scene.lead_id);
  const range = lead && lead.distance_m != null ? lead.distance_m : scene.cipo_dist;
  $("lead-range").textContent = range == null ? "--" : `${Math.round(range)}`;
  $("lead-kind").textContent = lead ? lead.cls.replace("_", " ") : "no target";
  $("lead-closing").textContent =
    lead && lead.closing_mps != null && lead.closing_mps > 0.5
      ? `closing ${lead.closing_mps.toFixed(1)} m/s`
      : "";

  const accel = scene.accel;
  const accelEl = $("accel");
  accelEl.textContent = accel == null ? "--" : accel.toFixed(1);
  accelEl.className = accelClass(accel);
  // Meter fills with braking demand, saturating at the emergency band.
  $("accel-meter").style.width =
    accel == null ? "0%" : `${Math.min(100, Math.max(0, -accel / 6) * 100)}%`;

  const steer = steerAngleFromCurvature(scene.curvature);
  $("steer").textContent = steer == null ? "--" : steer.toFixed(1);
  // ±8° covers the full width of the indicator; beyond that it pins.
  $("wheel-mark").style.transform =
    steer == null ? "translateX(0)" : `translateX(${Math.max(-26, Math.min(26, steer * 3.2))}px)`;

  $("curve").textContent = scene.curvature == null ? "--" : scene.curvature.toFixed(4);
}

function setLamp(id, on) {
  $(id).className = on ? "lamp on" : "lamp";
}

/* ---------- pipeline spine ---------- */

function markStage(name) {
  state.stages[name] = performance.now();
}

function refreshStages() {
  const now = performance.now();
  for (const [name, at] of Object.entries(state.stages)) {
    const el = document.querySelector(`#stages li[data-stage="${name}"]`);
    if (el) el.classList.toggle("live", at > 0 && now - at < STAGE_TTL_MS);
  }
}

/* ---------- events ---------- */

function onEvent(ev) {
  state.events.unshift({ ...ev, at: new Date() });
  state.events.length = Math.min(state.events.length, TICKER_MAX);
  armEdgeAlert(ev);

  $("ticker").replaceChildren(
    ...state.events.map((e) => {
      const li = document.createElement("li");
      li.className = e.level;
      const at = document.createElement("span");
      at.className = "at";
      at.textContent = e.at.toTimeString().slice(0, 8);
      li.append(at, document.createTextNode(
        e.kind.replace(/_/g, " ") +
        (e.context && e.context.ttc_s != null ? ` · ttc ${e.context.ttc_s}s` : "")
      ));
      return li;
    })
  );
}

function onUtterance(u) {
  const line = $("voice-line");
  line.textContent = u.text;
  line.classList.remove("quiet");
}

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

/* ---------- render loop ---------- */

/* Overlay repaint.

   Video and overlay live on separate canvases, so they are updated on their own
   clocks: each frame is painted once, on arrival, in the order the socket
   delivers it; the overlay is repainted whenever a scene arrives. Nothing is
   ever repainted with older content, which is what made the video step
   backward — frame 22, 23, then back to 18.

   An earlier version buffered frames and held each until its own scene landed.
   That bought exact pairing, but perception runs slower than the camera, so
   half the frames can never have a scene of their own and the queue only added
   latency and pacing bugs to reach the same place. The overlay is at most one
   perception period behind the image beneath it — and, unlike the extrapolation
   this replaced, it is always something perception actually computed. */
function repaintOverlay() {
  if (!state.frameSize || !state.scene) return;
  // Camera-frame px → displayed px. The overlay canvas matches the streamed
  // bitmap, so this is the downscale factor the encoder applied.
  const scale = (overlayCanvas.width / dpr) / state.frameSize.srcW;
  drawOverlay(overlayCtx, state.scene, scale, dpr);
}

/* ---------- websocket ---------- */

let reconnectDelay = RECONNECT_MIN_MS;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    reconnectDelay = RECONNECT_MIN_MS;
    $("lamp-link").className = "lamp on";
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
      onBinary(ev.data);
    }
  };

  ws.onclose = () => {
    $("lamp-link").className = "lamp warn";
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
  };

  ws.onerror = () => ws.close();
}

/* ---------- housekeeping ---------- */

function rate(samples) {
  const n = samples.length;
  if (n < 2) return null;
  const span = samples[n - 1] - samples[0];
  return span > 0 ? (n - 1) / (span / 1000) : null;
}

setInterval(() => {
  const fps = rate(state.frameTimes);
  $("cam-fps").textContent = fps == null ? "--.-" : fps.toFixed(1);
  const hz = rate(state.sceneTimes);
  $("scene-hz").textContent = hz == null ? "--.-" : hz.toFixed(1);

  // Server timestamps come from time.monotonic and are not comparable to the
  // client clock, so report how long ago the last frame arrived here.
  const age = state.lastFrameAt ? Math.round(performance.now() - state.lastFrameAt) : null;
  $("frame-age").textContent = age == null ? "---" : String(Math.min(age, 9999));
  if (age != null && age > 3000) $("nosignal").classList.remove("hidden");

  applyEdgeAlerts();
  refreshStages();
}, 400);

// Size the plan view to its box once, and again whenever the layout reflows.
function sizeBev() {
  const rect = bevCanvas.getBoundingClientRect();
  bevCanvas.width = Math.round(rect.width * dpr);
  bevCanvas.height = Math.round(rect.height * dpr);
  drawBev(bevCanvas, state.scene, dpr);
}
window.addEventListener("resize", sizeBev);

sizeBev();
connect();
