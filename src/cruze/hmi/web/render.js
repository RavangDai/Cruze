/* Overlay renderer — pure drawing, no app state.

   Coordinates arrive in camera-frame pixels. The stream is downscaled before
   encoding (hmi.stream_width), so app.js passes a scale factor rather than
   assuming the canvas matches the frame grid.

   Emissive marks are drawn as two passes — a wide low-alpha stroke under a
   narrow bright one. Canvas shadowBlur produces a similar glow but is one of
   the most expensive 2D operations there is, and the old renderer called it
   several times per object per frame. Two strokes cost a fraction of that and
   read closer to real projected light, which falls off linearly rather than
   in a soft gaussian halo. */

/* Palette — see style.css for the derivation. Road-scene materials, plus one
   colour road scenes never contain, reserved for the machine's own intent. */
const PAINT = "#F2EFE6";   // retroreflective white — nominal
const SODIUM = "#FFB43D";  // sodium vapour — attention
const INTENT = "#7C5CFF";  // planned path / designated lead
const FLARE = "#FF4438";   // critical

const LAMP_COLOR = { red: FLARE, yellow: SODIUM, green: "#5BD98A", unknown: "#7A7E88" };

/* Vehicle proximity bands. A visualization advisory, deliberately separate from
   the FCW thresholds in reasoning/ — monocular distance is the primary signal
   and a positive closing rate promotes a box one band early. Metres, empirical. */
const NEAR_M = 12;              // ~one car length plus a gap
const APPROACH_M = 28;          // a tight following gap at city speed
const APPROACH_CLOSING_M = 45;  // amber if actively closing within this range
const CLOSING_MIN_MPS = 0.5;    // ignore sub-noise closing rates
const VEHICLE_CLASSES = new Set(["car", "truck", "bus", "motorcycle"]);

/* IDM urgency bands (m/s² of demanded deceleration), mirroring reasoning
   config: < -5 emergency, -5..-3 brake-advised, -3..-1 firm. */
export function urgencyColor(accel) {
  if (accel == null || accel >= -1) return PAINT;
  if (accel < -5) return FLARE;
  if (accel < -3) return "#FF7A3C";
  return SODIUM;
}

/* A track's mark colour: vehicles by proximity, falling back to neutral paint;
   traffic lights take their lamp colour; people take attention amber. */
export function trackColor(t) {
  if (t.cls === "traffic_light") return LAMP_COLOR[t.light] || LAMP_COLOR.unknown;
  if (t.cls === "person" || t.cls === "bicycle") return SODIUM;
  if (!VEHICLE_CLASSES.has(t.cls)) return PAINT;
  const d = t.distance_m;
  if (d == null) return PAINT;
  if (d < NEAR_M) return FLARE;
  if (d < APPROACH_M) return SODIUM;
  if (t.closing_mps != null && t.closing_mps > CLOSING_MIN_MPS && d < APPROACH_CLOSING_M) {
    return SODIUM;
  }
  return PAINT;
}

/* #rrggbb + alpha → rgba(), for gradients and the underglow pass. */
export function rgba(hex, a) {
  const h = hex.replace("#", "");
  return `rgba(${parseInt(h.slice(0, 2), 16)}, ${parseInt(h.slice(2, 4), 16)}, ${parseInt(h.slice(4, 6), 16)}, ${a})`;
}

/* Emissive stroke: a wide dim pass under a narrow bright one. */
function glowStroke(ctx, path, color, width) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = rgba(color, 0.18);
  ctx.lineWidth = width * 3.5;
  ctx.stroke(path);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke(path);
}

/* ---------- target designators ---------- */

/* Corner brackets, not a closed box: the gap keeps the object itself readable,
   which matters when the whole point is judging what it is doing. */
function bracketPath(bbox) {
  const [x1, y1, x2, y2] = bbox;
  const arm = Math.max(9, Math.min(x2 - x1, y2 - y1) * 0.22);
  const p = new Path2D();
  p.moveTo(x1, y1 + arm); p.lineTo(x1, y1); p.lineTo(x1 + arm, y1);
  p.moveTo(x2 - arm, y1); p.lineTo(x2, y1); p.lineTo(x2, y1 + arm);
  p.moveTo(x2, y2 - arm); p.lineTo(x2, y2); p.lineTo(x2 - arm, y2);
  p.moveTo(x1 + arm, y2); p.lineTo(x1, y2); p.lineTo(x1, y2 - arm);
  return p;
}

function drawBrackets(ctx, t, color, isLead) {
  ctx.save();
  if (t.cls === "person" || t.cls === "bicycle") ctx.setLineDash([6, 5]);
  glowStroke(ctx, bracketPath(t.bbox), isLead ? INTENT : color, isLead ? 3 : 2);
  ctx.restore();
}

/* Type placard, set in the interface's own instrument voice: no track ids, no
   speeds. Lamp state stays — it is the object's state, not metadata about it. */
function trackLabel(t) {
  const type = t.cls.replace("_", " ").toUpperCase();
  return t.cls === "traffic_light" && t.light ? `${type} ${t.light.toUpperCase()}` : type;
}

// A placard is only worth its ink on a target big enough to act on. Near the
// horizon a dozen boxes stack into a few dozen pixels, and labelling them all
// turns the useful part of the frame into a wall of overlapping text that says
// nothing the bracket did not. Expressed as a share of frame height so it holds
// at any capture resolution. The lead is always labelled.
const LABEL_MIN_HEIGHT_FRAC = 0.035;

function drawLabel(ctx, t, color) {
  const [x1, y1] = t.bbox;
  const text = trackLabel(t);
  ctx.save();
  ctx.font = "600 12px Bahnschrift, 'DIN Alternate', 'SF Compact Display', system-ui, sans-serif";
  ctx.letterSpacing = "0.08em";
  const w = ctx.measureText(text).width;
  const y = Math.max(16, y1 - 7);
  // A rule under the text rather than a filled chip: less ink over the road,
  // and it ties the placard to the bracket it belongs to.
  ctx.fillStyle = rgba("#07090C", 0.55);
  ctx.fillRect(x1 - 1, y - 13, w + 10, 15);
  ctx.fillStyle = color;
  ctx.fillRect(x1 - 1, y + 1, w + 10, 1.5);
  ctx.textBaseline = "alphabetic";
  ctx.fillText(text, x1 + 4, y - 2);
  ctx.restore();
}

/* Range readout for the designated lead — the one number worth the ink. */
function drawLeadRange(ctx, t) {
  if (t.distance_m == null) return;
  const [x1, , x2, y2] = t.bbox;
  const text = `${Math.round(t.distance_m)}`;
  ctx.save();
  ctx.font = "700 26px Bahnschrift, 'DIN Alternate', 'SF Compact Display', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const cx = (x1 + x2) / 2;
  const y = Math.min(ctx.canvas.height / (ctx.__dpr || 1) - 20, y2 + 22);
  ctx.fillStyle = rgba("#07090C", 0.6);
  ctx.fillRect(cx - 34, y - 14, 68, 28);
  ctx.fillStyle = INTENT;
  ctx.fillText(text, cx - 6, y);
  ctx.font = "600 11px Bahnschrift, 'DIN Alternate', system-ui, sans-serif";
  ctx.fillStyle = rgba(INTENT, 0.75);
  ctx.fillText("M", cx + 22, y + 2);
  ctx.restore();
}

/* Mini lamp stack beside a traffic light — the active lamp reads at a glance. */
function drawLampStack(ctx, t) {
  const [, y1, x2, y2] = t.bbox;
  const r = Math.max(3, Math.min(6, (y2 - y1) * 0.08));
  const cx = x2 + r * 2.4;
  ctx.save();
  ["red", "yellow", "green"].forEach((lamp, i) => {
    const cy = y1 + (y2 - y1) * (0.25 + 0.25 * i);
    const on = t.light === lamp;
    ctx.beginPath();
    ctx.arc(cx, cy, on ? r : r * 0.7, 0, Math.PI * 2);
    ctx.fillStyle = on ? LAMP_COLOR[lamp] : "rgba(122, 126, 136, 0.3)";
    ctx.fill();
    if (on) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * 2.1, 0, Math.PI * 2);
      ctx.fillStyle = rgba(LAMP_COLOR[lamp], 0.16);
      ctx.fill();
    }
  });
  ctx.restore();
}

/* ---------- road shape ---------- */

// A US lane (~3.7 m) fills roughly this share of a dash cam's lower field of
// view; the band tapers toward the horizon by perspective.
const SYN_HALF_FRAC = 0.17;
// Default horizon (fraction of frame height) when no lane geometry exists.
const HORIZON_FRAC = 0.46;
// cte_m → horizontal shift for the synthetic band. Without calibration the
// metre→pixel gain is unknown, so use a modest fraction of width per metre.
const CTE_FRAC_PER_M = 0.06;

/* A boundary as [[x,y],...], bottom first. Prefers the curved polyline. */
function polyPoints(line, poly) {
  const pts = poly && poly.length >= 4 ? poly : line;
  if (!pts || pts.length < 4) return null;
  const out = [];
  for (let i = 0; i + 1 < pts.length; i += 2) out.push([pts[i], pts[i + 1]]);
  return out;
}

/* x on a boundary polyline at image row y; clamps beyond the sampled span. */
function xAt(points, y) {
  if (y >= points[0][1]) return points[0][0];
  const last = points[points.length - 1];
  if (y <= last[1]) return last[0];
  for (let i = 0; i < points.length - 1; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[i + 1];
    if (y <= y0 && y >= y1) return x0 + ((y - y0) / (y1 - y0)) * (x1 - x0);
  }
  return points[0][0];
}

/* The tracked road shape: one corridor anchored at the car, coloured by
   braking urgency. Reads the same whether two boundaries were found, one, or
   none — which is most of why it replaced the old pair of lane lines. */
function drawRoadShape(ctx, lanes, accel, W, H) {
  const color = urgencyColor(accel);
  const left = lanes ? polyPoints(lanes.left, lanes.left_poly) : null;
  const right = lanes ? polyPoints(lanes.right, lanes.right_poly) : null;

  const yBottom = H;
  let yTop = H * HORIZON_FRAC;
  if (left) yTop = Math.min(yTop, left[left.length - 1][1]);
  if (right) yTop = Math.min(yTop, right[right.length - 1][1]);
  yTop = Math.max(yTop, H * 0.3);  // guard a runaway extrapolated boundary

  const halfBottom = W * SYN_HALF_FRAC;
  const cte = lanes && lanes.cte_m != null ? lanes.cte_m : 0;

  const N = 14;
  const leftEdge = [], rightEdge = [];
  for (let i = 0; i <= N; i++) {
    const y = yBottom + (yTop - yBottom) * (i / N);
    const frac = (yBottom - y) / (yBottom - yTop);  // 0 at the car → 1 at horizon
    let cx, half;
    if (left && right) {
      const lx = xAt(left, y), rx = xAt(right, y);
      cx = (lx + rx) / 2;
      // Lean on the smooth synthetic taper and let the measured width only
      // nudge it: hugging the raw fit reads strict and jitters frame to frame.
      const synth = Math.max(halfBottom * (1 - frac), 6);
      half = Math.max(0.35 * ((rx - lx) / 2) + 0.65 * synth, 10);
    } else {
      half = Math.max(halfBottom * (1 - frac), 6);
      if (left) cx = xAt(left, y) + half;
      else if (right) cx = xAt(right, y) - half;
      else cx = W / 2 - cte * W * CTE_FRAC_PER_M;
    }
    leftEdge.push([cx - half, y]);
    rightEdge.push([cx + half, y]);
  }

  const band = new Path2D();
  band.moveTo(leftEdge[0][0], leftEdge[0][1]);
  for (const [x, y] of leftEdge) band.lineTo(x, y);
  for (let i = rightEdge.length - 1; i >= 0; i--) band.lineTo(rightEdge[i][0], rightEdge[i][1]);
  band.closePath();

  const fill = ctx.createLinearGradient(0, yBottom, 0, yTop);
  fill.addColorStop(0, rgba(color, 0.22));
  fill.addColorStop(0.55, rgba(color, 0.08));
  fill.addColorStop(1, rgba(color, 0));
  ctx.save();
  ctx.fillStyle = fill;
  ctx.fill(band);

  for (const side of [leftEdge, rightEdge]) {
    const edge = new Path2D();
    edge.moveTo(side[0][0], side[0][1]);
    for (const [x, y] of side) edge.lineTo(x, y);
    glowStroke(ctx, edge, color, 2);
  }
  ctx.restore();
}

/* ---------- ego path ---------- */

// Half-width of the planned corridor at the frame bottom, as a share of width.
const EGO_HALF_BOTTOM_FRAC = 0.11;

/* The AutoSteer planned trajectory: the machine's intent, in the one colour a
   road scene never supplies on its own. Rungs across the corridor give it
   depth the way a real projected path marker does. */
function drawEgoPath(ctx, egoPath, W, H) {
  if (!egoPath || egoPath.length < 4) return;
  const pts = [];
  for (let i = 0; i + 1 < egoPath.length; i += 2) pts.push([egoPath[i], egoPath[i + 1]]);
  if (pts.length < 2) return;

  const halfBottom = W * EGO_HALF_BOTTOM_FRAC;
  const leftEdge = [], rightEdge = [];
  let yNear = -Infinity, yFar = Infinity;
  for (const [x, y] of pts) {
    const near = Math.max(0, Math.min(1, y / H));  // 0 at horizon → 1 at the car
    const half = Math.max(halfBottom * near, 4);
    leftEdge.push([x - half, y]);
    rightEdge.push([x + half, y]);
    if (y > yNear) yNear = y;
    if (y < yFar) yFar = y;
  }

  const band = new Path2D();
  band.moveTo(leftEdge[0][0], leftEdge[0][1]);
  for (const [x, y] of leftEdge) band.lineTo(x, y);
  for (let i = rightEdge.length - 1; i >= 0; i--) band.lineTo(rightEdge[i][0], rightEdge[i][1]);
  band.closePath();

  const fill = ctx.createLinearGradient(0, yNear, 0, yFar);
  fill.addColorStop(0, rgba(INTENT, 0.42));
  fill.addColorStop(0.6, rgba(INTENT, 0.16));
  fill.addColorStop(1, rgba(INTENT, 0));
  ctx.save();
  ctx.fillStyle = fill;
  ctx.fill(band);

  // Rungs every few waypoints, fading with distance.
  ctx.lineWidth = 1.5;
  for (let i = 0; i < pts.length; i += 3) {
    const t = Math.max(0, Math.min(1, pts[i][1] / H));
    ctx.strokeStyle = rgba(INTENT, 0.1 + 0.4 * t);
    ctx.beginPath();
    ctx.moveTo(leftEdge[i][0], leftEdge[i][1]);
    ctx.lineTo(rightEdge[i][0], rightEdge[i][1]);
    ctx.stroke();
  }

  const centre = new Path2D();
  centre.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts) centre.lineTo(x, y);
  glowStroke(ctx, centre, INTENT, 2.5);
  ctx.restore();
}

/**
 * Draw one overlay frame.
 *
 * @param ctx     overlay 2D context, already sized to the displayed bitmap
 * @param scene   the scene computed from the frame currently on screen
 * @param scale   camera-frame px → displayed px (the stream is downscaled)
 * @param dpr     device pixel ratio the canvas backing store is scaled by
 */
export function drawOverlay(ctx, scene, scale = 1, dpr = 1) {
  const W = ctx.canvas.width / dpr;
  const H = ctx.canvas.height / dpr;
  ctx.__dpr = dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!scene) return;

  ctx.save();
  ctx.scale(scale, scale);
  const fw = W / scale, fh = H / scale;

  drawRoadShape(ctx, scene.lanes, scene.accel, fw, fh);
  drawEgoPath(ctx, scene.ego_path, fw, fh);

  for (const t of scene.tracks || []) {
    const isLead = t.id === scene.lead_id;
    const color = trackColor(t);
    drawBrackets(ctx, t, color, isLead);
    if (t.cls === "traffic_light") drawLampStack(ctx, t);
    if (isLead || t.bbox[3] - t.bbox[1] >= fh * LABEL_MIN_HEIGHT_FRAC) {
      drawLabel(ctx, t, isLead ? INTENT : color);
    }
    if (isLead) drawLeadRange(ctx, t);
  }
  ctx.restore();
}
