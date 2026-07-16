/* Overlay renderer — pure drawing, no app state.
   All coordinates arrive in frame-pixel space; app.js keeps the overlay
   canvas the same pixel size as the video frame, so no scaling math here. */

// Cockpit semantics: vehicle targets in the green family, vulnerable road
// users amber, mandatory-stop red. Traffic lights take their lamp colour.
const CLASS_COLOR = {
  car: "#39ff6a",
  truck: "#2ec95c",
  bus: "#27b350",
  motorcycle: "#7dffa3",
  bicycle: "#a5ffbe",
  person: "#ffb000",
  stop_sign: "#ff3b30",
  traffic_light: "#9adfae",
  unknown: "#6b8f74",
};

const LIGHT_COLOR = {
  red: "#ff3b30",
  yellow: "#ffb000",
  green: "#39ff6a",
  unknown: "#8a8a8a",
};

const LANE_COLOR = "#39ff6a";
const LEAD_COLOR = "#b6ffcb";
const MPH = 2.237; // m/s → mph

/* IDM urgency bands (m/s² of demanded deceleration), mirroring the
   reasoning config: < -5 emergency, -5..-3 brake-advised, -3..-1 firm. */
function urgencyColor(accel) {
  if (accel == null) return LANE_COLOR;
  if (accel < -5) return "#ff3b30";
  if (accel < -3) return "#ff7a00";
  if (accel < -1) return "#ffb000";
  return LANE_COLOR;
}

function trackColor(t) {
  if (t.cls === "traffic_light" && t.light) return LIGHT_COLOR[t.light] || LIGHT_COLOR.unknown;
  return CLASS_COLOR[t.cls] || CLASS_COLOR.unknown;
}

function maskPath(mask) {
  const p = new Path2D();
  p.moveTo(mask[0], mask[1]);
  for (let i = 2; i < mask.length; i += 2) p.lineTo(mask[i], mask[i + 1]);
  p.closePath();
  return p;
}

function drawMask(ctx, t, color) {
  if (!t.mask || t.mask.length < 6) return;
  const path = maskPath(t.mask);
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.14;
  ctx.fill(path);
  ctx.globalAlpha = 0.55;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = color;
  ctx.stroke(path);
  ctx.restore();
}

/* Corner brackets — the target designator. Four L-shapes, no full box. */
function drawBrackets(ctx, t, color, isLead) {
  const [x1, y1, x2, y2] = t.bbox;
  const w = x2 - x1, h = y2 - y1;
  const arm = Math.max(8, Math.min(w, h) * 0.22);

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = isLead ? 3 : 2;
  if (t.cls === "person") ctx.setLineDash([6, 4]);
  if (isLead) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
  }
  ctx.beginPath();
  ctx.moveTo(x1, y1 + arm); ctx.lineTo(x1, y1); ctx.lineTo(x1 + arm, y1);
  ctx.moveTo(x2 - arm, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + arm);
  ctx.moveTo(x2, y2 - arm); ctx.lineTo(x2, y2); ctx.lineTo(x2 - arm, y2);
  ctx.moveTo(x1 + arm, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - arm);
  ctx.stroke();
  ctx.restore();
}

function trackLabel(t, isLead) {
  const parts = [`${t.cls.replace("_", " ").toUpperCase()}-${t.id}`];
  if (t.light) parts.push(t.light.toUpperCase());
  if (t.speed_mps != null) parts.push(`${Math.round(t.speed_mps * MPH)} MPH`);
  if (t.distance_m != null) parts.push(`${Math.round(t.distance_m)} M`);
  if (isLead) parts.push("LEAD");
  return parts.join(" · ");
}

function drawLabel(ctx, t, color, isLead) {
  const [x1, y1] = t.bbox;
  const text = trackLabel(t, isLead);
  ctx.save();
  ctx.font = "600 13px ui-monospace, Consolas, monospace";
  const pad = 4;
  const width = ctx.measureText(text).width + pad * 2;
  const y = Math.max(16, y1 - 6);
  ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
  ctx.fillRect(x1 - 1, y - 13, width, 17);
  ctx.fillStyle = color;
  ctx.fillText(text, x1 + pad - 1, y);
  ctx.restore();
}

/* Mini 3-lamp stack beside a traffic light — the active lamp glows. */
function drawLampStack(ctx, t) {
  const [, y1, x2, y2] = t.bbox;
  const r = Math.max(3, Math.min(6, (y2 - y1) * 0.08));
  const cx = x2 + r * 2.2;
  const states = ["red", "yellow", "green"];
  ctx.save();
  states.forEach((s, i) => {
    const cy = y1 + (y2 - y1) * (0.25 + 0.25 * i);
    const active = t.light === s;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = active ? LIGHT_COLOR[s] : "rgba(120,120,120,0.35)";
    if (active) {
      ctx.shadowColor = LIGHT_COLOR[s];
      ctx.shadowBlur = 12;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
  });
  ctx.restore();
}

/* One boundary as a Path2D: curved polyline when present, else the straight
   segment. Both arrive as flat [x1,y1,x2,y2,...] arrays. */
function boundaryPath(line, poly) {
  const pts = poly && poly.length >= 4 ? poly : line;
  if (!pts) return null;
  const p = new Path2D();
  p.moveTo(pts[0], pts[1]);
  for (let i = 2; i < pts.length; i += 2) p.lineTo(pts[i], pts[i + 1]);
  return p;
}

function topX(line, poly) {
  if (poly && poly.length >= 4) return poly[poly.length - 2];
  return line ? line[2] : null;
}

/* Corridor ring: down the left boundary, back up the right. Skipped when the
   extrapolated tops cross (lines extend past the vanishing point) — a
   self-intersecting fill reads as a glitch. */
function corridorPath(lanes) {
  const lx = topX(lanes.left, lanes.left_poly);
  const rx = topX(lanes.right, lanes.right_poly);
  if (lx == null || rx == null || lx >= rx) return null;
  const lpts = lanes.left_poly && lanes.left_poly.length >= 4 ? lanes.left_poly : lanes.left;
  const rpts = lanes.right_poly && lanes.right_poly.length >= 4 ? lanes.right_poly : lanes.right;
  const p = new Path2D();
  p.moveTo(lpts[0], lpts[1]);
  for (let i = 2; i < lpts.length; i += 2) p.lineTo(lpts[i], lpts[i + 1]);
  for (let i = rpts.length - 2; i >= 0; i -= 2) p.lineTo(rpts[i], rpts[i + 1]);
  p.closePath();
  return p;
}

function drawLanes(ctx, lanes, accel) {
  if (!lanes) return;
  const color = urgencyColor(accel);
  const urgent = accel != null && accel < -3; // brake-advised band and worse

  if (lanes.left && lanes.right) {
    const corridor = corridorPath(lanes);
    if (corridor) {
      ctx.save();
      ctx.fillStyle = color;
      ctx.globalAlpha = urgent ? 0.12 : 0.06;
      ctx.fill(corridor);
      ctx.restore();
    }
  }

  for (const [line, poly] of [
    [lanes.left, lanes.left_poly],
    [lanes.right, lanes.right_poly],
  ]) {
    const path = boundaryPath(line, poly);
    if (!path) continue;
    ctx.save();
    // Wide soft pass then bright core = phosphor glow.
    ctx.strokeStyle = color;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalAlpha = 0.25;
    ctx.lineWidth = 12;
    ctx.shadowColor = color;
    ctx.shadowBlur = 18;
    ctx.stroke(path);
    ctx.globalAlpha = 1;
    ctx.lineWidth = 3;
    ctx.shadowBlur = 0;
    ctx.stroke(path);
    ctx.restore();
  }
}

/* Distance chip pinned under the lead vehicle — the one number that matters. */
function drawLeadChip(ctx, t) {
  if (t.distance_m == null) return;
  const [x1, , x2, y2] = t.bbox;
  const text = `${Math.round(t.distance_m)} M`;
  ctx.save();
  ctx.font = "700 20px ui-monospace, Consolas, monospace";
  const w = ctx.measureText(text).width + 14;
  const cx = (x1 + x2) / 2;
  const y = Math.min(ctx.canvas.height - 8, y2 + 26);
  ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
  ctx.fillRect(cx - w / 2, y - 20, w, 27);
  ctx.strokeStyle = LEAD_COLOR;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(cx - w / 2, y - 20, w, 27);
  ctx.fillStyle = LEAD_COLOR;
  ctx.textAlign = "center";
  ctx.fillText(text, cx, y);
  ctx.restore();
}

/** Draw one full overlay frame for the latest scene. */
export function drawOverlay(ctx, scene) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  if (!scene) return;

  drawLanes(ctx, scene.lanes, scene.accel);

  for (const t of scene.tracks || []) {
    const isLead = t.id === scene.lead_id;
    const color = isLead ? LEAD_COLOR : trackColor(t);
    drawMask(ctx, t, color);
    drawBrackets(ctx, t, color, isLead);
    if (t.cls === "traffic_light") drawLampStack(ctx, t);
    drawLabel(ctx, t, color, isLead);
    if (isLead) drawLeadChip(ctx, t);
  }
}
