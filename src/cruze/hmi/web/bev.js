/* Plan view — the road from above.

   Every object the pipeline tracks already has a ground-plane position: the
   bottom centre of its box, projected through the camera intrinsics to metres
   (see perception/depth.ground_position_xz, the (u,v) → (X,Y) step of the
   reference architecture). The camera view shows what is there; this shows
   where it is. Distance in a perspective image is exactly the thing a driver
   cannot read at a glance, so it gets its own instrument.

   Ego sits at the bottom centre. Forward range runs up the canvas. */

import { rgba, trackColor, urgencyColor } from "./render.js";

const PAINT = "#F2EFE6";
const INTENT = "#7C5CFF";
const GRID = "rgba(242, 239, 230, 0.10)";

// Field of view of the plan, in metres. 60 m forward covers the useful range of
// monocular depth (past that the estimate is noise) and roughly two seconds of
// motorway travel; ±10 m lateral spans about three lanes.
const RANGE_M = 60;
const HALF_WIDTH_M = 10;
// Range rings every 20 m — far enough apart to stay legible at this size.
const RING_STEP_M = 20;

export function drawBev(canvas, scene, dpr = 1) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const originX = w / 2;
  const originY = h - 14;          // ego sits just above the lower edge
  const pxPerM = (originY - 8) / RANGE_M;
  const lateralPxPerM = (w / 2 - 6) / HALF_WIDTH_M;

  const toX = (x) => originX + x * lateralPxPerM;
  const toY = (z) => originY - z * pxPerM;

  drawGrid(ctx, w, originX, originY, toY);
  drawCorridor(ctx, scene, toX, toY);
  drawEgo(ctx, originX, originY);
  drawObjects(ctx, scene, toX, toY, w);
}

/* Range rings and the lane-width reference: structure that says what the
   distances mean, not decoration. */
function drawGrid(ctx, w, originX, originY, toY) {
  ctx.save();
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 1;
  for (let z = RING_STEP_M; z <= RANGE_M; z += RING_STEP_M) {
    const y = toY(z);
    ctx.beginPath();
    ctx.moveTo(4, y);
    ctx.lineTo(w - 4, y);
    ctx.stroke();
    ctx.fillStyle = "rgba(242, 239, 230, 0.32)";
    ctx.font = "500 8px Bahnschrift, 'DIN Alternate', system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${z}`, 5, y - 3);
  }
  // Ego lane edges at ±1.85 m (a US lane is ~3.7 m wide).
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = "rgba(242, 239, 230, 0.16)";
  for (const side of [-1, 1]) {
    const x = originX + side * 1.85 * ((w / 2 - 6) / HALF_WIDTH_M);
    ctx.beginPath();
    ctx.moveTo(x, originY);
    ctx.lineTo(x, toY(RANGE_M));
    ctx.stroke();
  }
  ctx.restore();
}

/* The planned path in plan space, coloured by braking urgency — the same
   number that colours the corridor over the camera view. */
function drawCorridor(ctx, scene, toX, toY) {
  const accel = scene ? scene.accel : null;
  const color = urgencyColor(accel);
  ctx.save();
  const grad = ctx.createLinearGradient(0, toY(0), 0, toY(RANGE_M));
  grad.addColorStop(0, rgba(color, 0.2));
  grad.addColorStop(1, rgba(color, 0));
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(toX(-1.6), toY(0));
  ctx.lineTo(toX(-1.0), toY(RANGE_M));
  ctx.lineTo(toX(1.0), toY(RANGE_M));
  ctx.lineTo(toX(1.6), toY(0));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawEgo(ctx, x, y) {
  ctx.save();
  ctx.fillStyle = INTENT;
  ctx.beginPath();
  ctx.moveTo(x, y - 9);
  ctx.lineTo(x + 6, y + 4);
  ctx.lineTo(x, y + 1);
  ctx.lineTo(x - 6, y + 4);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/* Tracked objects at their measured position. Marks are sized by class, not by
   distance — a distant lorry is still a lorry, and scaling by range would
   double-encode the axis the view already shows. */
function drawObjects(ctx, scene, toX, toY, w) {
  if (!scene || !scene.tracks) return;
  ctx.save();
  for (const t of scene.tracks) {
    if (!t.xz) continue;
    const [lateral, forward] = t.xz;
    if (forward > RANGE_M || Math.abs(lateral) > HALF_WIDTH_M) continue;

    const x = toX(lateral);
    const y = toY(forward);
    const isLead = t.id === scene.lead_id;
    const color = isLead ? INTENT : trackColor(t);
    const big = t.cls === "truck" || t.cls === "bus";
    const halfW = big ? 5 : 4;
    const halfH = big ? 8 : 6;

    if (t.cls === "person" || t.cls === "bicycle") {
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    } else {
      ctx.fillStyle = rgba(color, isLead ? 0.9 : 0.7);
      ctx.fillRect(x - halfW, y - halfH, halfW * 2, halfH * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - halfW, y - halfH, halfW * 2, halfH * 2);
    }

    if (isLead) {
      // Range tick from ego to the lead: the gap, drawn as a gap.
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = rgba(INTENT, 0.5);
      ctx.beginPath();
      ctx.moveTo(x, y + halfH);
      ctx.lineTo(x, toY(0) - 10);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  ctx.restore();
}
