/* OpenThai2.0-OCR-Text in the browser — onnxruntime-web (WASM).
 *
 * Faithful JS port of iapp/OpenThai2.0-OCR-Text runtime/line_engine.py:
 *   image -> detector (DB prob map -> min-area boxes, unclipped)
 *         -> split at wide blank gaps -> crops from the full image
 *         -> recognizer (H=48 CTC) -> greedy CTC decode
 *         -> thai canonical fix-ups -> reading order (rows, XY-cut columns)
 *         -> line filter -> joined text.
 *
 * Model: https://huggingface.co/iapp/OpenThai2.0-OCR-Text (Apache-2.0)
 */
(function () {
  'use strict';

  // ---- config (defaults + det.yml PostProcess + shipped config.json) -------
  const DET = { limit: 1280, thresh: 0.2, boxThresh: 0.4, unclip: 1.4,
                minSize: 3, maxCandidates: 3000, splitGap: 1.2 };
  const REC = { height: 48, maxW: 3200, batch: 16, dropBelow: 0.8,
                minChars: 2, junkBelow: 0.9, junkInvalid: 0.05 };
  const ORI = { sample: 8, flipBelow: 0.8, flipMargin: 0.15, deskewAbove: 2.5 };
  // ImageNet mean/std applied on BGR channels (the Paddle detector's order).
  const DET_MEAN = [0.485, 0.456, 0.406];
  const DET_STD = [0.229, 0.224, 0.225];

  // ---- small utils ---------------------------------------------------------
  const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);
  const len2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const mean = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

  function median(arr) {
    if (!arr.length) return 0;
    const s = [...arr].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function makeCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  function getData(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  function capCanvas(src, maxSide) {
    const m = Math.max(src.width, src.height);
    if (m <= maxSide) return src;
    const s = maxSide / m;
    const out = makeCanvas(Math.max(1, Math.round(src.width * s)),
                           Math.max(1, Math.round(src.height * s)));
    const ctx = out.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, 0, 0, out.width, out.height);
    return out;
  }

  // ---- linear algebra ------------------------------------------------------
  function solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++)
        if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      if (Math.abs(M[piv][col]) < 1e-12) M[piv][col] = 1e-12;
      const t = M[col]; M[col] = M[piv]; M[piv] = t;
      const d = M[col][col];
      for (let r = col + 1; r < n; r++) {
        const f = M[r][col] / d;
        if (!f) continue;
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let s = M[i][n];
      for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  }

  // Homography src(4 pts) -> dst(4 pts), h22 fixed to 1.
  function homography(src, dst) {
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], [u, v] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -x * v, -y * v]); b.push(v);
    }
    const h = solveLinear(A, b);
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  function invert3(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = c * h - b * i, C = b * f - c * e;
    const D = f * g - d * i, E = a * i - c * g, F = c * d - a * f;
    const G = d * h - e * g, H = b * g - a * h, I = a * e - b * d;
    const det = a * A + b * D + c * G || 1e-12;
    return [A / det, B / det, C / det, D / det, E / det, F / det,
            G / det, H / det, I / det];
  }

  // ---- geometry: DB postprocess -------------------------------------------
  function convexHull(points) {
    const pts = [...points].sort((p, q) => p.x - q.x || p.y - q.y);
    if (pts.length <= 2) return pts;
    const cross = (o, a, b) =>
      (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const lower = [], upper = [];
    for (const p of pts) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
        lower.pop();
      lower.push(p);
    }
    for (let i = pts.length - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
        upper.pop();
      upper.push(p);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }

  // cv2.minAreaRect equivalent. Returns {box: [[x,y] x4], side: min dim}.
  function minAreaRect(points) {
    const hull = convexHull(points);
    if (hull.length < 3) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of hull) {
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      }
      const box = [[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY]];
      return { box, side: Math.min(maxX - minX, maxY - minY) };
    }
    let best = null;
    for (let i = 0; i < hull.length; i++) {
      const p1 = hull[i], p2 = hull[(i + 1) % hull.length];
      const dx = p2.x - p1.x, dy = p2.y - p1.y;
      const l = Math.hypot(dx, dy) || 1;
      const ux = dx / l, uy = dy / l, vx = -uy, vy = ux;
      let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
      for (const p of hull) {
        const pu = p.x * ux + p.y * uy, pv = p.x * vx + p.y * vy;
        if (pu < minU) minU = pu; if (pu > maxU) maxU = pu;
        if (pv < minV) minV = pv; if (pv > maxV) maxV = pv;
      }
      const area = (maxU - minU) * (maxV - minV);
      if (!best || area < best.area) {
        const toXY = (u, v) => [u * ux + v * vx, u * uy + v * vy];
        best = {
          area, wd: maxU - minU, ht: maxV - minV,
          box: [toXY(minU, minV), toXY(maxU, minV), toXY(maxU, maxV), toXY(minU, maxV)],
        };
      }
    }
    return { box: orderPoints(best.box), side: Math.min(best.wd, best.ht) };
  }

  // 4 pts -> TL, TR, BR, BL (as order_points in line_engine.py).
  function orderPoints(pts) {
    const idx = [0, 1, 2, 3].sort((a, b) => pts[a][0] - pts[b][0]);
    const left = [pts[idx[0]], pts[idx[1]]].sort((a, b) => a[1] - b[1]);
    const right = [pts[idx[2]], pts[idx[3]]].sort((a, b) => a[1] - b[1]);
    return [left[0], right[0], right[1], left[1]];
  }

  function quadArea(q) {
    let a = 0;
    for (let i = 0; i < 4; i++) {
      const [x1, y1] = q[i], [x2, y2] = q[(i + 1) % 4];
      a += x1 * y2 - x2 * y1;
    }
    return Math.abs(a / 2);
  }

  function quadPerimeter(q) {
    let p = 0;
    for (let i = 0; i < 4; i++) p += len2(q[i], q[(i + 1) % 4]);
    return p;
  }

  // pyclipper JT_ROUND offset for a rectangle: every edge moves outward by
  // d = area * ratio / length (corners re-fitted by minAreaRect afterwards).
  function unclipBox(box, ratio) {
    const area = quadArea(box), per = quadPerimeter(box);
    if (per === 0) return null;
    const d = area * ratio / per;
    const nrm = (a, b) => {
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const l = Math.hypot(dx, dy) || 1;
      return [dy / l, -dx / l]; // outward normal for the TL,TR,BR,BL (cw, y-down) order
    };
    const out = [];
    for (let i = 0; i < 4; i++) {
      const n1 = nrm(box[(i + 3) % 4], box[i]), n2 = nrm(box[i], box[(i + 1) % 4]);
      out.push([box[i][0] + (n1[0] + n2[0]) * d, box[i][1] + (n1[1] + n2[1]) * d]);
    }
    return out;
  }

  function pointInQuad(px, py, q) {
    let sgn = 0;
    for (let i = 0; i < 4; i++) {
      const [x1, y1] = q[i], [x2, y2] = q[(i + 1) % 4];
      const cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1);
      if (cross !== 0) {
        const s = cross > 0 ? 1 : -1;
        if (sgn === 0) sgn = s; else if (sgn !== s) return false;
      }
    }
    return true;
  }

  // Mean prob inside the quad (cv2.fillPoly + mask mean).
  function boxScore(prob, pw, ph, box) {
    const xs = box.map(p => p[0]), ys = box.map(p => p[1]);
    const x0 = clamp(Math.floor(Math.min(...xs)), 0, pw - 1);
    const x1 = clamp(Math.ceil(Math.max(...xs)), 0, pw - 1);
    const y0 = clamp(Math.floor(Math.min(...ys)), 0, ph - 1);
    const y1 = clamp(Math.ceil(Math.max(...ys)), 0, ph - 1);
    if (x1 < x0 || y1 < y0) return 0;
    let sum = 0, cnt = 0;
    for (let y = y0; y <= y1; y++) {
      const base = y * pw;
      for (let x = x0; x <= x1; x++) {
        if (pointInQuad(x + 0.5, y + 0.5, box)) { sum += prob[base + x]; cnt++; }
      }
    }
    return cnt ? sum / cnt : 0;
  }

  // 8-connected components; returns arrays of {x,y} per component.
  function connectedComponents(bitmap, w, h) {
    const labels = new Int32Array(w * h).fill(-1);
    const comps = [];
    const stack = new Int32Array(w * h);
    for (let start = 0; start < w * h; start++) {
      if (!bitmap[start] || labels[start] !== -1) continue;
      const id = comps.length;
      const pts = [];
      let sp = 0;
      stack[sp++] = start;
      labels[start] = id;
      while (sp > 0) {
        const p = stack[--sp];
        const x = p % w, y = (p / w) | 0;
        pts.push({ x, y });
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx < 0 || nx >= w) continue;
            const np = ny * w + nx;
            if (bitmap[np] && labels[np] === -1) { labels[np] = id; stack[sp++] = np; }
          }
        }
      }
      comps.push(pts);
    }
    return comps;
  }

  // ---- page-level geometry -------------------------------------------------
  function pageSkew(boxes) {
    const angles = [];
    for (const b of boxes) {
      const dx = b[1][0] - b[0][0], dy = b[1][1] - b[0][1];
      const h = len2(b[0], b[3]);
      if (Math.hypot(dx, dy) >= 3 * Math.max(h, 1))
        angles.push(Math.atan2(dy, dx) * 180 / Math.PI);
    }
    return angles.length ? median(angles) : 0;
  }

  function lineHeight(boxes) {
    const hs = boxes.map(b => Math.max(len2(b[0], b[3]), 1));
    const lines = [];
    boxes.forEach((b, i) => {
      if (len2(b[0], b[1]) >= 2 * hs[i]) lines.push(hs[i]);
    });
    const pool = lines.length >= 3 ? lines : hs;
    return pool.length ? median(pool) : 1.0;
  }

  // +1 vertical run, -1 horizontal line, 0 else.
  function verticality(box, ratio) {
    const w = len2(box[0], box[1]), h = len2(box[0], box[3]);
    return h >= ratio * Math.max(w, 1) ? 1 : (w >= ratio * Math.max(h, 1) ? -1 : 0);
  }

  function gaps(iv, minGap) {
    const s = [...iv].sort((a, b) => a[0] - b[0]);
    const out = [];
    let reach = s.length ? s[0][1] : 0;
    for (let i = 1; i < s.length; i++) {
      const [lo, hi] = s[i];
      if (lo - reach >= minGap) out.push([reach, lo]);
      reach = Math.max(reach, hi);
    }
    return out;
  }

  const DIGITS_SET = new Set('0123456789๐๑๒๓๔๕๖๗๘๙');
  const THAI_DIGITS = '๐๑๒๓๔๕๖๗๘๙';

  function enumNum(text) {
    const m = text.match(/^\s*\(?([0-9๐-๙]{1,4})\s*[.)]/);
    if (!m) return null;
    let s = m[1];
    for (let i = 0; i < THAI_DIGITS.length; i++)
      s = s.split(THAI_DIGITS[i]).join(String(i));
    return parseInt(s, 10);
  }

  function isProse(col, ext, hmed, texts) {
    if (col.length < 12) return false;
    const widths = col.map(i => ext[i][1] - ext[i][0]);
    const span = Math.max(...col.map(i => ext[i][1])) - Math.min(...col.map(i => ext[i][0]));
    if (median(widths) < 7 * hmed) return false;
    const fullShare = widths.filter(w => w >= 0.75 * span).length / widths.length;
    if (fullShare < 0.5) return false;
    if (texts) {
      let chars = 0, digits = 0;
      for (const i of col)
        for (const c of texts[i])
          if (!/\s/.test(c)) { chars++; if (DIGITS_SET.has(c)) digits++; }
      if (chars && digits > 0.3 * chars) return false;
    }
    return true;
  }

  function nextTo(side, ext, hmed, gutterOnRight) {
    if (!side.length) return side;
    const g = gaps(side.map(i => [ext[i][0], ext[i][1]]), 0.8 * hmed);
    if (!g.length) return side;
    if (gutterOnRight) return side.filter(i => ext[i][0] >= g[g.length - 1][1]);
    return side.filter(i => ext[i][1] <= g[0][0]);
  }

  function numberedAcross(lcol, rcol, ext, hmed, texts) {
    if (!texts) return false;
    let pairs = 0;
    for (const i of lcol) {
      const a = enumNum(texts[i]);
      if (a === null) continue;
      const cy = (ext[i][2] + ext[i][3]) / 2;
      for (const j of rcol) {
        if (enumNum(texts[j]) === a + 1 &&
            Math.abs((ext[j][2] + ext[j][3]) / 2 - cy) <= 0.5 * hmed) { pairs++; break; }
      }
    }
    return pairs >= 0.5 * Math.min(lcol.length, rcol.length);
  }

  // Recursive XY-cut into reading-order blocks (line_engine._blocks).
  function blocks(idx, ext, hmed, texts, depth) {
    if (idx.length < 6 || depth > 12) return [idx];
    const cuts = [];
    for (const [g0, g1] of gaps(idx.map(i => [ext[i][0], ext[i][1]]), 0.8 * hmed)) {
      const left = idx.filter(i => ext[i][1] <= g0);
      const right = idx.filter(i => ext[i][0] >= g1);
      const lcol = nextTo(left, ext, hmed, true);
      const rcol = nextTo(right, ext, hmed, false);
      if (isProse(lcol, ext, hmed, texts) && isProse(rcol, ext, hmed, texts) &&
          !numberedAcross(lcol, rcol, ext, hmed, texts))
        cuts.push([g1 - g0, left, right]);
    }
    if (cuts.length) {
      cuts.sort((a, b) => b[0] - a[0]);
      const [, left, right] = cuts[0];
      return [...blocks(left, ext, hmed, texts, depth + 1),
              ...blocks(right, ext, hmed, texts, depth + 1)];
    }
    const hg = gaps(idx.map(i => [ext[i][2], ext[i][3]]), 0.8 * hmed);
    if (!hg.length) return [idx];
    hg.sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]));
    const [g0, g1] = hg[0];
    const top = idx.filter(i => ext[i][3] <= g0);
    const bottom = idx.filter(i => ext[i][2] >= g1);
    return [...blocks(top, ext, hmed, texts, depth + 1),
            ...blocks(bottom, ext, hmed, texts, depth + 1)];
  }

  // Rows top to bottom, each left to right (line_engine.reading_order).
  function readingOrder(boxes, angle, texts) {
    if (!boxes.length) return [];
    const a = -(angle == null ? pageSkew(boxes) : angle) * Math.PI / 180;
    const cos = Math.cos(a), sin = Math.sin(a);
    const ext = boxes.map(b => {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const [x, y] of b) {
        const px = x * cos + y * sin, py = -x * sin + y * cos;
        if (px < minX) minX = px; if (px > maxX) maxX = px;
        if (py < minY) minY = py; if (py > maxY) maxY = py;
      }
      return [minX, maxX, minY, maxY];
    });
    const hmed = lineHeight(boxes);
    const idx = boxes.map((_, i) => i);
    const blks = blocks(idx, ext, hmed, texts, 0);
    const cy = ext.map(e => (e[2] + e[3]) / 2);
    const rx = ext.map(e => (e[0] + e[1]) / 2);
    const hs = boxes.map(b => Math.max(len2(b[0], b[3]), 1));
    const tol = 0.5 * hmed;
    const rows = [];
    for (const blk of blks) {
      const full = blk.filter(i => hs[i] >= 0.6 * hmed);
      const fullList = full.length ? full : [...blk];
      const fullSet = new Set(fullList);
      const small = blk.filter(i => !fullSet.has(i));
      const brows = [];
      for (const i of [...fullList].sort((p, q) => cy[p] - cy[q])) {
        if (brows.length) {
          const r = brows[brows.length - 1];
          const m = r.reduce((acc, j) => acc + cy[j], 0) / r.length;
          if (Math.abs(cy[i] - m) <= tol) r.push(i);
          else brows.push([i]);
        } else brows.push([i]);
      }
      const centre = brows.map(r => r.reduce((acc, j) => acc + cy[j], 0) / r.length);
      for (const i of small) {
        const d = centre.map(c => Math.abs(cy[i] - c));
        const minD = d.length ? Math.min(...d) : Infinity;
        if (d.length && minD <= tol + 0.5 * hs[i]) {
          let arg = 0;
          d.forEach((v, k) => { if (v === d[arg]) return; if (v < d[arg]) arg = k; });
          brows[arg].push(i);
        } else { brows.push([i]); centre.push(cy[i]); }
      }
      const order = brows.map((_, r) => r).sort((p, q) => centre[p] - centre[q]);
      for (const r of order) rows.push([...brows[r]].sort((p, q) => rx[p] - rx[q]));
    }
    return rows;
  }

  // Boxes that are pieces of a neighbouring full-height line.
  function fragments(boxes, recs) {
    const out = new Set();
    if (boxes.length < 3) return out;
    const hs = boxes.map(b => len2(b[0], b[3]));
    const hmed = lineHeight(boxes);
    for (let i = 0; i < boxes.length; i++) {
      if (hs[i] >= 0.35 * hmed) continue;
      if (recs[i].text.replace(/ /g, '').length > 3) continue;
      const b = boxes[i];
      const x0 = Math.min(...b.map(p => p[0])), x1 = Math.max(...b.map(p => p[0]));
      const y0 = Math.min(...b.map(p => p[1])), y1 = Math.max(...b.map(p => p[1]));
      for (let j = 0; j < boxes.length; j++) {
        if (j === i || hs[j] < 0.6 * hmed) continue;
        const c = boxes[j];
        const cx0 = Math.min(...c.map(p => p[0])), cx1 = Math.max(...c.map(p => p[0]));
        const cy0 = Math.min(...c.map(p => p[1])), cy1 = Math.max(...c.map(p => p[1]));
        if (cx0 <= x1 && x0 <= cx1 &&
            Math.min(Math.abs(y0 - cy1), Math.abs(cy0 - y1)) <= 0.5 * hmed) {
          out.add(i); break;
        }
      }
    }
    return out;
  }

  // ---- image ops -----------------------------------------------------------
  // Perspective-rectify one quad out of src (RGBA ImageData-like), bilinear,
  // borders replicated (cv2.warpPerspective BORDER_REPLICATE).
  function cropQuad(src, box, w, h) {
    const Hm = homography(box, [[0, 0], [w, 0], [w, h], [0, h]]);
    const Hi = invert3(Hm);
    const out = new Uint8ClampedArray(w * h * 4);
    const { data, width: sw, height: sh } = src;
    let o = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const den = Hi[6] * x + Hi[7] * y + Hi[8];
        const sx = (Hi[0] * x + Hi[1] * y + Hi[2]) / den;
        const sy = (Hi[3] * x + Hi[4] * y + Hi[5]) / den;
        const xi = Math.floor(sx), yi = Math.floor(sy);
        const fx = sx - xi, fy = sy - yi;
        const x0 = clamp(xi, 0, sw - 1), x1 = clamp(xi + 1, 0, sw - 1);
        const y0 = clamp(yi, 0, sh - 1), y1 = clamp(yi + 1, 0, sh - 1);
        const i00 = (y0 * sw + x0) * 4, i10 = (y0 * sw + x1) * 4;
        const i01 = (y1 * sw + x0) * 4, i11 = (y1 * sw + x1) * 4;
        out[o++] = data[i00] * (1 - fx) * (1 - fy) + data[i10] * fx * (1 - fy) +
                   data[i01] * (1 - fx) * fy + data[i11] * fx * fy;
        out[o++] = data[i00 + 1] * (1 - fx) * (1 - fy) + data[i10 + 1] * fx * (1 - fy) +
                   data[i01 + 1] * (1 - fx) * fy + data[i11 + 1] * fx * fy;
        out[o++] = data[i00 + 2] * (1 - fx) * (1 - fy) + data[i10 + 2] * fx * (1 - fy) +
                   data[i01 + 2] * (1 - fx) * fy + data[i11 + 2] * fx * fy;
        out[o++] = 255;
      }
    }
    return { data: out, width: w, height: h };
  }

  // np.rot90(img, k): k=1 counter-clockwise, k=-1 clockwise.
  // k=1: out(x, y) = in(w-1-y, x);  k=-1: out(x, y) = in(y, h-1-x)
  function rot90Crop(crop, k) {
    const { data, width: w, height: h } = crop;
    const W = h, H = w; // 90° rotation swaps the image dimensions
    const out = new Uint8ClampedArray(W * H * 4);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let xi, yi;
        if (k === 1) { xi = w - 1 - y; yi = x; }
        else { xi = y; yi = h - 1 - x; }
        const si = (yi * w + xi) * 4;
        const di = (y * W + x) * 4;
        out[di] = data[si]; out[di + 1] = data[si + 1];
        out[di + 2] = data[si + 2]; out[di + 3] = 255;
      }
    }
    return { data: out, width: W, height: H };
  }

  function rot180Crop(crop) {
    const { data, width: w, height: h } = crop;
    const out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const si = ((h - 1 - y) * w + (w - 1 - x)) * 4;
        const di = (y * w + x) * 4;
        out[di] = data[si]; out[di + 1] = data[si + 1];
        out[di + 2] = data[si + 2]; out[di + 3] = 255;
      }
    return { data: out, width: w, height: h };
  }

  function cropToGray(crop) {
    const { data } = crop;
    const n = crop.width * crop.height;
    const g = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * 4;
      g[i] = (data[p] * 0.299 + data[p + 1] * 0.587 + data[p + 2] * 0.114) | 0;
    }
    return g;
  }

  // 1D sliding median along x, borders reflected (cv2.medianBlur on the
  // 3-row repeat collapses to this).
  function median1D(arr, win, n) {
    const half = win >> 1;
    const out = new Float32Array(n);
    const buf = new Float32Array(win);
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let k = -half; k <= half; k++) {
        let j = i + k;
        if (j < 0) j = -j;           // BORDER_REFLECT_101
        if (j >= n) j = 2 * n - 2 - j;
        if (j < 0) j = 0;
        if (j >= n) j = n - 1;
        buf[m++] = arr[j];
      }
      const s = Array.from(buf.slice(0, m)).sort((a, b) => a - b);
      out[i] = s[m >> 1];
    }
    return out;
  }

  // line_engine._flatten: subtract each column's background (from the top and
  // bottom margins), then 255 - |g - bg| scaled to the 98th percentile.
  function flattenGray(g, w, h) {
    const k = Math.max(2, Math.floor(h / 6));
    const bg = new Float32Array(w);
    const vals = new Float32Array(2 * k);
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < k; y++) vals[y] = g[y * w + x];
      for (let y = h - k; y < h; y++) vals[k + (y - (h - k))] = g[y * w + x];
      const s = Array.from(vals).sort((a, b) => a - b);
      bg[x] = s.length % 2 ? s[(s.length - 1) >> 1]
                           : (s[s.length >> 1] + s[(s.length >> 1) - 1]) / 2;
    }
    const win = Math.max(3, Math.min(h | 1, 255));
    const bgSm = median1D(bg, win, w);
    const d = new Uint8Array(w * h);
    const hist = new Uint32Array(256);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const v = Math.min(255, Math.abs(g[y * w + x] - bgSm[x])) | 0;
        d[y * w + x] = v; hist[v]++;
      }
    let total = 0;
    for (let i = 0; i < 256; i++) total += hist[i];
    let c = 0, hi = 24;
    for (let i = 0; i < 256; i++) { c += hist[i]; if (c >= 0.98 * total) { hi = Math.max(i, 24); break; } }
    const out = new Uint8ClampedArray(w * h * 4);
    const scale = 255.0 / hi;
    for (let i = 0; i < w * h; i++) {
      const v = 255 - Math.min(255, d[i] * scale);
      const p = i * 4;
      out[p] = v; out[p + 1] = v; out[p + 2] = v; out[p + 3] = 255;
    }
    return { data: out, width: w, height: h };
  }

  function resizeCrop(crop, rw, rh) {
    if (crop.width === rw && crop.height === rh) return crop;
    const src = makeCanvas(crop.width, crop.height);
    src.getContext('2d').putImageData(
      new ImageData(new Uint8ClampedArray(crop.data), crop.width, crop.height), 0, 0);
    const dst = makeCanvas(rw, rh);
    const ctx = dst.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src, 0, 0, rw, rh);
    return getData(dst);
  }

  function deskewCanvas(src, angleDeg) {
    const W = src.width, H = src.height;
    const rad = angleDeg * Math.PI / 180;
    const c = Math.abs(Math.cos(rad)), s = Math.abs(Math.sin(rad));
    const nW = Math.round(H * s + W * c), nH = Math.round(H * c + W * s);
    const out = makeCanvas(nW, nH);
    const ctx = out.getContext('2d');
    // fill with the page's own border colour (median of the edge pixels)
    const d = getData(src).data;
    const samples = [];
    const step = Math.max(1, ((W + H) / 400) | 0);
    for (let x = 0; x < W; x += step) { samples.push(x, 0, x, H - 1); }
    for (let y = 0; y < H; y += step) { samples.push(0, y, W - 1, y); }
    const rs = [], gs = [], bs = [];
    for (const q of samples) {
      const i = (q[1] * W + q[0]) * 4;
      rs.push(d[i]); gs.push(d[i + 1]); bs.push(d[i + 2]);
    }
    ctx.fillStyle = `rgb(${Math.round(median(rs))},${Math.round(median(gs))},${Math.round(median(bs))})`;
    ctx.fillRect(0, 0, nW, nH);
    ctx.translate(nW / 2, nH / 2);
    ctx.rotate(-rad); // cv2 positive angle = counter-clockwise
    ctx.translate(-W / 2, -H / 2);
    ctx.drawImage(src, 0, 0);
    return out;
  }

  function rot90Canvas(src, k) {
    const W = src.width, H = src.height;
    const out = makeCanvas(H, W); // dims swap
    const ctx = out.getContext('2d');
    if (k === 1) { // CCW: x_d = y_s, y_d = W_s - x_s
      ctx.setTransform(0, -1, 1, 0, 0, W);
    } else {       // CW: x_d = H_s - y_s, y_d = x_s
      ctx.setTransform(0, 1, -1, 0, H, 0);
    }
    ctx.drawImage(src, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return out;
  }

  function rot180Canvas(src) {
    const W = src.width, H = src.height;
    const out = makeCanvas(W, H);
    const ctx = out.getContext('2d');
    ctx.setTransform(-1, 0, 0, -1, W, H);
    ctx.drawImage(src, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return out;
  }

  // ---- Thai text canonicalisation (thai_norm.py port) ----------------------
  const PUA_MAP = {
    0xF700: 0x0E10, 0xF701: 0x0E34, 0xF702: 0x0E35, 0xF703: 0x0E36, 0xF704: 0x0E37,
    0xF705: 0x0E48, 0xF706: 0x0E49, 0xF707: 0x0E4A, 0xF708: 0x0E4B, 0xF709: 0x0E4C,
    0xF70A: 0x0E48, 0xF70B: 0x0E49, 0xF70C: 0x0E4A, 0xF70D: 0x0E4B, 0xF70E: 0x0E4C,
    0xF70F: 0x0E0D, 0xF710: 0x0E31, 0xF711: 0x0E4D, 0xF712: 0x0E47, 0xF713: 0x0E48,
    0xF714: 0x0E49, 0xF715: 0x0E4A, 0xF716: 0x0E4B, 0xF717: 0x0E4C, 0xF718: 0x0E38,
    0xF719: 0x0E39, 0xF71A: 0x0E3A,
  };
  const DROP_CP = new Set([0x200B, 0x200C, 0x200D, 0x2060, 0x00AD, 0xFEFF]);
  const VOWEL_SET = new Set([...'ัิีึืฺุู็']);
  const TONE_SET = new Set([...'่้๊๋์๎']);
  const MARK_RUN_RE = /[ัิีึืฺุู็่้๊๋์๎ํ]{2,}/g;
  const SARA_AM_RE = /([่-๋]?)ํ([่-๋]?)า/g;

  function canonical(text) {
    if (!text) return '';
    let s = '';
    for (const ch of text) {
      let cp = ch.codePointAt(0);
      if (PUA_MAP[cp] !== undefined) cp = PUA_MAP[cp];
      else if (DROP_CP.has(cp)) continue;
      else if (cp === 0x00A0) cp = 0x20;
      s += String.fromCodePoint(cp);
    }
    s = s.normalize('NFC');
    s = s.replace(MARK_RUN_RE, run => {
      const rank = c => VOWEL_SET.has(c) ? 0 : (TONE_SET.has(c) ? 1 : 2);
      const arr = [...run].sort((a, b) => rank(a) - rank(b));
      const res = [];
      for (const c of arr) if (!res.length || res[res.length - 1] !== c) res.push(c);
      return res.join('');
    });
    s = s.replace(SARA_AM_RE, (m, a, b) => a + b + 'ำ');
    s = s.split('เเ').join('แ').split('ำำ').join('ำ');
    s = s.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    return s.split('\n').map(l => l.replace(/[ \t\f\v]+/g, ' ').trim())
      .join('\n').trim();
  }

  function fixScripts(text) {
    let t = text.replace(/(?<![A-Za-z])[Ww](?=\.?[\u0E01-\u0E2E\u0E40-\u0E44])/g, 'พ');
    t = t.replace(/(?<=[๐-๙])[Oo0]|[Oo0](?=[๐-๙])/g, '๐');
    return t;
  }

  const THAI_MARKS = new Set([...'ัิีึืฺุู็่้๊๋์ํ๎']);
  const THAI_LEADING = new Set([...'เแโใไ']);
  const isCons = ch => ch >= '\u0E01' && ch <= '\u0E2E';

  function invalidShare(text) {
    let thai = 0, bad = 0;
    const arr = [...text];
    for (let i = 0; i < arr.length; i++) {
      const ch = arr[i];
      if (!(ch >= '\u0E00' && ch <= '\u0E7F')) continue;
      thai++;
      const prev = i ? arr[i - 1] : ' ';
      if (THAI_MARKS.has(ch)) {
        const ok = isCons(prev) ||
          (THAI_MARKS.has(prev) && i >= 2 && isCons(arr[i - 2]));
        if (!ok) bad++;
      } else if (THAI_LEADING.has(ch)) {
        if (!(i + 1 < arr.length && isCons(arr[i + 1]))) bad++;
      }
    }
    return thai ? bad / thai : 0;
  }

  // ---- CTC -----------------------------------------------------------------
  // Greedy CTC: argmax per timestep, collapse repeats, drop blank (id 0).
  function ctcDecode(probs, T, C, chars) {
    const ids = new Int32Array(T), tops = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      const base = t * C;
      let best = 0, bv = -1;
      for (let c = 0; c < C; c++) { const v = probs[base + c]; if (v > bv) { bv = v; best = c; } }
      ids[t] = best; tops[t] = bv;
    }
    const charsOut = [], cs = [];
    let t = 0;
    while (t < T) {
      let end = t + 1;
      while (end < T && ids[end] === ids[t]) end++;
      if (ids[t] !== 0) {
        charsOut.push(ids[t] < chars.length ? chars[ids[t]] : '');
        cs.push(tops[t]);
      }
      t = end;
    }
    return {
      text: charsOut.join(''),
      conf: cs.length ? mean(cs) : 0,
      chars: null,
    };
  }

  // ---- engine ---------------------------------------------------------------
  class OcrEngine {
    constructor(base = '') {
      this.base = base;
      this.chars = null;
      this.detSession = null;
      this.recSession = null;
    }

    async load(onStatus) {
      onStatus && onStatus('dict');
      const dictText = await (await fetch(this.base + 'models/rec_dict.txt')).text();
      const dict = dictText.split('\n').map(s => s.replace(/\r$/, '')).filter(s => s.length > 0);
      this.chars = ['', ...dict, ' '];
      try {
        const cfg = await (await fetch(this.base + 'models/config.json')).json();
        if (cfg.rec) {
          if (typeof cfg.rec.drop_below === 'number') REC.dropBelow = cfg.rec.drop_below;
          if (typeof cfg.rec.min_chars === 'number') REC.minChars = cfg.rec.min_chars;
          if (typeof cfg.rec.junk_below === 'number') REC.junkBelow = cfg.rec.junk_below;
          if (typeof cfg.rec.junk_invalid === 'number') REC.junkInvalid = cfg.rec.junk_invalid;
        }
      } catch (e) { /* shipped defaults already match */ }

      onStatus && onStatus('det');
      this.detSession = await ort.InferenceSession.create(this.base + 'models/det.onnx',
        { executionProviders: ['wasm'] });
      onStatus && onStatus('rec');
      this.recSession = await ort.InferenceSession.create(this.base + 'models/rec.onnx',
        { executionProviders: ['wasm'] });
      this.detInput = this.detSession.inputNames[0];
      this.recInput = this.recSession.inputNames[0];
    }

    get ready() { return !!(this.detSession && this.recSession && this.chars); }

    // -- detection ------------------------------------------------------------
    async detect(src) {
      const H0 = src.height, W0 = src.width;
      const pad = Math.max(8, Math.round(0.05 * Math.min(H0, W0)));
      const pc = makeCanvas(W0 + 2 * pad, H0 + 2 * pad);
      const pctx = pc.getContext('2d');
      pctx.fillStyle = '#ffffff';
      pctx.fillRect(0, 0, pc.width, pc.height);
      pctx.putImageData(new ImageData(new Uint8ClampedArray(src.data), W0, H0), pad, pad);
      const h = pc.height, w = pc.width;
      const s = Math.min(DET.limit / Math.max(h, w), 4.0);
      const rh = Math.max(32, Math.round(h * s / 32) * 32);
      const rw = Math.max(32, Math.round(w * s / 32) * 32);
      const rc = makeCanvas(rw, rh);
      const rctx = rc.getContext('2d', { willReadFrequently: true });
      rctx.imageSmoothingEnabled = true;
      rctx.imageSmoothingQuality = 'high';
      rctx.drawImage(pc, 0, 0, rw, rh);
      const px = rctx.getImageData(0, 0, rw, rh);
      const n = rw * rh;
      const x = new Float32Array(3 * n);
      const d = px.data;
      for (let i = 0, p = 0; i < n; i++, p += 4) {
        const b = d[p + 2] / 255, g = d[p + 1] / 255, r = d[p] / 255;
        x[i] = (b - DET_MEAN[0]) / DET_STD[0];
        x[n + i] = (g - DET_MEAN[1]) / DET_STD[1];
        x[2 * n + i] = (r - DET_MEAN[2]) / DET_STD[2];
      }
      const results = await this.detSession.run(
        { [this.detInput]: new ort.Tensor('float32', x, [1, 3, rh, rw]) });
      const probT = results[this.detSession.outputNames[0]];
      const ph = probT.dims[2], pw = probT.dims[3];
      const prob = probT.data;
      const bitmap = new Uint8Array(ph * pw);
      for (let i = 0; i < prob.length; i++) bitmap[i] = prob[i] > DET.thresh ? 1 : 0;
      const comps = connectedComponents(bitmap, pw, ph);
      const sx = w / pw, sy = h / ph;
      const boxes = [];
      for (const comp of comps) {
        if (boxes.length >= DET.maxCandidates) break;
        const { box, side } = minAreaRect(comp);
        if (side < DET.minSize) continue;
        if (boxScore(prob, pw, ph, box) < DET.boxThresh) continue;
        const big = unclipBox(box, DET.unclip);
        if (!big) continue;
        const r2 = minAreaRect(big.map(p => ({ x: p[0], y: p[1] })));
        if (r2.side < DET.minSize + 2) continue;
        const qb = r2.box.map(pt => [
          clamp(pt[0] * sx - pad, 0, W0 - 1),
          clamp(pt[1] * sy - pad, 0, H0 - 1),
        ]);
        const bw = Math.max(...qb.map(p => p[0])) - Math.min(...qb.map(p => p[0]));
        const bh = Math.max(...qb.map(p => p[1])) - Math.min(...qb.map(p => p[1]));
        if (bw >= 2 && bh >= 2) boxes.push(qb);
      }
      return boxes;
    }

    // -- recognition ----------------------------------------------------------
    async recognizeDetail(crops, onBatch) {
      const H = REC.height;
      const flat = [];
      for (const cr of crops) {
        const rw = Math.max(1, Math.ceil(H * cr.width / Math.max(cr.height, 1)));
        flat.push(flattenGray(cropToGray(resizeCrop(cr, rw, H)), rw, H));
      }
      const ratios = flat.map(c => c.width / Math.max(c.height, 1));
      const out = new Array(crops.length);
      const order = ratios.map((r, i) => [r, i]).sort((a, b) => a[0] - b[0]).map(p => p[1]);
      for (let b0 = 0; b0 < order.length; b0 += REC.batch) {
        const idx = order.slice(b0, b0 + REC.batch);
        const maxRatio = Math.max(...idx.map(i => ratios[i]));
        const W = Math.min(REC.maxW, Math.max(16, Math.ceil(H * maxRatio)));
        const N = idx.length;
        const batch = new Float32Array(N * 3 * H * W);
        idx.forEach((i, nn) => {
            const rw = Math.min(W, Math.max(1, Math.ceil(H * ratios[i])));
            const im = resizeCrop(flat[i], rw, H);
          for (let y = 0; y < H; y++) {
            const rowBase = y * rw * 4;
            const outBase = (nn * 3 * H + y) * W;
            for (let xx = 0; xx < rw; xx++) {
              const v = (im.data[rowBase + xx * 4] / 255 - 0.5) / 0.5;
              batch[outBase + xx] = v;
              batch[W * H + outBase + xx] = v;
              batch[2 * W * H + outBase + xx] = v;
            }
          }
        });
        const results = await this.recSession.run(
          { [this.recInput]: new ort.Tensor('float32', batch, [N, 3, H, W]) });
        const t = results[this.recSession.outputNames[0]];
        const T = t.dims[1], C = t.dims[2];
        idx.forEach((i, nn) => {
          out[i] = ctcDecode(t.data.subarray(nn * T * C, (nn + 1) * T * C),
            T, C, this.chars);
        });
        onBatch && onBatch(Math.min(b0 + REC.batch, order.length), order.length);
        if (onBatch) await new Promise(r => setTimeout(r, 0));
      }
      return out;
    }

    // A box taller than wide is read as cut and turned both ways; the most
    // confident reading wins (line_engine._read_boxes).
    async readBoxes(src, boxes, onBatch) {
      const raw = boxes.map(b => cropQuad(src, b,
        Math.max(1, Math.round(Math.max(len2(b[0], b[1]), len2(b[2], b[3])))),
        Math.max(1, Math.round(Math.max(len2(b[0], b[3]), len2(b[1], b[2]))))));
      const tall = [];
      raw.forEach((c, nn) => { if (c.height >= 1.5 * c.width) tall.push(nn); });
      const all = [...raw];
      for (const nn of tall) { all.push(rot90Crop(raw[nn], 1)); all.push(rot90Crop(raw[nn], -1)); }
      const res = await this.recognizeDetail(all, onBatch);
      const out = res.slice(0, raw.length);
      tall.forEach((nn, j) => {
        let best = out[nn];
        for (const alt of [res[raw.length + j], res[raw.length + tall.length + j]])
          if (alt.conf > best.conf) best = { text: alt.text, conf: alt.conf, chars: null };
        out[nn] = best;
      });
      return out;
    }

    // line_engine.split_one: cut a line box at blank stretches >= ratio*height.
    splitOne(src, b, ratio) {
      if (!ratio) return { pieces: [b], changed: false };
      const W = Math.max(1, Math.round(Math.max(len2(b[0], b[1]), len2(b[2], b[3]))));
      const H = Math.max(1, Math.round(Math.max(len2(b[0], b[3]), len2(b[1], b[2]))));
      if (W < 4 * H || H < 4) return { pieces: [b], changed: false };
      const c = cropQuad(src, b, W, H);
      const fl = flattenGray(cropToGray(c), W, H);
      const ink = new Uint8Array(W);
      const fd = fl.data;
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (fd[(y * W + x) * 4] < 128) ink[x] = 1;
      const segs = [];
      let start = null, blank = 0;
      for (let x = 0; x < W; x++) {
        if (ink[x]) {
          if (start === null) start = x;
          else if (blank >= ratio * H) { segs.push([start, x - blank]); start = x; }
          blank = 0;
        } else if (start !== null) blank++;
      }
      if (start !== null) segs.push([start, W - blank]);
      if (!segs.length ||
          (segs.length === 1 && segs[0][0] < ratio * H && W - segs[0][1] < ratio * H))
        return { pieces: [b], changed: false };
      const top = [b[1][0] - b[0][0], b[1][1] - b[0][1]];
      const bot = [b[2][0] - b[3][0], b[2][1] - b[3][1]];
      const pad = 0.15 * H;
      const pieces = segs.map(([x0, x1]) => {
        const t0 = Math.max(0, (x0 - pad) / W), t1 = Math.min(1, (x1 + pad) / W);
        return [
          [b[0][0] + t0 * top[0], b[0][1] + t0 * top[1]],
          [b[0][0] + t1 * top[0], b[0][1] + t1 * top[1]],
          [b[3][0] + t1 * bot[0], b[3][1] + t1 * bot[1]],
          [b[3][0] + t0 * bot[0], b[3][1] + t0 * bot[1]],
        ];
      });
      return { pieces, changed: true };
    }

    // line_engine._page: split, read, keep pieces if they read as well.
    async pageFromBoxes(src, boxes, onBatch) {
      const groups = boxes.map(b => this.splitOne(src, b, DET.splitGap));
      const pieces = [], wholeBoxes = [];
      groups.forEach((g, i) => {
        if (g.changed) wholeBoxes.push(boxes[i]);
        pieces.push(...g.pieces);
      });
      const res = await this.readBoxes(src, [...pieces, ...wholeBoxes], onBatch);
      const final = [], out = [];
      let k = 0, e = pieces.length;
      groups.forEach((g, i) => {
        const r = res.slice(k, k + g.pieces.length);
        k += g.pieces.length;
        if (g.changed) {
          const alt = res[e]; e++;
          const nPieces = r.reduce((a, x) => a + x.text.replace(/ /g, '').length, 0);
          const minConf = r.length ? Math.min(...r.map(x => x.conf)) : 0;
          if (minConf < REC.dropBelow ||
              nPieces < 0.9 * alt.text.replace(/ /g, '').length) {
            final.push(boxes[i]); out.push(alt);
            return;
          }
        }
        final.push(...g.pieces); out.push(...r);
      });
      const angle = pageSkew(final);
      const rows = readingOrder(final, angle, out.map(x => x.text));
      return { rows, angle, rec: out, boxes: final };
    }

    // line_engine._sideways
    async sideways(src, boxes) {
      const votes = boxes.map(b => verticality(b, 2.0));
      const vert = boxes.filter((b, i) => votes[i] === 1);
      const nMinus = votes.filter(v => v === -1).length;
      if (vert.length < 3 || vert.length <= nMinus) return 0;
      vert.sort((a, b) => len2(b[0], b[3]) - len2(a[0], a[3]));
      const sample = vert.slice(0, ORI.sample);
      const raw = sample.map(b => cropQuad(src, b,
        Math.max(1, Math.round(Math.max(len2(b[0], b[1]), len2(b[2], b[3])))),
        Math.max(1, Math.round(Math.max(len2(b[0], b[3]), len2(b[1], b[2]))))));
      const rots = [];
      for (const kk of [1, -1]) for (const c of raw) rots.push(rot90Crop(c, kk));
      const res = await this.recognizeDetail(rots);
      const ccw = mean(res.slice(0, raw.length).map(r => r.conf));
      const cw = mean(res.slice(raw.length).map(r => r.conf));
      return ccw > cw ? 1 : -1;
    }

    // line_engine._upside_down
    async upsideDown(src, page) {
      const confs = page.rec.filter(r => r.text.length >= 2).map(r => r.conf);
      if (confs.length < 3 || median(confs) >= ORI.flipBelow) return false;
      const idx = page.boxes.map((b, i) => i)
        .filter(i => verticality(page.boxes[i], 2.0) === -1)
        .sort((i, j) => len2(page.boxes[j][0], page.boxes[j][1]) -
                        len2(page.boxes[i][0], page.boxes[i][1]))
        .slice(0, ORI.sample);
      if (idx.length < 3) return false;
      const up = mean(idx.map(i => page.rec[i].conf));
      const crops = idx.map(i => {
        const b = page.boxes[i];
        return rot180Crop(cropQuad(src, b,
          Math.max(1, Math.round(Math.max(len2(b[0], b[1]), len2(b[2], b[3])))),
          Math.max(1, Math.round(Math.max(len2(b[0], b[3]), len2(b[1], b[2]))))));
      });
      const res = await this.recognizeDetail(crops);
      return mean(res.map(r => r.conf)) > up + ORI.flipMargin;
    }

    // Apply the line filter + reading order to a finished page.
    assemble(page, filter) {
      const frag = fragments(page.boxes, page.rec);
      const lines = [], dropped = [], rowTexts = [];
      page.rows.forEach((r, ri) => {
        const parts = [];
        for (const i of r) {
          const raw = page.rec[i];
          const text = fixScripts(canonical(raw.text));
          const conf = raw.conf;
          const entry = { text, conf: Math.round(conf * 10000) / 10000, row: ri, box: page.boxes[i] };
          let reason = null;
          if (frag.has(i)) continue;
          if (!text) reason = 'empty';
          else if (filter && conf < REC.dropBelow) reason = 'low_confidence';
          else if (filter && text.replace(/ /g, '').length < REC.minChars) reason = 'too_short';
          else if (filter && conf < REC.junkBelow && invalidShare(text) > REC.junkInvalid) reason = 'invalid_thai';
          if (reason) { dropped.push({ ...entry, reason }); continue; }
          parts.push(text);
          lines.push(entry);
        }
        if (parts.length) rowTexts.push(parts.join(' '));
      });
      return { lines, dropped, rowTexts };
    }

    // line_engine.read
    async read(sourceCanvas, opts = {}) {
      const autoOrient = opts.autoOrient !== false;
      const onBatch = opts.onBatch;
      const onStatus = opts.onStatus;
      let work = capCanvas(sourceCanvas, 4096);
      const t0 = performance.now();
      let imgData = getData(work);
      onStatus && onStatus('detect');
      let boxes = await this.detect(imgData);
      let h = work.height, w = work.width;
      if (!boxes.length && h <= 200 && w >= 1.5 * h)
        boxes = [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]];
      let rotation = 0;
      if (autoOrient) {
        onStatus && onStatus('orientation');
        const kk = await this.sideways(imgData, boxes);
        if (kk) {
          work = rot90Canvas(work, kk);
          imgData = getData(work);
          boxes = await this.detect(imgData);
          rotation = kk === -1 ? 90 : 270;
          h = work.height; w = work.width;
        }
        const skew = pageSkew(boxes);
        if (Math.abs(skew) >= ORI.deskewAbove) {
          work = deskewCanvas(work, skew);
          imgData = getData(work);
          boxes = await this.detect(imgData);
          h = work.height; w = work.width;
        }
      }
      const t1 = performance.now();
      onStatus && onStatus('recognize');
      let page = await this.pageFromBoxes(imgData, boxes, onBatch);
      if (autoOrient && !rotation) {
        onStatus && onStatus('flip-check');
        if (await this.upsideDown(imgData, page)) {
          work = rot180Canvas(work);
          imgData = getData(work);
          boxes = await this.detect(imgData);
          rotation = 180;
          page = await this.pageFromBoxes(imgData, boxes, onBatch);
        }
      }
      const t2 = performance.now();

      const filter = opts.filter !== false;
      let assembled = this.assemble(page, filter);
      // Safety net for upside-down images with too few lines for the
      // statistical flip heuristic: if nothing survived the filter, try the
      // image rotated 180° and keep whichever reads better.
      if (autoOrient && !rotation && !assembled.lines.length && assembled.dropped.length) {
        onStatus && onStatus('flip-retry');
        const flipped = rot180Canvas(work);
        const img2 = getData(flipped);
        const page2 = await this.pageFromBoxes(img2, await this.detect(img2), onBatch);
        const assembled2 = this.assemble(page2, filter);
        if (assembled2.lines.length > assembled.lines.length) {
          work = flipped;
          rotation = 180;
          page = page2;
          assembled = assembled2;
        }
      }
      const { lines, dropped, rowTexts } = assembled;
      return {
        text: rowTexts.join('\n'),
        lines, dropped, rotation,
        working: work,
        boxes: page.boxes,
        timing: {
          det: Math.round(t1 - t0),
          rec: Math.round(t2 - t1),
          n: page.rec.length,
        },
      };
    }
  }

  window.OcrEngine = OcrEngine;
})();
