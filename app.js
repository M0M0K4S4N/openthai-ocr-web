/* UI wiring for OpenThai2.0-OCR in-browser demo. */
(function () {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const dropzone = $('#dropzone');
  const fileInput = $('#file');
  const btnSample = $('#btnSample');
  const btnCopy = $('#btnCopy');
  const btnClear = $('#btnClear');
  const statusEl = $('#status');
  const previewWrap = $('#previewWrap');
  const preview = $('#preview');
  const pctx = preview.getContext('2d');
  const resultWrap = $('#resultWrap');
  const out = $('#out');
  const statsEl = $('#stats');

  const ORT_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
  const MODELS = [
    { dir: 'models/', label: 'OpenThai2.0-OCR-Text (v2.0.0)' },
    { dir: 'models21/', label: 'OpenThai2.1-OCR-Text (v2.1.4)' },
  ];
  const modelSel = $('#modelSel');
  const modelCache = {}; // per-model-dir fetched dict/cfg/onnx buffers
  let engine = null;
  let busy = false;

  for (const m of MODELS) {
    const o = document.createElement('option');
    o.value = m.dir;
    o.textContent = m.label;
    modelSel.appendChild(o);
  }

  function setStatus(msg, cls) {
    statusEl.textContent = msg;
    statusEl.className = 'status' + (cls ? ' ' + cls : '');
  }

  function setBusy(b) {
    busy = b;
    [btnSample, btnCopy, btnClear].forEach(el => el.disabled = b);
    fileInput.disabled = b;
    modelSel.disabled = b;
  }

  async function ensureEngine() {
    const dir = modelSel.value;
    const label = (MODELS.find(m => m.dir === dir) || {}).label || dir;
    if (engine && engine.ready && engine.base === dir) return engine;
    if (!window.ort) throw new Error('โหลด onnxruntime-web จาก CDN ไม่สำเร็จ (ต้องมีอินเทอร์เน็ตครั้งแรก)');
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.numThreads = 1; // no COOP/COEP headers on static hosting
    const files = (modelCache[dir] = modelCache[dir] || {});
    const eng = new window.OcrEngine(dir);
    setStatus(`กำลังโหลดโมเดล ${label} ...`);
    await eng.load((what) => {
      const names = { dict: 'dictionary', det: 'detector', rec: 'recognizer', rec2: 'recognizer #2 (loopless)' };
      setStatus(`กำลังโหลด ${names[what] || what} — ${label} ...`);
    }, files);
    engine = eng;
    setStatus(`โมเดลพร้อมแล้ว ✓ (${label})`, 'ok');
    return engine;
  }

  modelSel.addEventListener('change', () => {
    if (!busy) setStatus('เปลี่ยนโมเดลแล้ว — จะโหลดอัตโนมัติเมื่อใช้งานครั้งถัดไป');
  });

  // ---------- image intake ----------
  function fileToCanvas(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        c.getContext('2d').drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        resolve(c);
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('อ่านไฟล์รูปไม่สำเร็จ')); };
      img.src = url;
    });
  }

  async function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) { setStatus('ไฟล์ไม่ใช่รูปภาพ', 'err'); return; }
    try {
      const canvas = await fileToCanvas(file);
      await runOcr(canvas);
    } catch (e) {
      console.error(e);
      setStatus('ผิดพลาด: ' + e.message, 'err');
    }
  }

  dropzone.addEventListener('click', () => !busy && fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
    fileInput.value = '';
  });
  ['dragenter', 'dragover'].forEach(ev =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach(ev =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove('drag'); }));
  dropzone.addEventListener('drop', (e) => {
    if (busy) return;
    const f = [...(e.dataTransfer?.files || [])].find(f => f.type.startsWith('image/'));
    if (f) handleFile(f);
  });
  document.addEventListener('paste', (e) => {
    if (busy) return;
    const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
    if (item) handleFile(item.getAsFile());
  });

  // ---------- sample image (drawn locally, no download) ----------
  function makeSample() {
    const W = 1680, H = 420;
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#111111';
    ctx.textBaseline = 'top';
    ctx.font = '600 44px "Thonburi", "Sukhumvit Set", "Sarabun", sans-serif';
    ctx.fillText('สวัสดีชาวโลก ทดสอบ OpenThai OCR ในเบราว์เซอร์', 60, 60);
    ctx.font = '400 36px "Thonburi", "Sukhumvit Set", "Sarabun", sans-serif';
    ctx.fillText('onnxruntime-web (WASM) ประมวลผลบนเครื่องผู้ใช้ทั้งหมด', 60, 150);
    ctx.fillText('๑๒๓๔๕๖๗๘๙๐ — จำนวนเงิน 1,234,567.89 บาท (รวมภาษี)', 60, 230);
    ctx.fillText('The quick brown fox jumps over the lazy dog 0123456789', 60, 310);
    return c;
  }

  btnSample.addEventListener('click', async () => {
    if (busy) return;
    await runOcr(makeSample());
  });

  // ---------- OCR run ----------
  function showPreview(canvas) {
    previewWrap.classList.remove('hidden');
    preview.width = canvas.width;
    preview.height = canvas.height;
    pctx.drawImage(canvas, 0, 0);
  }

  function drawBoxes(lines, dropped) {
    pctx.save();
    pctx.lineWidth = Math.max(1.5, preview.width / 700);
    pctx.font = '600 16px sans-serif';
    for (const l of lines) {
      pctx.strokeStyle = 'rgba(52,211,153,.9)';
      pctx.beginPath();
      l.box.forEach(([x, y], i) => i ? pctx.lineTo(x, y) : pctx.moveTo(x, y));
      pctx.closePath();
      pctx.stroke();
    }
    pctx.strokeStyle = 'rgba(248,113,113,.9)';
    for (const l of dropped) {
      pctx.beginPath();
      l.box.forEach(([x, y], i) => i ? pctx.lineTo(x, y) : pctx.moveTo(x, y));
      pctx.closePath();
      pctx.stroke();
    }
    pctx.restore();
  }

  async function runOcr(canvas) {
    if (busy) return;
    setBusy(true);
    try {
      const eng = await ensureEngine();
      showPreview(canvas);
      setStatus('กำลัง OCR ...');
      const res = await eng.read(canvas, {
        autoOrient: $('#optOrient').checked,
        filter: $('#optFilter').checked,
        onStatus: (what) => {
          const names = {
            detect: 'กำลัง detect กล่องข้อความ',
            orientation: 'กำลังเช็คการวางแนวภาพ',
            recognize: 'กำลังอ่านตัวอักษร',
            'flip-check': 'กำลังเช็คภาพกลับหัว',
          };
          setStatus(names[what] || what + ' ...');
        },
        onBatch: (done, total) => setStatus(`กำลังอ่านตัวอักษร (${done}/${total} ชิ้น) ...`),
      });
      showPreview(res.working);
      drawBoxes(res.lines, res.dropped);
      out.value = res.text;
      resultWrap.classList.remove('hidden');
      const avgConf = res.lines.length
        ? (res.lines.reduce((a, l) => a + l.conf, 0) / res.lines.length).toFixed(3) : '—';
      const label = (MODELS.find(m => m.dir === engine.base) || {}).label || engine.base;
      statsEl.innerHTML =
        `<span>โมเดล <b>${label}</b></span>` +
        `<span>บรรทัดที่อ่านได้ <b>${res.lines.length}</b></span>` +
        `<span>conf เฉลี่ย <b>${avgConf}</b></span>` +
        `<span>กรองทิ้ง <b>${res.dropped.length}</b></span>` +
        `<span>det <b>${res.timing.det}ms</b> · rec <b>${res.timing.rec}ms</b></span>` +
        (res.rotation ? `<span>หมุนภาพ <b>${res.rotation}°</b></span>` : '');
      setStatus('เสร็จแล้ว ✓', 'ok');
    } catch (e) {
      console.error(e);
      setStatus('ผิดพลาด: ' + (e && e.message ? e.message : e), 'err');
    } finally {
      setBusy(false);
    }
  }

  btnCopy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(out.value);
      btnCopy.textContent = '✓ คัดลอกแล้ว';
      setTimeout(() => btnCopy.textContent = '📋 คัดลอก', 1200);
    } catch (e) { out.select(); document.execCommand('copy'); }
  });

  btnClear.addEventListener('click', () => {
    out.value = '';
    resultWrap.classList.add('hidden');
    previewWrap.classList.add('hidden');
    setStatus('ยังไม่ได้เลือกรูป');
  });

  // test hook (also handy in devtools): window.__ocrRun(canvas)
  window.__ocrRun = runOcr;
  window.__getEngine = () => engine;
})();
