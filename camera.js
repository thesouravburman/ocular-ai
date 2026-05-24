/* ============================================================
   camera.js — OcularAI Live Camera Module
   Dependencies (loaded in index.html before this file):
     @mediapipe/face_mesh  — face landmark detection
     @mediapipe/camera_utils — wraps getUserMedia cleanly
     mediapipe_worker.js   — window.OcularUtils helpers
   ============================================================ */

'use strict';

/* ── CONSTANTS ──────────────────────────────────────────────── */
const IPD_REAL_MM      = 63;   // average adult IPD in millimetres
const FOCAL_LENGTH_PX  = 650;  // estimated focal length for phone front camera
const FRAME_BUFFER_MAX = 20;   // rolling-average window size (Issue #3)

// MediaPipe 478-landmark model iris indices
// (only available when refineLandmarks: true is set)
const IDX_L_IRIS = 468;  // left iris centre
const IDX_R_IRIS = 473;  // right iris centre

// Left and right iris ring indices (for drawing the circles)
const L_RING = [468, 469, 470, 471, 472];
const R_RING = [473, 474, 475, 476, 477];

/* ── STATE ──────────────────────────────────────────────────── */
let faceMeshInst  = null;   // MediaPipe FaceMesh instance
let cameraInst    = null;   // MediaPipe Camera helper instance
let frameBuffer   = [];     // rolling IPD values (max 20)
let camRunning    = false;

/* ── DOM REFS (populated on DOMContentLoaded) ───────────────── */
let videoEl, canvasEl, canvasCtx;
let elDist, elDiopt, elIPD;
let elStatus, elCounter, elBar, elSeverity;
let elStartBtn, elStopBtn;

/* ── INIT ────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  videoEl  = document.getElementById('camera-video');
  canvasEl = document.getElementById('camera-canvas-overlay');
  canvasCtx = canvasEl.getContext('2d');

  // Match canvas resolution to the video container
  canvasEl.width  = 480;
  canvasEl.height = 360;

  elDist     = document.getElementById('live-distance');
  elDiopt    = document.getElementById('live-diopter');
  elIPD      = document.getElementById('live-ipd');
  elStatus   = document.getElementById('frame-status');
  elCounter  = document.getElementById('frame-counter');
  elBar      = document.getElementById('frame-bar');
  elSeverity = document.getElementById('live-severity');
  elStartBtn = document.getElementById('start-cam-btn');
  elStopBtn  = document.getElementById('stop-cam-btn');
});

/* ── CALIBRATION (Issue #5) ─────────────────────────────────── */
// showCalibration() is called from index.html buttons
// confirmCalibration() is called from the overlay confirm button

window.showCalibration = function () {
  const overlay = document.getElementById('calibration-overlay');
  overlay.style.display = 'flex';

  const slider  = document.getElementById('calib-slider');
  const rect    = document.getElementById('card-rect');
  const label   = document.getElementById('calib-px-label');

  // Restore previous calibration value if available
  const stored = localStorage.getItem('ocular_card_px');
  if (stored) slider.value = stored;

  function sync() {
    rect.style.width  = slider.value + 'px';
    label.textContent = slider.value + ' px';
  }
  sync();
  slider.addEventListener('input', sync);
};

window.confirmCalibration = function () {
  const sliderVal   = parseInt(document.getElementById('calib-slider').value, 10);
  const pxPerMm     = sliderVal / 85.6;  // 85.6 mm = ISO 7810 credit card width

  localStorage.setItem('ocular_calibrated',   'true');
  localStorage.setItem('ocular_pixels_per_mm', pxPerMm.toFixed(5));
  localStorage.setItem('ocular_card_px',       sliderVal);

  document.getElementById('calibration-overlay').style.display = 'none';
  console.log('[OcularAI] Calibration saved:', pxPerMm.toFixed(4), 'px/mm');
};

/* ── START CAMERA ────────────────────────────────────────────── */
window.startCamera = async function () {
  if (camRunning) return;

  // First-time users must calibrate before measurement
  if (!localStorage.getItem('ocular_calibrated')) {
    showCalibration();
    return;
  }

  elStartBtn.style.display = 'none';
  elStopBtn.style.display  = 'inline-flex';
  setStatus('Initialising face detection…');

  try {
    /* Build FaceMesh
       locateFile tells MediaPipe where to download its WASM files from.
       WASM = WebAssembly, a fast binary format that runs ML models in the browser. */
    faceMeshInst = new FaceMesh({
      locateFile: (file) =>
        'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/' + file
    });

    faceMeshInst.setOptions({
      maxNumFaces:            1,
      refineLandmarks:        true,  // REQUIRED for iris landmarks 468-477
      minDetectionConfidence: 0.55,
      minTrackingConfidence:  0.50
    });

    faceMeshInst.onResults(onResults);

    /* Camera helper wraps getUserMedia (the browser API that opens the camera)
       and feeds each video frame into FaceMesh automatically. */
    cameraInst = new Camera(videoEl, {
      onFrame: async () => {
        await faceMeshInst.send({ image: videoEl });
      },
      width:       640,
      height:      480,
      facingMode: 'user'  // front camera
    });

    await cameraInst.start();
    camRunning = true;
    setStatus('Look at the camera — searching for face…');

  } catch (err) {
    console.error('[OcularAI] Camera error:', err);
    setStatus('❌ Camera denied or unavailable. Check browser permissions.');
    elStartBtn.style.display = 'inline-flex';
    elStopBtn.style.display  = 'none';
  }
};

/* ── STOP CAMERA ─────────────────────────────────────────────── */
window.stopCamera = function () {
  if (cameraInst)   { cameraInst.stop();   cameraInst   = null; }
  if (faceMeshInst) { faceMeshInst.close(); faceMeshInst = null; }

  camRunning  = false;
  frameBuffer = [];

  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  elStartBtn.style.display = 'inline-flex';
  elStopBtn.style.display  = 'none';
  setStatus('Camera stopped.');
  elCounter.textContent = '0 / 20 frames';
  elBar.style.width     = '0%';
  resetDisplay();
};

/* ── MEDIAPIPE RESULTS CALLBACK ─────────────────────────────── */
function onResults(results) {
  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks.length) {
    setStatus('No face detected — move closer and face the camera.');
    return;
  }

  const lm = results.multiFaceLandmarks[0];

  // ── Draw iris rings on canvas overlay ───────────────────────
  drawIris(lm);

  // ── Measure IPD in pixels ────────────────────────────────────
  const lx = lm[IDX_L_IRIS].x * canvasEl.width;
  const ly = lm[IDX_L_IRIS].y * canvasEl.height;
  const rx = lm[IDX_R_IRIS].x * canvasEl.width;
  const ry = lm[IDX_R_IRIS].y * canvasEl.height;
  const ipdPx = Math.hypot(rx - lx, ry - ly);

  // Sanity check — skip obviously wrong frames
  if (ipdPx < 8) return;

  // ── 20-frame rolling buffer (Issue #3) ──────────────────────
  frameBuffer.push(ipdPx);
  if (frameBuffer.length > FRAME_BUFFER_MAX) frameBuffer.shift();

  // Mean of buffer = smoothed IPD
  const smoothIPD = frameBuffer.reduce((a, b) => a + b, 0) / frameBuffer.length;

  // ── Calculate distance and diopters ─────────────────────────
  const distMm = (IPD_REAL_MM * FOCAL_LENGTH_PX) / smoothIPD;
  const distCm = distMm / 10;
  const diopt  = -100 / distCm;

  // ── Update the display ───────────────────────────────────────
  updateDisplay(distCm, diopt, ipdPx);
  updateProgress();
}

/* ── DRAW IRIS ON CANVAS ─────────────────────────────────────── */
function drawIris(lm) {
  const W = canvasEl.width, H = canvasEl.height;

  // Draw ring for each iris
  [L_RING, R_RING].forEach((ring, side) => {
    canvasCtx.beginPath();
    ring.forEach((idx, i) => {
      const x = lm[idx].x * W, y = lm[idx].y * H;
      i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
    });
    canvasCtx.closePath();
    canvasCtx.strokeStyle = side === 0 ? 'rgba(0,212,255,0.85)' : 'rgba(124,58,237,0.85)';
    canvasCtx.lineWidth   = 1.8;
    canvasCtx.stroke();
  });

  // Centre dots
  [IDX_L_IRIS, IDX_R_IRIS].forEach(idx => {
    const x = lm[idx].x * W, y = lm[idx].y * H;
    canvasCtx.beginPath();
    canvasCtx.arc(x, y, 3.5, 0, Math.PI * 2);
    canvasCtx.fillStyle = '#00D4FF';
    canvasCtx.fill();
  });

  // IPD dashed line
  const lx = lm[IDX_L_IRIS].x * W, ly = lm[IDX_L_IRIS].y * H;
  const rx = lm[IDX_R_IRIS].x * W, ry = lm[IDX_R_IRIS].y * H;
  canvasCtx.beginPath();
  canvasCtx.moveTo(lx, ly);
  canvasCtx.lineTo(rx, ry);
  canvasCtx.strokeStyle = 'rgba(6,255,165,0.55)';
  canvasCtx.lineWidth   = 1;
  canvasCtx.setLineDash([5, 4]);
  canvasCtx.stroke();
  canvasCtx.setLineDash([]);
}

/* ── UPDATE DISPLAY ─────────────────────────────────────────── */
function updateDisplay(distCm, diopt, ipdPx) {
  elDist.textContent = distCm.toFixed(1);
  elDiopt.textContent = diopt.toFixed(2) + ' D';
  elIPD.textContent   = ipdPx.toFixed(0);

  const absD = Math.abs(diopt);
  let sevText, sevCls, dioptCls;

  if (distCm >= 50) {
    sevText  = '✅ Healthy Distance';  sevCls = 'severity-normal';  dioptCls = '';
  } else if (absD < 3) {
    sevText  = '🟡 Mild Range';        sevCls = 'severity-mild';    dioptCls = '';
  } else if (absD < 6) {
    sevText  = '🟠 Moderate — Move Back'; sevCls = 'severity-moderate'; dioptCls = 'warning';
  } else {
    sevText  = '🔴 High — Move Back!'; sevCls = 'severity-high';   dioptCls = 'danger';
  }

  elSeverity.textContent = sevText;
  elSeverity.className   = 'severity-badge ' + sevCls;
  elDiopt.className      = 'live-value ' + dioptCls;
  elDist.className       = distCm < 25 ? 'live-value danger' : 'live-value';
}

function updateProgress() {
  const n   = frameBuffer.length;
  const pct = (n / FRAME_BUFFER_MAX) * 100;
  elBar.style.width     = pct + '%';
  elCounter.textContent = n + ' / ' + FRAME_BUFFER_MAX + ' frames';
  setStatus(n < FRAME_BUFFER_MAX
    ? 'Warming up… (' + n + '/' + FRAME_BUFFER_MAX + ' frames buffered)'
    : '✅ Live — 20-frame average active');
}

function setStatus(msg) {
  if (elStatus) elStatus.textContent = msg;
}

function resetDisplay() {
  elDist.textContent  = '--';  elDist.className  = 'live-value';
  elDiopt.textContent = '--';  elDiopt.className = 'live-value';
  elIPD.textContent   = '--';
  elSeverity.textContent = '—';
  elSeverity.className   = 'severity-badge';
}
