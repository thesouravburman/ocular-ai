/* ============================================================
   mediapipe_worker.js — OcularAI Shared Utilities
   Exposes window.OcularUtils for use by camera.js.

   NOTE: MediaPipe Face Mesh uses WebGL (GPU acceleration) which
   is only available in the main browser thread — not inside a
   Web Worker. This file therefore provides shared helper
   functions as a regular script rather than a Worker module.
   ============================================================ */

'use strict';

window.OcularUtils = {

  /* Classify an absolute diopter value into a severity level.
     Returns an object with label, CSS class, and advice text. */
  classify(absD) {
    if (absD < 0.5) return {
      label: 'Emmetropia (Normal)',
      cls:   'severity-normal',
      tip:   'Great! Your viewing distance appears within the normal range.'
    };
    if (absD < 3) return {
      label: 'Mild Myopia',
      cls:   'severity-mild',
      tip:   'Mild range. Follow the 20-20-20 rule: every 20 min, look 20 ft away for 20 sec.'
    };
    if (absD < 6) return {
      label: 'Moderate Myopia',
      cls:   'severity-moderate',
      tip:   'Moderate range. Increase viewing distance and schedule an eye exam.'
    };
    if (absD < 9) return {
      label: 'High Myopia',
      cls:   'severity-high',
      tip:   'High range. Move back significantly. Consult an optometrist soon.'
    };
    return {
      label: 'Severe Myopia',
      cls:   'severity-high',
      tip:   'Severe range. Please see an optometrist as soon as possible.'
    };
  },

  /* Compute the mean (average) of a numeric array. */
  mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  },

  /* Read the saved pixels-per-mm calibration value from localStorage.
     Returns null if the user has not calibrated yet. */
  getPixelsPerMm() {
    const v = parseFloat(localStorage.getItem('ocular_pixels_per_mm'));
    return isNaN(v) ? null : v;
  },

  /* Format a diopter number for display, e.g. -3.33 → "-3.33 D" */
  fmt(d) {
    return (d >= 0 ? '+' : '') + d.toFixed(2) + ' D';
  },

  /* Reference table used in methodology explanations */
  SEVERITY_TABLE: [
    { range: '0 to -0.5 D',  label: 'Normal',          color: '#06FFA5' },
    { range: '-0.5 to -3 D', label: 'Mild myopia',     color: '#FFB800' },
    { range: '-3 to -6 D',   label: 'Moderate myopia', color: '#FF6432' },
    { range: '-6 to -9 D',   label: 'High myopia',     color: '#FF3B3B' },
    { range: '> -9 D',       label: 'Severe myopia',   color: '#CC0000' }
  ],

  /* MediaPipe landmark indices (reference) */
  LANDMARKS: {
    LEFT_IRIS_CENTER:  468,
    RIGHT_IRIS_CENTER: 473,
    LEFT_IRIS_RING:    [468, 469, 470, 471, 472],
    RIGHT_IRIS_RING:   [473, 474, 475, 476, 477]
  }
};

console.log('[OcularAI] OcularUtils loaded. Calibrated:',
  !!localStorage.getItem('ocular_calibrated'));
