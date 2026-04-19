# 👁️ Ocular-AI — Smartphone Refractive Error Detection

> A software-only vision screening algorithm using smartphone screen-based photorefraction — no specialised hardware required.

**Status:** Research prototype · Advancing toward clinical validation trials
**Role:** Lead Author / Developer

---

## 🔬 Research Overview

Ocular-AI proposes a novel approach to vision screening using only a smartphone:

1. The screen emits a controlled light pattern
2. The front camera captures the light reflex from the subject's pupil
3. MediaPipe tracks pupil position and reflex characteristics in real time
4. An algorithm estimates refractive error (myopia/hyperopia) in diopters

This eliminates the need for retinoscopes, photorefractors, or any hardware attachments.

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Vision:** OpenCV, MediaPipe Face Mesh
- **Computation:** NumPy, SciPy
- **Visualisation:** Matplotlib

## 📁 Project Structure
## ⚡ Quick Start

```bash
git clone https://github.com/thesouravburman/ocular-ai
cd ocular-ai
pip install -r requirements.txt
python demo.py
```

> **Note:** Requires a front-facing camera. Works best in dim lighting.

## 📄 Research Status

The prototype implements the core pupil tracking and reflex analysis pipeline.
Clinical validation trials are currently being planned.

## 📬 Contact

**Sourav Burman** (Lead Author) — thesouravburman@gmail.com · [LinkedIn](https://linkedin.com/in/sourav-burman)
