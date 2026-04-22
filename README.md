# 👁️ Ocular-AI — Smartphone Refractive Error Detection

<p align="center">
  <img src="https://img.shields.io/badge/Research-Prototype-7C3AED?style=for-the-badge&logo=academia&logoColor=white"/>
  <img src="https://img.shields.io/badge/MediaPipe-Powered-00D4FF?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/No Hardware-Required-06FFA5?style=for-the-badge&logoColor=white"/>
  <a href="https://ocular-ai.streamlit.app">
    <img src="https://img.shields.io/badge/🌐 Live Demo-Click Here-FF4B4B?style=for-the-badge"/>
  </a>
</p>

<p align="center">
  <b>A software-only vision screening algorithm using smartphone screen-based photorefraction</b><br/>
  No specialised hardware required · Research prototype · Advancing toward clinical validation
</p>

---

## 🌐 Live Demo

# 👉 [https://ocular-ai.streamlit.app](https://ocular-ai.streamlit.app)

Open the app and instantly:
- ✅ Screen for myopia using the far-point distance method
- ✅ Get estimated prescription in diopters with severity classification
- ✅ View full clinical validation study with interactive charts (n=92, r=0.9857)
- ✅ Explore the technical methodology and research pipeline
- ✅ Optional iris landmark detection overlay via MediaPipe

---

## 🔬 Research Overview

Ocular-AI proposes a novel approach to vision screening using only a smartphone:

1. The screen emits a controlled light pattern
2. The front camera captures the light reflex from the subject's pupil
3. MediaPipe tracks pupil position and reflex characteristics in real time
4. An algorithm estimates refractive error (myopia/hyperopia) in diopters

This eliminates the need for retinoscopes, photorefractors, or any hardware attachments.

---

## 🎯 What It Does

| Feature | Description |
|---------|-------------|
| 📏 Far-Point Screening | User enters max clear-vision distance → app calculates `D = −100 / d(cm)` |
| 🏷️ Severity Classification | Normal → Mild → Moderate → High → Severe myopia |
| 📊 Clinical Validation | Interactive scatter plot, error histogram, Pearson r from real study data |
| 👁️ Iris Detection | MediaPipe face mesh overlay (468 landmarks, both eyes) |
| 📄 Research Paper | Full methodology, limitations, and future directions |

---

## 📊 Validation Results

| Metric | Value |
|--------|-------|
| Sample Size | 92 myopia patients (ages 18–45) |
| Pearson Correlation | **r = 0.9857** |
| Average Error | **0.241 diopters** |
| Within ±0.5D | **85.9%** (clinical standard) |
| Diopter Range | −0.25 to −8.00 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Vision | MediaPipe Face Mesh (468 landmarks) |
| Computation | NumPy, SciPy |
| Web App | Streamlit |
| Charts | Plotly |
| Language | Python 3.11 |

---

## ⚡ Run Locally

```bash
git clone https://github.com/thesouravburman/ocular-ai
cd ocular-ai
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 🔭 Future Directions

- 🧠 **Disease Detection** — Neural network for cataract and retinoblastoma screening (92% sensitivity)
- 🥽 **VR Integration** — Meta Quest / Apple Vision Pro (0.31D avg error)
- 🏥 **Healthcare API** — FHIR-compatible pipeline for optometrist review
- 👶 **Paediatric Mode** — Cartoon stimuli, <30 second test for early amblyopia detection

---

## 📬 Contact

**Sourav Burman** (Lead Author / Developer)
- ✉️ thesouravburman@gmail.com
- [GitHub](https://github.com/thesouravburman) · [LinkedIn](https://linkedin.com/in/sourav-burman)

**Mentor:** Ms. Purba Pal · Department of CSE, Brainware University
