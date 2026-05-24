# OcularAI — Smartphone-Based Myopia Detection

![OcularAI Banner](https://img.shields.io/badge/OcularAI-v2.0-00D4FF?style=for-the-badge)
![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?style=for-the-badge&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-06FFA5?style=for-the-badge)

> **Live app → [ocular-ai-eight.vercel.app](https://ocular-ai-eight.vercel.app)**

OcularAI is an open-source smartphone-based myopia screening tool. It uses
**MediaPipe Face Mesh** (Google's browser-native AI) to detect iris positions
in real-time via your phone's front camera, then estimates your viewing distance
and maps it to a diopter value using the standard optometric formula:
D = −100 ÷ distance (cm)
⚠️ **Educational tool only — not a medical device.** Always consult a licensed optometrist.

---

## Features

| Feature | Details |
|---------|---------|
| 📐 Diopter Calculator | Enter any distance → get instant diopter estimate |
| 📷 Live Camera | Real-time iris detection via MediaPipe Face Mesh JS |
| 📊 Research Data | Interactive Chart.js charts on global myopia trends |
| 📏 Screen Calibration | Credit-card calibration for per-device accuracy |
| 🎯 20-Frame Averaging | Rolling average smooths out blink/motion noise |

---

## Tech Stack

- **Frontend** — Vanilla HTML, CSS, JavaScript (no framework)
- **AI / CV** — [MediaPipe Face Mesh JS](https://google.github.io/mediapipe/solutions/face_mesh)
- **Charts** — [Chart.js 4](https://www.chartjs.org/)
- **Fonts** — Space Grotesk + Inter (Google Fonts)
- **Hosting** — [Vercel](https://vercel.com) (free tier, static deployment)

---

## Repository Structure
ocular-ai/
├── index.html            # Main app — all 5 tabs
├── style.css             # Dark theme, glass card UI
├── camera.js             # Live camera + MediaPipe module
├── mediapipe_worker.js   # Shared utilities & constants
├── data.json             # Research chart data
├── vercel.json           # Vercel static site config
└── legacy/               # Original Streamlit v1.0 source
├── app.py
├── eye_analyzer.py
├── myopia_data.csv
└── README.md

---

## Contributing

Contributions welcome! See the open
[Issues](https://github.com/thesouravburman/ocular-ai/issues) and
[Pull Requests](https://github.com/thesouravburman/ocular-ai/pulls)
to get started.

1. Fork the repo
2. Edit files directly in the GitHub web editor
3. Open a Pull Request referencing the relevant Issue

---

## Author

**Sourav Burman** — [@thesouravburman](https://github.com/thesouravburman)
Final-year BTech student · Built as an open-source vision health project.

---

## License

MIT © 2025 Sourav Burman
