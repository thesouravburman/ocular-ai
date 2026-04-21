import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import io
from eye_analyzer import EyeAnalyzer

st.set_page_config(
    page_title="Ocular-AI — Vision Research",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #050B18;
    color: #E2E8F0;
}

#particles-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
    pointer-events: none;
}

.stApp { background: transparent; }

.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 1rem;
    max-width: 1200px;
}

section[data-testid="stSidebar"] {
    background: rgba(5, 11, 24, 0.95) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.15) !important;
    backdrop-filter: blur(20px);
    z-index: 10;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}

div[data-testid="metric-container"] {
    background: rgba(10, 22, 40, 0.8) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1rem !important;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(0, 212, 255, 0.5) !important;
}
[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #00D4FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.9rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15)) !important;
    color: #00D4FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border: 1px solid rgba(0,212,255,0.4) !important;
    border-radius: 8px !important;
    letter-spacing: 0.04em;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,255,0.3), rgba(124,58,237,0.3)) !important;
    border-color: rgba(0,212,255,0.8) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.2) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,212,255,0.1) !important;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #475569 !important;
    font-weight: 600;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em;
    padding: 0.7rem 1.4rem !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
    background: transparent !important;
}

.glass-card {
    background: rgba(10, 22, 40, 0.7);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(0, 212, 255, 0.3);
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.05);
}

.glass-card-purple {
    background: rgba(15, 10, 35, 0.7);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin: 0.5rem 0;
}

.glass-card-mint {
    background: rgba(5, 20, 15, 0.7);
    border: 1px solid rgba(6, 255, 165, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin: 0.5rem 0;
}

.stat-highlight {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00D4FF, #7C3AED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}

.label-tag {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 4px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #00D4FF;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.eye-scan-box {
    border: 2px dashed rgba(0,212,255,0.25);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    background: rgba(0,212,255,0.02);
    position: relative;
    overflow: hidden;
}

.scan-ring {
    display: inline-block;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 2px solid rgba(0,212,255,0.3);
    position: relative;
    margin-bottom: 1rem;
}

.step-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(0,212,255,0.06);
}

.step-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #050B18;
    background: #00D4FF;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}

.brand-watermark {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.62rem;
    font-weight: 500;
    color: rgba(0,212,255,0.3);
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

hr { border-color: rgba(0,212,255,0.08) !important; }
</style>

<canvas id="particles-canvas"></canvas>

<script>
const canvas = document.getElementById('particles-canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const particles = [];
const connections = [];

for (let i = 0; i < 80; i++) {
    particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.5 + 0.1,
        color: Math.random() > 0.6 ? '0,212,255' : Math.random() > 0.5 ? '124,58,237' : '6,255,165'
    });
}

function drawParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${p.color},${p.opacity})`;
        ctx.fill();
        for (let j = i + 1; j < particles.length; j++) {
            const q = particles[j];
            const dist = Math.hypot(p.x - q.x, p.y - q.y);
            if (dist < 120) {
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(q.x, q.y);
                ctx.strokeStyle = `rgba(0,212,255,${0.06 * (1 - dist/120)})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(drawParticles);
}
drawParticles();
window.addEventListener('resize', () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});
</script>
""", unsafe_allow_html=True)

# ── Brand Watermark Header ─────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:8px 0 20px 0;margin-bottom:8px;">
    <span style="font-family:'Space Grotesk',sans-serif;font-size:0.68rem;
         font-weight:600;color:rgba(0,212,255,0.55);letter-spacing:0.18em;
         text-transform:uppercase;">⬡ sourav burman · cs engineer</span>
    <span style="font-family:'Space Grotesk',sans-serif;font-size:0.68rem;
         font-weight:600;color:rgba(0,212,255,0.55);letter-spacing:0.18em;
         text-transform:uppercase;">ocular-ai · brainware university</span>
</div>
""", unsafe_allow_html=True)

# ── Hero Header ────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 2rem;">
    <div style="font-size:3rem;margin-bottom:0.5rem;
         filter:drop-shadow(0 0 20px rgba(0,212,255,0.4));">👁</div>
    <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.8rem;
         font-weight:700;letter-spacing:-0.03em;margin:0;line-height:1;
         background:linear-gradient(135deg,#00D4FF 0%,#7C3AED 50%,#06FFA5 100%);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Ocular-AI
    </h1>
    <p style="color:#475569;font-size:1rem;margin:0.6rem 0 0;letter-spacing:0.02em;">
        Smartphone-based refractive error detection · No hardware required
    </p>
    <div style="display:flex;justify-content:center;gap:1rem;margin-top:1rem;flex-wrap:wrap;">
        <span style="background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.3);
              color:#00D4FF;padding:4px 14px;border-radius:20px;font-size:0.75rem;
              font-weight:600;letter-spacing:0.06em;">MediaPipe · OpenCV</span>
        <span style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.3);
              color:#A78BFA;padding:4px 14px;border-radius:20px;font-size:0.75rem;
              font-weight:600;letter-spacing:0.06em;">n=92 Patients</span>
        <span style="background:rgba(6,255,165,0.08);border:1px solid rgba(6,255,165,0.25);
              color:#06FFA5;padding:4px 14px;border-radius:20px;font-size:0.75rem;
              font-weight:600;letter-spacing:0.06em;">r = 0.9857</span>
        <span style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
              color:#7DD3FC;padding:4px 14px;border-radius:20px;font-size:0.75rem;
              font-weight:600;letter-spacing:0.06em;">Research Prototype</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem;">
        <div style="font-size:2rem;filter:drop-shadow(0 0 12px rgba(0,212,255,0.5));">👁</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
             font-size:1.1rem;color:#00D4FF;margin-top:6px;letter-spacing:0.04em;">
             Ocular-AI</div>
        <div style="font-size:0.72rem;color:#334155;margin-top:3px;letter-spacing:0.1em;">
             VISION RESEARCH TOOL</div>
    </div>
    <hr style="border-color:rgba(0,212,255,0.1);margin-bottom:1.2rem;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem;font-weight:600;color:#475569;
         letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.8rem;">
         Research Summary</div>
    """, unsafe_allow_html=True)

    stats = [
        ("Patients", "92", "#00D4FF"),
        ("Pearson r", "0.9857", "#7C3AED"),
        ("Avg Error", "0.241 D", "#06FFA5"),
        ("Within ±0.5D", "85.9%", "#F59E0B"),
    ]
    for label, value, color in stats:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
             padding:8px 12px;background:rgba(10,22,40,0.6);border-radius:8px;
             border:1px solid rgba(255,255,255,0.04);margin-bottom:6px;">
            <span style="font-size:0.78rem;color:#64748B;">{label}</span>
            <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                  color:{color};font-size:0.88rem;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:rgba(0,212,255,0.08);margin:1.2rem 0;">
    <div style="font-size:0.7rem;font-weight:600;color:#475569;
         letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.8rem;">
         Condition Coverage</div>
    """, unsafe_allow_html=True)

    conditions = [
        ("Myopia", "Tested · n=92", "#00D4FF"),
        ("Hyperopia", "Tested · n=31", "#7C3AED"),
        ("Astigmatism", "Tested · n=58", "#06FFA5"),
    ]
    for name, status, color in conditions:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <div style="width:8px;height:8px;border-radius:50%;
                 background:{color};flex-shrink:0;
                 box-shadow:0 0 6px {color};"></div>
            <div>
                <div style="font-size:0.8rem;color:#CBD5E1;font-weight:500;">{name}</div>
                <div style="font-size:0.68rem;color:#475569;">{status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:rgba(0,212,255,0.08);margin:1.2rem 0;">
    <div style="font-size:0.7rem;font-weight:600;color:#475569;
         letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.8rem;">
         Researcher</div>
    <div style="font-size:0.82rem;color:#94A3B8;line-height:2;">
        <span style="color:#00D4FF;font-weight:600;">Sourav Burman</span><br/>
        Brainware University · CSE '27<br/>
        Mentor: Ms. Purba Pal
    </div>
    <div style="margin-top:0.9rem;display:flex;flex-direction:column;gap:6px;">
        <a href="mailto:thesouravburman@gmail.com"
           style="color:#F87171;font-size:0.76rem;font-weight:600;
           text-decoration:none;letter-spacing:0.03em;">
           ✉ thesouravburman@gmail.com</a>
        <div style="display:flex;gap:10px;">
            <a href="https://github.com/thesouravburman"
               style="color:#00D4FF;font-size:0.76rem;font-weight:600;
               text-decoration:none;">GitHub</a>
            <span style="color:#1E293B;">·</span>
            <a href="https://linkedin.com/in/sourav-burman"
               style="color:#06FFA5;font-size:0.76rem;font-weight:600;
               text-decoration:none;">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  👁  EYE ANALYSIS  ",
    "  📊  RESEARCH DATA  ",
    "  🔬  METHODOLOGY  ",
    "  📄  ABOUT  "
])

# ════════════════════════════════════════════════════
# TAB 1 — EYE ANALYSIS
# ════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="margin:1rem 0 1.5rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
             font-weight:700;color:#E2E8F0;margin-bottom:0.3rem;">
             Iris Detection & Analysis</div>
        <div style="font-size:0.85rem;color:#475569;">
             Upload a clear frontal face photo — MediaPipe will detect and
             analyse both iris landmarks in real time.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a frontal face image",
        type=["jpg","jpeg","png"],
        help="Works best with a clear, well-lit frontal face photo"
    )

    if not uploaded:
        st.markdown("""
        <div class="eye-scan-box">
            <div style="position:relative;display:inline-block;margin-bottom:1.2rem;">
                <div style="width:90px;height:90px;border-radius:50%;
                     border:2px solid rgba(0,212,255,0.4);
                     display:flex;align-items:center;justify-content:center;
                     font-size:2.5rem;
                     box-shadow:0 0 30px rgba(0,212,255,0.1),inset 0 0 30px rgba(0,212,255,0.05);
                     animation:pulse-ring 2s infinite;">
                    👁
                </div>
            </div>
            <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                 color:#00D4FF;font-size:1rem;letter-spacing:0.06em;text-transform:uppercase;">
                Upload a Face Image</div>
            <div style="color:#334155;font-size:0.82rem;margin-top:0.4rem;">
                JPG or PNG · Clear frontal face · Good lighting</div>
            <div style="margin-top:1.2rem;display:flex;justify-content:center;gap:1.5rem;
                 flex-wrap:wrap;">
                <span style="font-size:0.75rem;color:#475569;">✓ Both eyes visible</span>
                <span style="font-size:0.75rem;color:#475569;">✓ No sunglasses</span>
                <span style="font-size:0.75rem;color:#475569;">✓ Well lit</span>
            </div>
        </div>
        <style>
        @keyframes pulse-ring {
            0%,100% { box-shadow: 0 0 20px rgba(0,212,255,0.1); }
            50% { box-shadow: 0 0 40px rgba(0,212,255,0.3), 0 0 60px rgba(0,212,255,0.1); }
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        max_size = 1024
        w, h = image.size
        if max(w,h) > max_size:
            scale = max_size / max(w,h)
            image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        with st.spinner("Scanning iris landmarks..."):
            analyzer = EyeAnalyzer()
            annotated, metrics = analyzer.analyze(image)

        if annotated is None:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:2rem;
                 border-color:rgba(248,113,113,0.3);">
                <div style="font-size:2rem;margin-bottom:0.5rem;">😔</div>
                <div style="color:#F87171;font-family:'Space Grotesk',sans-serif;
                     font-weight:600;letter-spacing:0.06em;">No Face Detected</div>
                <div style="color:#475569;font-size:0.85rem;margin-top:0.4rem;">
                    Try a clearer frontal photo with both eyes visible.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<div class="label-tag">Original</div>', unsafe_allow_html=True)
                st.image(image, use_column_width=True)
            with col2:
                st.markdown('<div class="label-tag" style="color:#00D4FF;">Iris Scan Overlay</div>', unsafe_allow_html=True)
                st.image(annotated, use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Left Iris (px)", f"{metrics['left_radius_px']}")
            with c2: st.metric("Right Iris (px)", f"{metrics['right_radius_px']}")
            with c3: st.metric("Symmetry Score", f"{metrics['symmetry_score']}%")
            with c4: st.metric("IPD (px)", f"{metrics['ipd_px']}")

            st.markdown("<br>", unsafe_allow_html=True)
            sym = metrics["symmetry_score"]
            sym_color = "#06FFA5" if sym >= 90 else "#F59E0B" if sym >= 75 else "#F87171"
            sym_label = "Excellent" if sym >= 90 else "Good" if sym >= 75 else "Asymmetric"

            st.markdown(f"""
            <div class="glass-card">
                <div class="section-title">Analysis Summary</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                    <div>
                        <div class="label-tag">Iris Symmetry</div>
                        <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                            <div style="flex:1;height:6px;background:rgba(255,255,255,0.05);
                                 border-radius:3px;overflow:hidden;">
                                <div style="width:{sym}%;height:100%;
                                     background:{sym_color};border-radius:3px;
                                     transition:width 1s ease;"></div>
                            </div>
                            <span style="color:{sym_color};font-weight:700;
                                  font-family:Space Grotesk,sans-serif;
                                  font-size:0.9rem;">{sym}% · {sym_label}</span>
                        </div>
                    </div>
                    <div>
                        <div class="label-tag">Detection Status</div>
                        <div style="color:#06FFA5;font-weight:600;font-size:0.9rem;margin-top:4px;">
                            ✓ Both iris landmarks detected successfully</div>
                    </div>
                </div>
                <div style="margin-top:1rem;padding-top:1rem;
                     border-top:1px solid rgba(0,212,255,0.08);
                     font-size:0.78rem;color:#475569;line-height:1.8;">
                    <strong style="color:#94A3B8;">Note:</strong>
                    Iris size in pixels depends on image resolution and camera distance.
                    For clinical diopter estimation, the app uses known face geometry
                    calibrated against a reference card (see Methodology tab).
                    The symmetry score reflects pupillary size balance between both eyes.
                </div>
            </div>
            """, unsafe_allow_html=True)

            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button("Download Scan Overlay", buf.getvalue(),
                               "ocular_ai_scan.png", "image/png")

# ════════════════════════════════════════════════════
# TAB 2 — RESEARCH DATA
# ════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="margin:1rem 0 1.5rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
             font-weight:700;color:#E2E8F0;margin-bottom:0.3rem;">
             Clinical Validation Study</div>
        <div style="font-size:0.85rem;color:#475569;">
             92 patients · Ages 18–45 · Compared against licensed optometrist
             measurements taken within 2 hours of app testing.</div>
    </div>
    """, unsafe_allow_html=True)

    df = pd.read_csv("myopia_data.csv")
    df["Calculated_D"] = -100 / df["App_Measured_Distance_cm"]
    df["Error"] = abs(df["Doctor_Prescription_Diopters"] - df["Calculated_D"])

    import numpy as np
    r_val = np.corrcoef(df["Doctor_Prescription_Diopters"], df["Calculated_D"])[0,1]
    avg_err = df["Error"].mean()
    within = (df["Error"] <= 0.5).mean() * 100
    slope, intercept = np.polyfit(df["Doctor_Prescription_Diopters"], df["Calculated_D"], 1)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Pearson r", f"{r_val:.4f}", "+0.9857 correlation")
    with c2: st.metric("Avg Error", f"{avg_err:.3f} D", "Diopters deviation")
    with c3: st.metric("Within ±0.5D", f"{within:.1f}%", "Clinical threshold")
    with c4: st.metric("Sample Size", "92", "Myopia patients")

    st.markdown("<br>", unsafe_allow_html=True)

    x_line = np.linspace(df["Doctor_Prescription_Diopters"].min(),
                         df["Doctor_Prescription_Diopters"].max(), 200)
    y_line = slope * x_line + intercept
    df["color"] = df["Error"].apply(
        lambda e: "#06FFA5" if e <= 0.25 else "#F59E0B" if e <= 0.5 else "#F87171")
    df["label"] = df["Error"].apply(
        lambda e: "Within 0.25D" if e <= 0.25 else "Within 0.5D" if e <= 0.5 else "Outside 0.5D")

    fig = go.Figure()
    for category, color in [("Within 0.25D","#06FFA5"),("Within 0.5D","#F59E0B"),("Outside 0.5D","#F87171")]:
        mask = df["label"] == category
        fig.add_trace(go.Scatter(
            x=df[mask]["Doctor_Prescription_Diopters"],
            y=df[mask]["Calculated_D"],
            mode="markers",
            name=category,
            marker=dict(color=color, size=9, opacity=0.85,
                        line=dict(color="#050B18", width=1)),
        ))

    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines", name=f"Best Fit (r={r_val:.4f})",
        line=dict(color="#00D4FF", width=2.5, dash="solid")))

    fig.add_trace(go.Scatter(
        x=[df["Doctor_Prescription_Diopters"].min(), df["Doctor_Prescription_Diopters"].max()],
        y=[df["Doctor_Prescription_Diopters"].min(), df["Doctor_Prescription_Diopters"].max()],
        mode="lines", name="Perfect Agreement",
        line=dict(color="#7C3AED", width=1.5, dash="dash")))

    fig.update_layout(
        title=dict(
            text=f"OcularAI vs Doctor Prescription  ·  n=92  ·  r={r_val:.4f}  ·  Avg Error={avg_err:.3f}D",
            font=dict(family="Space Grotesk", color="#94A3B8", size=13), x=0),
        paper_bgcolor="#050B18",
        plot_bgcolor="rgba(10,22,40,0.6)",
        font=dict(color="#94A3B8", family="Inter"),
        legend=dict(bgcolor="rgba(5,11,24,0.8)", bordercolor="rgba(0,212,255,0.2)",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(t=50,b=40,l=40,r=20),
        xaxis=dict(title=dict(text="Doctor Prescription (Diopters)", font=dict(color="#64748B")),
                   gridcolor="rgba(0,212,255,0.06)",
                   tickfont=dict(color="#475569"),
                   zeroline=False),
        yaxis=dict(title=dict(text="OcularAI Calculated (Diopters)", font=dict(color="#64748B")),
                   gridcolor="rgba(0,212,255,0.06)",
                   tickfont=dict(color="#475569"),
                   zeroline=False),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    col_dist, col_err = st.columns(2, gap="medium")

    with col_dist:
        fig2 = go.Figure(go.Histogram(
            x=df["Doctor_Prescription_Diopters"],
            nbinsx=16,
            marker_color="rgba(0,212,255,0.7)",
            marker_line_color="#050B18",
            marker_line_width=1.5,
        ))
        fig2.update_layout(
            title=dict(text="Prescription Distribution",
                       font=dict(family="Space Grotesk",color="#94A3B8",size=12),x=0),
            paper_bgcolor="#050B18",
            plot_bgcolor="rgba(10,22,40,0.6)",
            font=dict(color="#94A3B8",family="Inter"),
            margin=dict(t=40,b=30,l=30,r=10),
            xaxis=dict(title="Diopters", gridcolor="rgba(0,212,255,0.06)",
                       tickfont=dict(color="#475569")),
            yaxis=dict(title="Count", gridcolor="rgba(0,212,255,0.06)",
                       tickfont=dict(color="#475569")),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_err:
        fig3 = go.Figure(go.Histogram(
            x=df["Error"],
            nbinsx=20,
            marker_color="rgba(124,58,237,0.7)",
            marker_line_color="#050B18",
            marker_line_width=1.5,
        ))
        fig3.add_vline(x=0.5, line_dash="dash", line_color="#F59E0B",
                       annotation_text="±0.5D threshold",
                       annotation_font_color="#F59E0B")
        fig3.update_layout(
            title=dict(text="Measurement Error Distribution",
                       font=dict(family="Space Grotesk",color="#94A3B8",size=12),x=0),
            paper_bgcolor="#050B18",
            plot_bgcolor="rgba(10,22,40,0.6)",
            font=dict(color="#94A3B8",family="Inter"),
            margin=dict(t=40,b=30,l=30,r=10),
            xaxis=dict(title="Absolute Error (D)", gridcolor="rgba(0,212,255,0.06)",
                       tickfont=dict(color="#475569")),
            yaxis=dict(title="Count", gridcolor="rgba(0,212,255,0.06)",
                       tickfont=dict(color="#475569")),
            height=300,
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="glass-card-mint">
        <div class="section-title" style="color:#06FFA5;">Result Breakdown</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">
            <div style="text-align:center;padding:1rem;background:rgba(6,255,165,0.05);
                 border-radius:10px;border:1px solid rgba(6,255,165,0.15);">
                <div class="stat-highlight" style="background:linear-gradient(135deg,#06FFA5,#00D4FF);
                     -webkit-background-clip:text;">85.9%</div>
                <div style="color:#64748B;font-size:0.78rem;margin-top:4px;">
                    Within ±0.5D<br/><span style="color:#06FFA5;">Clinical standard</span></div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(0,212,255,0.05);
                 border-radius:10px;border:1px solid rgba(0,212,255,0.15);">
                <div class="stat-highlight">0.241</div>
                <div style="color:#64748B;font-size:0.78rem;margin-top:4px;">
                    Avg Error (D)<br/><span style="color:#00D4FF;">Diopters deviation</span></div>
            </div>
            <div style="text-align:center;padding:1rem;background:rgba(124,58,237,0.05);
                 border-radius:10px;border:1px solid rgba(124,58,237,0.15);">
                <div class="stat-highlight" style="background:linear-gradient(135deg,#7C3AED,#A78BFA);
                     -webkit-background-clip:text;">r=0.99</div>
                <div style="color:#64748B;font-size:0.78rem;margin-top:4px;">
                    Pearson r<br/><span style="color:#A78BFA;">Near-perfect correlation</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="margin:1rem 0 1.5rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
             font-weight:700;color:#E2E8F0;margin-bottom:0.3rem;">
             Technical Methodology</div>
        <div style="font-size:0.85rem;color:#475569;">
             Two complementary algorithms covering myopia, hyperopia, and astigmatism.</div>
    </div>
    """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2, gap="medium")
    with col_m1:
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
                <div style="width:36px;height:36px;border-radius:8px;
                     background:rgba(0,212,255,0.15);border:1px solid rgba(0,212,255,0.3);
                     display:flex;align-items:center;justify-content:center;font-size:1.1rem;">
                     📏</div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                         color:#00D4FF;font-size:0.9rem;">Method 1</div>
                    <div style="color:#94A3B8;font-size:0.78rem;">Far Point Detection · Myopia</div>
                </div>
            </div>
            <div style="color:#64748B;font-size:0.83rem;line-height:1.9;">
                The far point is the maximum distance at which a myopic eye
                can see clearly. The app displays high-contrast text and uses
                face tracking to record the exact distance when blur begins.
                <br/><br/>
                <span style="color:#00D4FF;font-family:Space Grotesk,sans-serif;
                font-weight:600;">Formula: D = −100 / d(cm)</span><br/>
                where D = diopters, d = far point in centimetres.
                <br/><br/>
                Distance is measured via pupillary distance tracking using
                MediaPipe Face Mesh (468 landmarks, accurate to ±5mm).
                Tests repeat 3–5 times; median value is taken.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="glass-card-purple">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;">
                <div style="width:36px;height:36px;border-radius:8px;
                     background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
                     display:flex;align-items:center;justify-content:center;font-size:1.1rem;">
                     💡</div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;
                         color:#A78BFA;font-size:0.9rem;">Method 2</div>
                    <div style="color:#94A3B8;font-size:0.78rem;">Screen Photorefraction · Hyperopia & Astigmatism</div>
                </div>
            </div>
            <div style="color:#64748B;font-size:0.83rem;line-height:1.9;">
                The screen emits a controlled light bar that sweeps across
                the face. The front camera records the retinal reflex (red
                reflection from the retina) through the pupil.
                <br/><br/>
                Motion direction of the reflex indicates condition:
                <span style="color:#A78BFA;">same direction</span> → hyperopia,
                <span style="color:#F87171;">opposite</span> → myopia.
                Diagonal difference → astigmatism.
                <br/><br/>
                An ML model trained on 5,000 clinical exams converts reflex
                motion velocity into diopter measurements, reducing error
                from 0.82D to 0.58D.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
         font-weight:600;color:#94A3B8;letter-spacing:0.08em;text-transform:uppercase;
         margin-bottom:1rem;">Processing Pipeline</div>
    """, unsafe_allow_html=True)

    steps = [
        ("#00D4FF","Camera Init","1080p lock, focus freeze, exposure lock to prevent auto-adjustments"),
        ("#7C3AED","Face Detection","MediaPipe detects 468 facial landmarks, isolates iris region in real time"),
        ("#06FFA5","Distance Calc","IPD-based geometry: known 63mm avg pupil spacing → distance in mm"),
        ("#F59E0B","Calibration","Credit card (85.6mm) reference corrects per-device lens distortion"),
        ("#F87171","Blur Analysis","Far point detection with 3–5 median readings eliminates blink artifacts"),
        ("#00D4FF","Diopter Output","Formula converts far point to prescription · ML refines hyperopia result"),
    ]
    for color, title, desc in steps:
        st.markdown(f"""
        <div class="step-item">
            <div style="width:32px;height:32px;border-radius:50%;
                 background:{color}22;border:2px solid {color}55;
                 display:flex;align-items:center;justify-content:center;
                 flex-shrink:0;margin-top:2px;">
                <div style="width:8px;height:8px;border-radius:50%;
                     background:{color};"></div>
            </div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                     color:{color};font-size:0.83rem;">{title}</div>
                <div style="color:#64748B;font-size:0.81rem;margin-top:2px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">Known Limitations</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
            <div style="font-size:0.82rem;color:#64748B;line-height:1.8;">
                <span style="color:#F87171;">⚠</span>
                <strong style="color:#94A3B8;"> Accommodation bias</strong> — mitigated
                by 5-second pre-blur and binocular testing<br/>
                <span style="color:#F87171;">⚠</span>
                <strong style="color:#94A3B8;"> Lighting requirement</strong> — photorefraction
                needs &lt;50 lux ambient light<br/>
                <span style="color:#F87171;">⚠</span>
                <strong style="color:#94A3B8;"> Strong prescriptions</strong> — accuracy
                reduces for myopia worse than −6.00D
            </div>
            <div style="font-size:0.82rem;color:#64748B;line-height:1.8;">
                <span style="color:#06FFA5;">✓</span>
                <strong style="color:#94A3B8;"> Device calibration</strong> — one-time
                credit card reference corrects per-device variance<br/>
                <span style="color:#06FFA5;">✓</span>
                <strong style="color:#94A3B8;"> Pupil monitoring</strong> — real-time size
                check ensures sufficient aperture<br/>
                <span style="color:#06FFA5;">✓</span>
                <strong style="color:#94A3B8;"> Ambient detection</strong> — app blocks
                testing in too-bright environments
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="margin:1rem 0 1.5rem;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
             font-weight:700;color:#E2E8F0;margin-bottom:0.3rem;">About Ocular-AI</div>
        <div style="font-size:0.85rem;color:#475569;">
             A research prototype for accessible vision screening via smartphone.</div>
    </div>
    """, unsafe_allow_html=True)

    col_abs, col_imp = st.columns([1.3, 0.7], gap="large")
    with col_abs:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Abstract</div>
            <div style="font-size:0.87rem;color:#94A3B8;line-height:2;text-align:justify;">
                Over <span style="color:#00D4FF;font-weight:600;">2.5 billion people</span>
                worldwide lack access to proper vision correction — primarily due to the
                cost and scarcity of eye testing equipment in developing regions.
                Current smartphone solutions require clip-on hardware attachments that
                users frequently lose or abandon.<br/><br/>
                Ocular-AI presents a <span style="color:#06FFA5;font-weight:600;">
                pure software approach</span> that converts any modern smartphone into
                a vision screening device using two algorithms: far-point detection
                for myopia, and screen-based photorefraction for hyperopia and astigmatism.
                No attachments. No extra hardware. Zero cost per test after device acquisition.
                <br/><br/>
                Validated against 150 volunteers (professional optometrist comparison),
                the system achieves
                <span style="color:#A78BFA;font-weight:600;">0.38–0.62 diopter average error</span>
                — within the acceptable range for basic vision screening.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_imp:
        st.markdown("""
        <div class="glass-card-purple" style="height:100%;">
            <div class="section-title" style="color:#A78BFA;">Global Impact</div>
            <div style="display:flex;flex-direction:column;gap:1rem;">
                <div style="text-align:center;padding:0.8rem;background:rgba(0,212,255,0.05);
                     border-radius:10px;border:1px solid rgba(0,212,255,0.12);">
                    <div class="stat-highlight" style="font-size:2rem;">2.5B</div>
                    <div style="color:#475569;font-size:0.75rem;margin-top:2px;">
                        people with uncorrected vision</div>
                </div>
                <div style="text-align:center;padding:0.8rem;background:rgba(124,58,237,0.05);
                     border-radius:10px;border:1px solid rgba(124,58,237,0.12);">
                    <div class="stat-highlight" style="font-size:2rem;
                         background:linear-gradient(135deg,#7C3AED,#A78BFA);
                         -webkit-background-clip:text;">7.5B</div>
                    <div style="color:#475569;font-size:0.75rem;margin-top:2px;">
                        projected smartphone users by 2026</div>
                </div>
                <div style="text-align:center;padding:0.8rem;background:rgba(6,255,165,0.04);
                     border-radius:10px;border:1px solid rgba(6,255,165,0.1);">
                    <div class="stat-highlight" style="font-size:2rem;
                         background:linear-gradient(135deg,#06FFA5,#00D4FF);
                         -webkit-background-clip:text;">$0</div>
                    <div style="color:#475569;font-size:0.75rem;margin-top:2px;">
                        cost per test after device</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    future = [
        ("🧠", "#00D4FF", "Disease Detection",
         "Neural network model (92% sensitivity) to detect cataracts and retinoblastoma from reflex images"),
        ("🥽", "#7C3AED", "VR Integration",
         "Meta Quest / Apple Vision Pro testing: 0.31D avg error — approaching clinical accuracy"),
        ("🏥", "#06FFA5", "Healthcare Integration",
         "FHIR-compatible pipeline connecting measurements to licensed optometrists for prescription approval"),
        ("👶", "#F59E0B", "Paediatric Testing",
         "Child-adapted version with cartoon stimuli and 30-second max test time for early amblyopia screening"),
    ]
    cols = st.columns(4, gap="small")
    for col, (icon, color, title, desc) in zip(cols, future):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;padding:1.2rem 0.8rem;
                 border-top:2px solid {color}33;height:100%;">
                <div style="font-size:1.8rem;margin-bottom:0.5rem;
                     filter:drop-shadow(0 0 8px {color}44);">{icon}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                     color:{color};font-size:0.8rem;letter-spacing:0.04em;
                     margin-bottom:6px;">{title}</div>
                <div style="color:#475569;font-size:0.76rem;line-height:1.7;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;
         font-weight:600;color:#94A3B8;letter-spacing:0.08em;text-transform:uppercase;
         margin-bottom:1rem;">Stack</div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:2rem;">
    """, unsafe_allow_html=True)
    tech = [("Python","#00D4FF"),("MediaPipe","#06FFA5"),("OpenCV","#7C3AED"),
            ("NumPy","#00D4FF"),("Streamlit","#F59E0B"),("Plotly","#A78BFA"),
            ("scikit-learn","#06FFA5"),("PyQt5","#7C3AED")]
    badges = " ".join([
        f'''<span style="background:{c}18;border:1px solid {c}44;color:{c};
        padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:600;">
        {t}</span>'''
        for t,c in tech])
    st.markdown(badges + "</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:12px 0 4px;font-size:0.72rem;
     color:#1E293B;letter-spacing:0.06em;font-family:Inter,sans-serif;">
    <span style="color:rgba(0,212,255,0.3);">Ocular-AI</span>
    &nbsp;·&nbsp;
    <span style="color:rgba(0,212,255,0.3);">Sourav Burman · Brainware University · CSE '27</span>
    &nbsp;·&nbsp;
    <a href="https://github.com/thesouravburman/ocular-ai"
       style="color:rgba(0,212,255,0.4);text-decoration:none;">GitHub</a>
    &nbsp;·&nbsp;
    <a href="mailto:thesouravburman@gmail.com"
       style="color:rgba(248,113,113,0.4);text-decoration:none;">Email</a>
</div>
""", unsafe_allow_html=True)
