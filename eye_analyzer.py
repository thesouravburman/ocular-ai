import numpy as np
from PIL import Image, ImageDraw

MEDIAPIPE_AVAILABLE = False
_mp = None

def _try_load_mediapipe():
    global MEDIAPIPE_AVAILABLE, _mp
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, os
        MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        MODEL_PATH = "/tmp/face_landmarker.task"
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)
        _mp = {"mp": mp, "detector": detector}
        MEDIAPIPE_AVAILABLE = True
    except Exception:
        MEDIAPIPE_AVAILABLE = False

_try_load_mediapipe()

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def analyze_iris(image: Image.Image):
    """Returns (annotated_image, metrics_dict) or (None, {}) if unavailable."""
    if not MEDIAPIPE_AVAILABLE or _mp is None:
        return None, {}
    try:
        mp = _mp["mp"]
        detector = _mp["detector"]
        img_rgb = image.convert("RGB")
        img_array = np.array(img_rgb)
        h, w = img_array.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
        result = detector.detect(mp_image)
        if not result.face_landmarks:
            return None, {}
        landmarks = result.face_landmarks[0]
        annotated = img_rgb.copy()
        draw = ImageDraw.Draw(annotated)

        def get_iris(indices):
            pts = np.array([(landmarks[i].x*w, landmarks[i].y*h) for i in indices], dtype=np.float32)
            center = pts.mean(axis=0)
            radius = float(np.linalg.norm(pts - center, axis=1).mean())
            return center, radius

        lc, lr = get_iris(LEFT_IRIS)
        rc, rr = get_iris(RIGHT_IRIS)

        for center, radius in [(lc, lr), (rc, rr)]:
            cx, cy, r = int(center[0]), int(center[1]), max(int(radius), 3)
            draw.ellipse([cx-r-6,cy-r-6,cx+r+6,cy+r+6], outline=(0,80,100), width=1)
            draw.ellipse([cx-r-3,cy-r-3,cx+r+3,cy+r+3], outline=(0,150,180), width=1)
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=(0,212,255), width=3)
            draw.ellipse([cx-3,cy-3,cx+3,cy+3], fill=(6,255,165))
            draw.arc([cx-r+3,cy-r+3,cx+r-3,cy+r-3], 0, 90, fill=(6,255,165), width=2)
            draw.arc([cx-r+3,cy-r+3,cx+r-3,cy+r-3], 180, 270, fill=(6,255,165), width=2)

        draw.line([int(lc[0]),int(lc[1]),int(rc[0]),int(rc[1])], fill=(124,58,237), width=1)

        size_diff = abs(lr-rr) / max(lr,rr,1)
        symmetry = max(0.0, min(100.0, 100.0 - size_diff*300))
        ipd = float(np.linalg.norm(lc-rc))

        return annotated, {
            "left_radius_px":  round(lr,1),
            "right_radius_px": round(rr,1),
            "symmetry_score":  round(symmetry,1),
            "ipd_px":          round(ipd,1),
        }
    except Exception:
        return None, {}
