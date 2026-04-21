import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

class EyeAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze(self, image: Image.Image):
        img_rgb = image.convert("RGB")
        img_array = np.array(img_rgb)
        h, w = img_array.shape[:2]
        results = self.face_mesh.process(img_array)
        if not results.multi_face_landmarks:
            return None, {}
        landmarks = results.multi_face_landmarks[0].landmark
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
            cx, cy, r = int(center[0]), int(center[1]), int(radius)
            draw.ellipse([cx-r-6,cy-r-6,cx+r+6,cy+r+6], outline=(0,80,100), width=1)
            draw.ellipse([cx-r-3,cy-r-3,cx+r+3,cy+r+3], outline=(0,150,180), width=1)
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], outline=(0,212,255), width=3)
            draw.ellipse([cx-3,cy-3,cx+3,cy+3], fill=(6,255,165))
            for dx in [(-r-14,0),(-r+8,0),(r-8,0),(r+14,0)]:
                draw.line([cx+dx[0]-6,cy,cx+dx[0]+6,cy], fill=(0,212,255), width=1)
            for dy in [(0,-r-14),(0,-r+8),(0,r-8),(0,r+14)]:
                draw.line([cx,cy+dy[1]-6,cx,cy+dy[1]+6], fill=(0,212,255), width=1)
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

    def __del__(self):
        try: self.face_mesh.close()
        except: pass
