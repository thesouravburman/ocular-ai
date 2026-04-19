"""
Real-time pupil tracking using MediaPipe Face Mesh.
"""

import cv2
import mediapipe as mp
import numpy as np

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

class PupilTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def track(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return {}
        h, w = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark

        def get_iris_data(indices):
            pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices], dtype=np.float32)
            center = pts.mean(axis=0)
            radius = np.linalg.norm(pts - center, axis=1).mean()
            return tuple(center.astype(int)), float(radius)

        left_center, left_radius   = get_iris_data(LEFT_IRIS)
        right_center, right_radius = get_iris_data(RIGHT_IRIS)
        return {
            "left_center":  left_center,
            "right_center": right_center,
            "left_radius":  left_radius,
            "right_radius": right_radius,
        }

    def __del__(self):
        self.face_mesh.close()
