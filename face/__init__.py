"""
face/__init__.py
================
Face extraction and quality estimation module (Track P1).

Provides:
    - FaceDetector         : MediaPipe-based face detection + mesh landmarks
    - DetectedFace         : Data class for a single detected face
    - FacePose             : Head pose angles (yaw, pitch, roll)
    - FaceFeatureExtractor : 512-d face embedding extraction (ArcFace / fallback)
    - estimate_face_quality: Composite quality scoring with fusion weight advice
"""

from face.detector import FaceDetector, DetectedFace, FacePose, align_face_5points, verify_anatomical_face_geometry
from face.quality import estimate_face_quality
from face.extractor import FaceFeatureExtractor, cosine_similarity
from face.multiview import (
    synthesize_3d_perspective_view,
    synthesize_multiview_gallery,
    MultiViewGalleryItem,
    MultiViewEnrollmentEngine,
)


def __getattr__(name: str):
    if name == "VideoSearchEngine":
        from face.video_search import VideoSearchEngine
        return VideoSearchEngine
    if name in ("StudentCCTVFaceNet", "SurveillanceDegradation"):
        from face.distillation import StudentCCTVFaceNet, SurveillanceDegradation
        return StudentCCTVFaceNet if name == "StudentCCTVFaceNet" else SurveillanceDegradation
    raise AttributeError(f"module 'face' has no attribute '{name}'")


__all__ = [
    "FaceDetector",
    "DetectedFace",
    "FacePose",
    "align_face_5points",
    "verify_anatomical_face_geometry",
    "FaceFeatureExtractor",
    "cosine_similarity",
    "estimate_face_quality",
    "synthesize_3d_perspective_view",
    "synthesize_multiview_gallery",
    "MultiViewGalleryItem",
    "MultiViewEnrollmentEngine",
    "VideoSearchEngine",
    "StudentCCTVFaceNet",
    "SurveillanceDegradation",
]



