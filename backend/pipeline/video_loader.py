import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import urllib.request

# ── Model bootstrap (same logic as pose_test.py) ──────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = Path(__file__).parent.parent / "data" / "pose_landmarker_lite.task"

def _ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading pose model (~7 MB)...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

# ── Landmark extraction ────────────────────────────────────────────────────
def load_video_landmarks(video_path: str, show_preview: bool = False):
    """
    Opens a video file and runs MediaPipe pose estimation on every frame.

    Returns:
        landmarks_per_frame : list[list | None]  — one entry per frame
        fps                 : float
        total_frames        : int

    Each entry is either:
        - a list of 33 NormalizedLandmark objects  (person detected)
        - None                                      (no person in frame)
    """
    _ensure_model()

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.1f} fps")

    # ── Build PoseLandmarker (VIDEO mode) ─────────────────────────────────
    BaseOptions       = mp.tasks.BaseOptions
    PoseLandmarker    = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode       = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarks_per_frame = []
    frame_idx           = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR → RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp must increase every frame (milliseconds)
            timestamp_ms = int(frame_idx * (1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks_per_frame.append(result.pose_landmarks[0])  # first person
            else:
                landmarks_per_frame.append(None)

            # Optional live preview while processing
            if show_preview:
                cv2.imshow("Processing...", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  processed {frame_idx}/{total_frames} frames")

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    print(f"Done. Landmarks extracted for {sum(l is not None for l in landmarks_per_frame)}/{total_frames} frames.")
    return landmarks_per_frame, fps, total_frames