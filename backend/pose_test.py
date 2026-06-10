"""OpenCV + MediaPipe pose overlay demo.

Uses the Tasks API (PoseLandmarker + VIDEO mode) for MediaPipe 0.10+ / Python 3.13
where ``mp.solutions`` is unavailable. Falls back to the legacy solutions API when
present.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Paths are anchored to this file's directory (backend/).
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR / "data"
_VIDEO_PATH = _DATA_DIR / "raw" / "test_clip.mp4"
_MODEL_FILENAME = "pose_landmarker_lite.task"
_MODEL_PATH = _DATA_DIR / _MODEL_FILENAME
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_lite/float16/1/{_MODEL_FILENAME}"
)

WINDOW_NAME = "RacketSense - Pose Test"

# BlazePose topology (mirrors mediapipe.python.solutions.pose_connections).
POSE_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ]
)


def _ensure_pose_task_model() -> Path:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _MODEL_PATH.is_file():
        return _MODEL_PATH
    print(f"Downloading pose model to {_MODEL_PATH} ...", flush=True)
    req = urllib.request.Request(
        _MODEL_URL,
        headers={"User-Agent": "RacketSense/pose_test"},
    )
    with urllib.request.urlopen(req) as resp, _MODEL_PATH.open("wb") as out:
        out.write(resp.read())
    return _MODEL_PATH


def _draw_tasks_pose(
    frame_bgr: np.ndarray,
    pose_landmarks: list | None,
) -> None:
    if not pose_landmarks:
        return
    h, w = frame_bgr.shape[:2]
    for pose in pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2, lineType=cv2.LINE_AA)
        for xy in pts:
            cv2.circle(frame_bgr, xy, 4, (0, 128, 255), -1, lineType=cv2.LINE_AA)


def _run_tasks_pose() -> None:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = _ensure_pose_task_model()
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )

    cap = cv2.VideoCapture(str(_VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {_VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delta_ms = max(int(round(1000.0 / fps)), 1) if fps and fps > 0 else 33

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_ts_ms = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)
            frame_ts_ms += frame_delta_ms

            _draw_tasks_pose(frame, result.pose_landmarks)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def _run_legacy_solutions_pose() -> None:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(str(_VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {_VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    solutions = getattr(mp, "solutions", None)
    if solutions is not None and getattr(solutions, "pose", None) is not None:
        _run_legacy_solutions_pose()
    else:
        _run_tasks_pose()


if __name__ == "__main__":
    main()
