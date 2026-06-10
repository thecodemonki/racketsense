from pipeline.video_loader import load_video_landmarks
from pipeline.swing_detector import detect_swings, extract_swing_windows
from pipeline.feature_extractor import extract_features_batch
from pathlib import Path

VIDEO = str(Path(__file__).parent / "data/raw/test_clip.mp4")

print("=== Step 1: Loading landmarks ===")
landmarks, fps, total_frames = load_video_landmarks(VIDEO)

print("\n=== Step 2: Detecting swings ===")
swing_frames = detect_swings(landmarks, fps=fps)
print(f"Found {len(swing_frames)} swings at frames: {swing_frames}")

print("\n=== Step 3: Extracting windows ===")
windows = extract_swing_windows(landmarks, swing_frames)
print(f"Extracted {len(windows)} swing windows")

print("\n=== Step 4: Extracting features ===")
features = extract_features_batch(windows)
print(f"Got {len(features)} feature vectors")

for f in features:
    if f["features"] is not None:
        print(f"  Frame {f['frame_idx']}: {f['features'].round(3)}")