import numpy as np

# These are the MediaPipe landmark indices for the wrists
# MediaPipe gives you 33 landmarks total — each one is a body part
# Index 15 = left wrist, 16 = right wrist
LEFT_WRIST = 15
RIGHT_WRIST = 16

def get_wrist_positions(landmarks, dominant="right"):
    """
    Given a list of landmarks from one frame, return the (x, y) position
    of the dominant wrist as a numpy array.
    """
    idx = RIGHT_WRIST if dominant == "right" else LEFT_WRIST
    lm = landmarks[idx]
    return np.array([lm.x, lm.y])  # x and y are normalized 0-1 (% of frame width/height)


def detect_swings(frame_landmarks, fps=30, dominant="right", threshold=0.02, cooldown_frames=15):
    """
    Takes a list of landmark sets (one per frame) and returns a list of
    frame indices where a swing was detected.

    How it works:
    - For each frame, we get the wrist position (x, y)
    - We calculate how far the wrist moved since the last frame (velocity)
    - If velocity exceeds the threshold, that's a swing
    - Cooldown prevents the same swing from firing multiple times

    Args:
        frame_landmarks: list of landmark results, one per frame
        fps: frames per second of the video
        dominant: "right" or "left" handed player
        threshold: how fast the wrist needs to move to count as a swing
                   (in normalized units — 0.02 means 2% of frame width per frame)
        cooldown_frames: after a swing is detected, ignore the next N frames
                         so one swing doesn't get counted 5 times

    Returns:
        List of frame indices where swings were detected
    """
    positions = []

    # Step 1: Extract wrist position for every frame that has landmarks
    for landmarks in frame_landmarks:
        if landmarks is None:
            positions.append(None)  # no person detected in this frame
        else:
            positions.append(get_wrist_positions(landmarks, dominant))

    swing_frames = []
    cooldown = 0  # counts down after each detected swing

    # Step 2: Loop through every frame and compute velocity
    for i in range(1, len(positions)):  # start at 1 so we can look back at i-1
        if cooldown > 0:
            cooldown -= 1
            continue  # skip this frame, we're in the cooldown window

        prev = positions[i - 1]
        curr = positions[i]

        # Skip if either frame had no landmarks
        if prev is None or curr is None:
            continue

        # Velocity = euclidean distance the wrist moved between frames
        # np.linalg.norm = sqrt((x2-x1)^2 + (y2-y1)^2)
        velocity = np.linalg.norm(curr - prev)

        if velocity > threshold:
            swing_frames.append(i)
            cooldown = cooldown_frames  # start cooldown so we don't double-count

    return swing_frames


def extract_swing_windows(frame_landmarks, swing_frames, window_size=15):
    """
    For each detected swing frame, grab a window of frames around it.
    This gives us context before and after the swing peak — important
    for the classifier to understand the full motion.

    Example: swing at frame 100, window_size=15
    → returns frames 92 to 107 (7 before, 8 after)

    Args:
        frame_landmarks: full list of landmarks for every frame
        swing_frames: list of frame indices from detect_swings()
        window_size: total number of frames to capture per swing

    Returns:
        List of windows, each window is a list of ~window_size landmark sets
    """
    half = window_size // 2
    total_frames = len(frame_landmarks)
    windows = []

    for frame_idx in swing_frames:
        start = max(0, frame_idx - half)           # don't go below frame 0
        end = min(total_frames, frame_idx + half)  # don't go past last frame
        window = frame_landmarks[start:end]
        windows.append({
            "frame_idx": frame_idx,
            "window": window
        })

    return windows