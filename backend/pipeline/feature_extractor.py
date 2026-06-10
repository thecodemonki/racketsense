import numpy as np

# MediaPipe landmark indices we care about
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_KNEE      = 25
RIGHT_KNEE     = 26

def angle_between(a, b, c):
    """
    Calculate the angle at point B formed by points A, B, C.
    This is how we measure joint angles — elbow angle, wrist angle, etc.

    Think of it like: A = shoulder, B = elbow, C = wrist
    → gives you the bend angle at the elbow

    Returns angle in degrees (0-180)
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b  # vector from B to A
    bc = c - b  # vector from B to C

    # Dot product formula to get angle between two vectors
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)  # clamp to avoid floating point errors in arccos
    return np.degrees(np.arccos(cosine))


def get_landmark_xy(landmarks, idx):
    """Helper to grab (x, y) from a landmark by index."""
    lm = landmarks[idx]
    return [lm.x, lm.y]


def extract_features(window_landmarks, dominant="right"):
    """
    Takes a window of ~15 frames of landmarks and returns a single
    feature vector of 9 numbers representing the swing.

    This is what the Random Forest classifier will learn from.
    Each number describes one aspect of the body position/movement.

    Args:
        window_landmarks: list of landmark sets (~15 frames)
        dominant: "right" or "left" handed

    Returns:
        numpy array of 9 features, or None if not enough data
    """
    # Filter out frames with no landmarks
    valid = [lm for lm in window_landmarks if lm is not None]
    if len(valid) < 5:
        return None  # not enough frames to extract meaningful features

    # Pick the correct side based on dominant hand
    shoulder_idx = RIGHT_SHOULDER if dominant == "right" else LEFT_SHOULDER
    elbow_idx    = RIGHT_ELBOW    if dominant == "right" else LEFT_ELBOW
    wrist_idx    = RIGHT_WRIST    if dominant == "right" else LEFT_WRIST
    hip_idx      = RIGHT_HIP      if dominant == "right" else LEFT_HIP
    knee_idx     = RIGHT_KNEE     if dominant == "right" else LEFT_KNEE
    opp_shoulder = LEFT_SHOULDER  if dominant == "right" else RIGHT_SHOULDER

    # --- Feature 1: Elbow angle at peak frame ---
    # We use the middle frame as the "peak" of the swing
    peak = valid[len(valid) // 2]
    shoulder_pos = get_landmark_xy(peak, shoulder_idx)
    elbow_pos    = get_landmark_xy(peak, elbow_idx)
    wrist_pos    = get_landmark_xy(peak, wrist_idx)
    elbow_angle  = angle_between(shoulder_pos, elbow_pos, wrist_pos)

    # --- Feature 2: Wrist height relative to shoulder ---
    # A smash = wrist way above shoulder, net kill = wrist at net height
    # Negative = wrist is above shoulder (y=0 is top of frame)
    wrist_height_ratio = peak[wrist_idx].y - peak[shoulder_idx].y

    # --- Feature 3: Body lean (left/right) ---
    # Compare hip center x position vs shoulder center x position
    hip_x      = (peak[LEFT_HIP].x + peak[RIGHT_HIP].x) / 2
    shoulder_x = (peak[LEFT_SHOULDER].x + peak[RIGHT_SHOULDER].x) / 2
    body_lean_x = shoulder_x - hip_x  # positive = leaning right

    # --- Feature 4: Forward/backward lean ---
    hip_y      = (peak[LEFT_HIP].y + peak[RIGHT_HIP].y) / 2
    shoulder_y = (peak[LEFT_SHOULDER].y + peak[RIGHT_SHOULDER].y) / 2
    body_lean_y = shoulder_y - hip_y  # how far shoulder is from hip vertically

    # --- Feature 5: Knee bend ---
    hip_pos  = get_landmark_xy(peak, hip_idx)
    knee_pos = get_landmark_xy(peak, knee_idx)
    # We approximate ankle as slightly below knee for angle calc
    ankle_approx = [knee_pos[0], knee_pos[1] + 0.1]
    knee_angle = angle_between(hip_pos, knee_pos, ankle_approx)

    # --- Feature 6: Wrist velocity peak ---
    # Max speed the wrist reached across the whole window
    wrist_positions = []
    for lm in valid:
        wrist_positions.append(np.array([lm[wrist_idx].x, lm[wrist_idx].y]))

    velocities = []
    for i in range(1, len(wrist_positions)):
        v = np.linalg.norm(wrist_positions[i] - wrist_positions[i-1])
        velocities.append(v)
    wrist_velocity_peak = max(velocities) if velocities else 0.0

    # --- Feature 7: Wrist trajectory direction ---
    # Is the wrist moving upward or downward overall?
    # Compare wrist y at start vs end of window (y increases downward in image coords)
    wrist_y_start = valid[0][wrist_idx].y
    wrist_y_end   = valid[-1][wrist_idx].y
    wrist_trajectory = wrist_y_end - wrist_y_start
    # Negative = wrist moved up (smash/clear), Positive = wrist moved down (drop/net)

    # --- Feature 8: Shoulder rotation ---
    # How much are the shoulders rotated (facing camera vs side-on)?
    shoulder_width = abs(peak[LEFT_SHOULDER].x - peak[RIGHT_SHOULDER].x)
    # Wide = facing camera, narrow = side-on (like a smash wind-up)

    # --- Feature 9: Wrist x trajectory (left/right movement) ---
    wrist_x_start = valid[0][wrist_idx].x
    wrist_x_end   = valid[-1][wrist_idx].x
    wrist_x_traj  = wrist_x_end - wrist_x_start

    # Stack all 9 features into one array
    features = np.array([
        elbow_angle,          # 1. bend at elbow
        wrist_height_ratio,   # 2. how high wrist is vs shoulder
        body_lean_x,          # 3. left/right lean
        body_lean_y,          # 4. forward/back lean
        knee_angle,           # 5. how bent the knees are
        wrist_velocity_peak,  # 6. max wrist speed
        wrist_trajectory,     # 7. up/down wrist movement
        shoulder_width,       # 8. shoulder rotation prox