import cv2
import mediapipe as mp
import numpy as np
import os

VIDEO_PATH = "videos"
DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 375

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def extract_landmarks(results):
    data = []

    # Pose: 33 * 3 = 99
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * 99)

    # Face: first 50 * 3 = 150
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark[:50]:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * 150)

    # Left hand: 21 * 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * 63)

    # Right hand: 21 * 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * 63)

    arr = np.array(data)
    if arr.shape[0] != FEATURE_SIZE:
        raise ValueError(f"Feature size mismatch: {arr.shape[0]}")

    return arr


def get_next_sequence_id(action_path):
    seq_ids = []
    for name in os.listdir(action_path):
        full_path = os.path.join(action_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            seq_ids.append(int(name))
    return (max(seq_ids) + 1) if seq_ids else 0


def save_sequence(action_path, seq_id, sequence_frames):
    seq_path = os.path.join(action_path, str(seq_id))
    os.makedirs(seq_path, exist_ok=True)

    for idx, frame_data in enumerate(sequence_frames):
        np.save(os.path.join(seq_path, f"{idx}.npy"), frame_data)

    print(f"✅ Saved sequence: {os.path.basename(action_path)}/{seq_id}")


if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video folder not found: {VIDEO_PATH}")
    exit()

os.makedirs(DATA_PATH, exist_ok=True)

for action in sorted(os.listdir(VIDEO_PATH)):
    action_path = os.path.join(VIDEO_PATH, action)
    if not os.path.isdir(action_path):
        continue

    save_path = os.path.join(DATA_PATH, action)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n📂 Action: {action}")

    for video_file in sorted(os.listdir(action_path)):
        video_path = os.path.join(action_path, video_file)
        if not os.path.isfile(video_path):
            continue

        print(f"🎬 Opening: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_file}")
            continue

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(rgb)
            landmarks = extract_landmarks(results)
            frames.append(landmarks)

        cap.release()
        print(f"🧠 Total frames: {len(frames)}")

        if len(frames) < SEQUENCE_LENGTH:
            print(f"⚠ Not enough frames, skipped: {video_file}")
            continue

        seq_id = get_next_sequence_id(save_path)
        for i in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
            seq = frames[i:i + SEQUENCE_LENGTH]
            save_sequence(save_path, seq_id, seq)
            seq_id += 1

        print(f"✔ Finished: {video_file}")

holistic.close()
print("\n✅ Done converting videos to sequence dataset")