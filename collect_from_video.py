import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
import os
import shutil
import argparse

VIDEO_PATH = "videos"
DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30
POSE_KEYPOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FACE_KEYPOINT_IDS = [10, 9, 8, 6, 4, 1, 33, 263, 61, 291, 13, 14]
FEATURE_SIZE = ((len(POSE_KEYPOINT_IDS) + len(FACE_KEYPOINT_IDS) + 21 + 21) * 3)

HOLISTIC_MODEL_PATH = "holistic_landmarker.task"
if not os.path.exists(HOLISTIC_MODEL_PATH):
    import urllib.request
    _dl_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
    )
    print(f"⬇️  Downloading {HOLISTIC_MODEL_PATH} ...")
    urllib.request.urlretrieve(_dl_url, HOLISTIC_MODEL_PATH)
    print(f"✅ Downloaded {HOLISTIC_MODEL_PATH}")

_base_opts = _mp_tasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
_holistic_opts = _mp_vision.HolisticLandmarkerOptions(
    base_options=_base_opts,
    running_mode=_mp_vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_landmarks_confidence=0.7,
    min_hand_landmarks_confidence=0.7,
    min_face_detection_confidence=0.7,
    min_face_landmarks_confidence=0.7,
)
holistic = _mp_vision.HolisticLandmarker.create_from_options(_holistic_opts)


def extract_landmarks(results):
    data = []

    # Pose: upper-body focus
    if results.pose_landmarks:
        for idx in POSE_KEYPOINT_IDS:
            lm = results.pose_landmarks[idx]
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * (len(POSE_KEYPOINT_IDS) * 3))

    # Face: important forehead/eye/mouth refs
    if results.face_landmarks:
        for idx in FACE_KEYPOINT_IDS:
            lm = results.face_landmarks[idx]
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * (len(FACE_KEYPOINT_IDS) * 3))

    # Left hand: 21 * 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * 63)

    # Right hand: 21 * 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks:
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

parser = argparse.ArgumentParser(description="Convert gesture videos to sequence dataset")
parser.add_argument("--action", type=str, default="", help="Process only one action folder")
parser.add_argument("--video", type=str, default="", help="Process only one video filename inside action")
parser.add_argument("--reset", action="store_true", help="Clear existing numeric sequences before processing")
args = parser.parse_args()

os.makedirs(DATA_PATH, exist_ok=True)

all_actions = sorted(os.listdir(VIDEO_PATH))
if args.action:
    all_actions = [a for a in all_actions if a == args.action]
    if not all_actions:
        print(f"❌ Action not found in videos/: {args.action}")
        holistic.close()
        exit()

for action in all_actions:
    action_path = os.path.join(VIDEO_PATH, action)
    if not os.path.isdir(action_path):
        continue

    save_path = os.path.join(DATA_PATH, action)
    os.makedirs(save_path, exist_ok=True)

    # ล้าง sequence เดิมของ action นี้เมื่อระบุ --reset
    if args.reset:
        for name in os.listdir(save_path):
            full_path = os.path.join(save_path, name)
            if os.path.isdir(full_path) and name.isdigit():
                shutil.rmtree(full_path)

    print(f"\n📂 Action: {action}")

    candidate_videos = sorted(os.listdir(action_path))
    if args.video:
        candidate_videos = [v for v in candidate_videos if v == args.video]
        if not candidate_videos:
            print(f"⚠️ Video not found in {action}: {args.video}")
            continue

    for video_file in candidate_videos:
        video_path = os.path.join(action_path, video_file)
        if not os.path.isfile(video_path):
            continue

        print(f"🎬 Opening: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_file}")
            continue

        frames = []
        _ts_ms = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _ts_ms += 33
            _mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = holistic.detect_for_video(_mp_img, _ts_ms)
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