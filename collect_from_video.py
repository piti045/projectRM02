import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
import os
import shutil
import argparse
import json

from feature_utils import FEATURE_SIZE, LandmarkFeatureExtractor

VIDEO_PATH = "videos"
DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30
GESTURE_META_PATH = os.path.join(DATA_PATH, "gestures.json")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
USE_TASK_HOLISTIC = hasattr(_mp_vision, "HolisticLandmarkerOptions")

HOLISTIC_MODEL_PATH = "holistic_landmarker.task"
if USE_TASK_HOLISTIC and not os.path.exists(HOLISTIC_MODEL_PATH):
    import urllib.request
    _dl_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
    )
    print(f"⬇️  Downloading {HOLISTIC_MODEL_PATH} ...")
    urllib.request.urlretrieve(_dl_url, HOLISTIC_MODEL_PATH)
    print(f"✅ Downloaded {HOLISTIC_MODEL_PATH}")

if USE_TASK_HOLISTIC:
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
else:
    print("⚠️ HolisticLandmarker Tasks API not available, falling back to mp.solutions.holistic")
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )


def detect_holistic(rgb_frame, timestamp_ms):
    if USE_TASK_HOLISTIC:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return holistic.detect_for_video(mp_image, timestamp_ms)
    return holistic.process(rgb_frame)


def get_next_sequence_id(action_path):
    seq_ids = []
    for name in os.listdir(action_path):
        full_path = os.path.join(action_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            seq_ids.append(int(name))
    return (max(seq_ids) + 1) if seq_ids else 0


def save_gesture_metadata(actions):
    meta = {action_name: action_name for action_name in sorted(actions)}
    with open(GESTURE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def clear_action_sequences(action_dataset_path):
    for name in os.listdir(action_dataset_path):
        full_path = os.path.join(action_dataset_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            shutil.rmtree(full_path)


def remove_top_level_npy(action_dataset_path):
    for name in os.listdir(action_dataset_path):
        full_path = os.path.join(action_dataset_path, name)
        if os.path.isfile(full_path) and name.endswith(".npy"):
            os.remove(full_path)


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
parser.add_argument(
    "--replace-all",
    action="store_true",
    help="Mirror dataset classes to videos classes (remove stale class folders and rebuild sequences)",
)
args = parser.parse_args()

os.makedirs(DATA_PATH, exist_ok=True)

all_actions = sorted(
    [d for d in os.listdir(VIDEO_PATH) if os.path.isdir(os.path.join(VIDEO_PATH, d))]
)
if args.action:
    all_actions = [a for a in all_actions if a == args.action]
    if not all_actions:
        print(f"❌ Action not found in videos/: {args.action}")
        holistic.close()
        exit()

if args.replace_all and not args.action:
    existing_actions = [
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ]
    stale_actions = sorted(set(existing_actions) - set(all_actions))
    for stale_action in stale_actions:
        stale_path = os.path.join(DATA_PATH, stale_action)
        shutil.rmtree(stale_path)
        print(f"🧹 Removed stale class folder: {stale_action}")

for action in all_actions:
    action_path = os.path.join(VIDEO_PATH, action)
    if not os.path.isdir(action_path):
        continue

    save_path = os.path.join(DATA_PATH, action)
    os.makedirs(save_path, exist_ok=True)

    # ล้าง sequence เดิมของ action นี้เมื่อระบุ --reset
    if args.reset or args.replace_all:
        clear_action_sequences(save_path)
        remove_top_level_npy(save_path)

    print(f"\n📂 Action: {action}")

    candidate_videos = sorted(
        [
            v for v in os.listdir(action_path)
            if os.path.isfile(os.path.join(action_path, v))
            and v.lower().endswith(VIDEO_EXTENSIONS)
        ]
    )
    if args.video:
        candidate_videos = [v for v in candidate_videos if v == args.video]
        if not candidate_videos:
            print(f"⚠️ Video not found in {action}: {args.video}")
            continue

    for video_file in candidate_videos:
        video_path = os.path.join(action_path, video_file)

        print(f"🎬 Opening: {video_path}")

        cap = cv2.VideoCapture(video_path)
        feature_extractor = LandmarkFeatureExtractor()

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
            results = detect_holistic(rgb, _ts_ms)
            landmarks = feature_extractor.extract(results)
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

save_gesture_metadata(all_actions)

holistic.close()
print("\n✅ Done converting videos to sequence dataset")