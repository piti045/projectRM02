import argparse
import os
import re

import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

from feature_utils import LandmarkFeatureExtractor

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30
HOLISTIC_MODEL_PATH = "holistic_landmarker.task"
USE_TASK_HOLISTIC = hasattr(_mp_vision, "HolisticLandmarkerOptions")


def sanitize_action_name(raw_name):
    cleaned = raw_name.strip().lower().replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def list_action_dirs():
    if not os.path.exists(DATA_PATH):
        return []
    return sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])


def get_next_sequence_id(action_path):
    seq_ids = []
    for name in os.listdir(action_path):
        full_path = os.path.join(action_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            seq_ids.append(int(name))
    return (max(seq_ids) + 1) if seq_ids else 0


def build_holistic_detector():
    if USE_TASK_HOLISTIC:
        if not os.path.exists(HOLISTIC_MODEL_PATH):
            raise FileNotFoundError(f"Missing model file: {HOLISTIC_MODEL_PATH}")

        base_opts = _mp_tasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
        holistic_opts = _mp_vision.HolisticLandmarkerOptions(
            base_options=base_opts,
            running_mode=_mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_landmarks_confidence=0.7,
            min_hand_landmarks_confidence=0.7,
            min_face_detection_confidence=0.7,
            min_face_landmarks_confidence=0.7,
        )
        return _mp_vision.HolisticLandmarker.create_from_options(holistic_opts)

    print("⚠️ HolisticLandmarker Tasks API not available, falling back to mp.solutions.holistic")
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )


def detect_holistic(holistic, rgb_frame, timestamp_ms):
    if USE_TASK_HOLISTIC:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return holistic.detect_for_video(mp_img, timestamp_ms)
    return holistic.process(rgb_frame)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect webcam sequences for gesture classes")
    parser.add_argument(
        "--actions",
        type=str,
        default="",
        help="Comma-separated class names (ex: hello,thank_you,how_are_you). If empty, use existing dataset folders.",
    )
    parser.add_argument("--sequences", type=int, default=5, help="Sequences per action")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.actions.strip():
        actions = [sanitize_action_name(x) for x in args.actions.split(",")]
        actions = [x for x in actions if x]
    else:
        actions = list_action_dirs()

    if not actions:
        print("❌ No actions found. Use --actions to define classes first")
        return

    os.makedirs(DATA_PATH, exist_ok=True)
    holistic = build_holistic_detector()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้อง")
        holistic.close()
        return

    print("✅ Ready to collect")
    print(f"📝 Actions: {actions}")
    print(f"📝 Sequences per action: {args.sequences}")
    print(f"📝 Frames per sequence: {SEQUENCE_LENGTH}")

    ts_ms = 0

    try:
        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            os.makedirs(action_path, exist_ok=True)

            for _ in range(args.sequences):
                seq_id = get_next_sequence_id(action_path)
                seq_path = os.path.join(action_path, str(seq_id))
                os.makedirs(seq_path, exist_ok=True)
                feature_extractor = LandmarkFeatureExtractor()

                print(f"\n🎥 Recording: {action}/{seq_id}")
                frame_idx = 0

                while frame_idx < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    ts_ms += 33
                    results = detect_holistic(holistic, rgb, ts_ms)
                    keypoints = feature_extractor.extract(results)

                    np.save(os.path.join(seq_path, f"{frame_idx}.npy"), keypoints)

                    cv2.putText(
                        frame,
                        f"{action}/{seq_id} frame {frame_idx + 1}/{SEQUENCE_LENGTH}",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        "Press q to quit | Esc to skip sequence",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
                    cv2.imshow("Collecting Data", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("\n❌ Quitting...")
                        return
                    if key == 27:
                        print(f"⏭️ Skipped sequence {action}/{seq_id}")
                        break

                    frame_idx += 1
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()

    print("\n✅ เก็บข้อมูลเสร็จแล้ว")


if __name__ == "__main__":
    main()
