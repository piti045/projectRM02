from flask import Flask, render_template, Response, request, jsonify
import json
import os
import re
import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.utils import to_categorical
import time
import threading
from collections import deque

app = Flask(__name__)

# ============================================
# 1. โหลด AI Model สำหรับการแปลภาษามือ
# ============================================
try:
    model = load_model("model.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# ============================================
# 2. กำหนดคำศัพท์ที่ต้องรู้จำ (อ่านจากโฟลเดอร์ dataset)
# ============================================
# สำคัญ: ต้องตรงกับตอนเทรน ถ้าเพิ่ม/ลบคำให้แก้ dataset แล้วเทรนใหม่
DATA_PATH = "dataset"

# รายชื่อโฟลเดอร์ภายใน dataset เป็นคำศัพท์
if os.path.exists(DATA_PATH):
    actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
else:
    actions = []

print(f"📝 Actions loaded: {actions}")

# ============================================
# Constants
# ============================================
SEQUENCE_LENGTH = 30         # จำนวนเฟรมต่อหนึ่งลำดับ
CONSENSUS_WINDOW = 5         # หน้าต่างสำหรับ smoothing การทำนาย
DISPLAY_CONFIDENCE_THRESHOLD = 0.20   # ค่าความมั่นใจขั้นต่ำสำหรับแสดงคำบนจอ
SENTENCE_CONFIDENCE_THRESHOLD = 0.35  # ค่าความมั่นใจขั้นต่ำสำหรับยืนยันเข้าประโยค
PREDICTION_MARGIN_THRESHOLD = 0.10    # ส่วนต่าง top1-top2 ขั้นต่ำ เพื่อลดคำทับซ้อน
LOW_SAMPLE_SUPPORT_THRESHOLD = 6      # ถ้าคลาสมี sequence น้อยกว่านี้ จะเพิ่มความเข้มงวด
LOW_SAMPLE_CONFIDENCE_BONUS = 0.10    # เพิ่ม threshold สำหรับคลาสข้อมูลน้อย
# โหมดใหม่: โฟกัสครึ่งตัว (คอ-เอว-แขน-มือ) + จุดหน้าผาก/ศีรษะสำหรับท่ามือเหนือหัว
POSE_KEYPOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # nose + shoulders/elbows/wrists/hips
FACE_KEYPOINT_IDS = [10, 9, 8, 6, 4, 1, 33, 263, 61, 291, 13, 14]  # รวม forehead + eye/mouth refs
FEATURE_SIZE = ((len(POSE_KEYPOINT_IDS) + len(FACE_KEYPOINT_IDS) + 21 + 21) * 3)
MODEL_PATH = "model.h5"
GESTURE_META_PATH = os.path.join(DATA_PATH, "gestures.json")
MODEL_LABELS_PATH = os.path.join(DATA_PATH, "model_labels.json")

# ============================================
# Global State Variables
# ============================================
state_lock = threading.Lock()
gesture_meanings = {}
model_needs_retrain = False
model_training = False
model_training_error = ""
model_actions = []
model_output_dim = 0
model_last_trained = None
expected_features = None
action_sample_counts = {}

# ============================================
# 3. โหลด MediaPipe Holistic (รู้จำใบหน้า, คอ, แขน, ตัว)
# ============================================
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

_holistic_base_opts = _mp_tasks.BaseOptions(model_asset_path=HOLISTIC_MODEL_PATH)
_holistic_opts = _mp_vision.HolisticLandmarkerOptions(
    base_options=_holistic_base_opts,
    running_mode=_mp_vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_landmarks_confidence=0.7,
    min_hand_landmarks_confidence=0.7,
    min_face_detection_confidence=0.7,
    min_face_landmarks_confidence=0.7,
)
holistic = _mp_vision.HolisticLandmarker.create_from_options(_holistic_opts)

# ===== Drawing helpers (OpenCV-only, no mediapipe drawing_utils needed) =====
_HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
])
_POSE_CONNECTIONS_UPPER = frozenset([
    (0, 11), (0, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
])


def _draw_landmarks(frame, landmarks, connections,
                    dot_color=(0, 255, 0), line_color=(200, 200, 200)):
    """Draw landmarks and connections with plain OpenCV (no mediapipe drawing_utils)."""
    if not landmarks:
        return
    h, w = frame.shape[:2]
    if connections:
        for s, e in connections:
            if s < len(landmarks) and e < len(landmarks):
                x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
                x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 1)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, dot_color, -1)

# ============================================
# 4. เปิดกล้อง (ใช้ CAP_DSHOW เพื่อประสิทธิภาพบน Windows)
# ============================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)      # ความกว้าง
cap.set(4, 480)      # ความสูง
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลดบัฟเฟอร์เพื่อลดความหน่วง

if not cap.isOpened():
    print("❌ เปิดกล้องไม่ได้")
    exit()

frame_global = None
prediction = ""
prediction_confidence = 0.0
sequence = deque(maxlen=SEQUENCE_LENGTH)
sentence = []
prediction_history = deque(maxlen=CONSENSUS_WINDOW)  # items: (word, conf)
last_pred_time = 0
last_word_time = time.time()
frame_count = 0
prediction_timeout = 2

recording_state = {
    "active": False,
    "action": "",
    "buffer": [],
    "last_saved": None,
    "error": None,
}


def sanitize_action_name(raw_name):
    cleaned = raw_name.strip().lower().replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def list_action_dirs():
    if not os.path.exists(DATA_PATH):
        return []
    return sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])


def save_gesture_meanings():
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(GESTURE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(gesture_meanings, f, ensure_ascii=False, indent=2)


def load_gesture_meanings():
    global gesture_meanings

    os.makedirs(DATA_PATH, exist_ok=True)
    meta = {}

    if os.path.exists(GESTURE_META_PATH):
        try:
            with open(GESTURE_META_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta = {str(k): str(v) for k, v in loaded.items()}
        except Exception as exc:
            print(f"⚠️ Failed to load gesture metadata: {exc}")

    current_actions = list_action_dirs()
    changed = False

    for action_name in current_actions:
        if action_name not in meta:
            meta[action_name] = action_name
            changed = True

    stale_keys = [k for k in meta if k not in current_actions]
    for stale_key in stale_keys:
        meta.pop(stale_key, None)
        changed = True

    gesture_meanings = meta

    if changed or not os.path.exists(GESTURE_META_PATH):
        save_gesture_meanings()


def save_model_labels(labels):
    with open(MODEL_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def load_model_labels(expected_len):
    if not os.path.exists(MODEL_LABELS_PATH):
        return None
    try:
        with open(MODEL_LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and len(labels) == expected_len:
            return [str(x) for x in labels]
    except Exception as exc:
        print(f"[WARN] Failed to load model labels: {exc}")
    return None


def count_valid_sequences(action_name):
    action_path = os.path.join(DATA_PATH, action_name)
    if not os.path.isdir(action_path):
        return 0

    count = 0
    for seq_name in os.listdir(action_path):
        seq_path = os.path.join(action_path, seq_name)
        if not os.path.isdir(seq_path):
            continue

        frame_files = [f for f in os.listdir(seq_path) if f.endswith(".npy")]
        if len(frame_files) == SEQUENCE_LENGTH:
            count += 1

    return count


def refresh_runtime_config():
    global actions, model_needs_retrain, action_sample_counts

    actions = list_action_dirs()
    action_sample_counts = {a: count_valid_sequences(a) for a in actions}
    load_gesture_meanings()

    low_support_actions = [a for a, c in action_sample_counts.items() if c < LOW_SAMPLE_SUPPORT_THRESHOLD]
    if low_support_actions:
        print(
            "[WARN] Low-support classes detected:",
            {a: action_sample_counts[a] for a in low_support_actions},
        )

    if model is None:
        model_needs_retrain = len(actions) > 0
        return

    model_needs_retrain = set(actions) != set(model_actions)


def load_runtime_model():
    global model, model_output_dim, model_actions, model_last_trained, expected_features

    model = None
    model_output_dim = 0
    model_actions = []
    model_last_trained = None
    expected_features = None

    if not os.path.exists(MODEL_PATH):
        print("⚠️ model.h5 not found, waiting for training")
        refresh_runtime_config()
        return

    try:
        loaded_model = load_model(MODEL_PATH)
        output_dim = int(loaded_model.output_shape[-1])
        labels = load_model_labels(output_dim)

        if labels is None:
            fallback_actions = list_action_dirs()
            if len(fallback_actions) == output_dim:
                labels = fallback_actions
            else:
                labels = fallback_actions[:output_dim]
                print("⚠️ model labels missing and size mismatch with dataset")

        model = loaded_model
        model_output_dim = output_dim
        model_actions = labels
        model_last_trained = time.strftime("%Y-%m-%d %H:%M:%S")
        expected_features = int(loaded_model.input_shape[-1])

        print(f"[OK] Model loaded successfully ({model_output_dim} classes)")
        print(f"[INFO] Model actions: {model_actions}")
    except Exception as exc:
        model = None
        model_output_dim = 0
        model_actions = []
        print(f"[ERROR] Failed to load model: {exc}")

    refresh_runtime_config()


def frame_sort_key(filename):
    stem, _ = os.path.splitext(filename)
    return int(stem) if stem.isdigit() else 10**9


def get_next_sequence_id(action_name):
    action_path = os.path.join(DATA_PATH, action_name)
    os.makedirs(action_path, exist_ok=True)

    ids = []
    for item in os.listdir(action_path):
        full_path = os.path.join(action_path, item)
        if os.path.isdir(full_path) and item.isdigit():
            ids.append(int(item))

    return (max(ids) + 1) if ids else 0


def save_recorded_sequence(action_name, frames):
    if len(frames) < SEQUENCE_LENGTH:
        raise ValueError(f"Need at least {SEQUENCE_LENGTH} frames")

    seq_id = get_next_sequence_id(action_name)
    seq_path = os.path.join(DATA_PATH, action_name, str(seq_id))
    os.makedirs(seq_path, exist_ok=True)

    for frame_idx, frame_data in enumerate(frames[:SEQUENCE_LENGTH]):
        arr = np.asarray(frame_data).reshape(-1)
        if arr.shape[0] != FEATURE_SIZE:
            raise ValueError(f"Feature size mismatch: expected {FEATURE_SIZE}, got {arr.shape[0]}")
        np.save(os.path.join(seq_path, f"{frame_idx}.npy"), arr)

    return seq_id


def summarize_dataset():
    summary = []
    for action_name in list_action_dirs():
        action_path = os.path.join(DATA_PATH, action_name)
        sample_count = 0

        for seq_name in os.listdir(action_path):
            seq_path = os.path.join(action_path, seq_name)
            if not os.path.isdir(seq_path):
                continue
            frame_files = [f for f in os.listdir(seq_path) if f.endswith(".npy")]
            if len(frame_files) == SEQUENCE_LENGTH:
                sample_count += 1

        summary.append(
            {
                "action": action_name,
                "meaning": gesture_meanings.get(action_name, action_name),
                "samples": sample_count,
            }
        )

    return summary


def load_training_samples(train_actions):
    sequences, labels = [], []
    skipped = []
    label_map = {label: idx for idx, label in enumerate(train_actions)}

    for action_name in train_actions:
        action_path = os.path.join(DATA_PATH, action_name)
        seq_dirs = sorted(
            [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
        )

        for seq_dir in seq_dirs:
            seq_path = os.path.join(action_path, seq_dir)
            frame_files = sorted(
                [f for f in os.listdir(seq_path) if f.endswith(".npy")],
                key=frame_sort_key,
            )

            if len(frame_files) != SEQUENCE_LENGTH:
                skipped.append(f"{action_name}/{seq_dir}: {len(frame_files)} frames")
                continue

            frames = []
            valid = True
            for frame_file in frame_files:
                arr = np.load(os.path.join(seq_path, frame_file)).reshape(-1)
                if arr.shape[0] != FEATURE_SIZE:
                    skipped.append(f"{action_name}/{seq_dir}: feature {arr.shape[0]}")
                    valid = False
                    break
                frames.append(arr)

            if not valid:
                continue

            sequences.append(np.array(frames))
            labels.append(label_map[action_name])

    return np.array(sequences), np.array(labels), skipped


def train_model_job():
    global model, model_output_dim, model_actions, model_needs_retrain
    global model_training, model_training_error, model_last_trained

    with state_lock:
        model_training = True
        model_training_error = ""

    try:
        train_actions = list_action_dirs()
        if len(train_actions) < 2:
            raise ValueError("Need at least 2 gestures before training")

        X, y, skipped = load_training_samples(train_actions)
        print(f"[LOAD] Training samples loaded: {X.shape[0]}")
        if skipped:
            print(f"[WARN] Skipped sequences: {len(skipped)}")

        if X.shape[0] < len(train_actions) * 2:
            raise ValueError("Not enough sequences. Record at least 2 sequences per gesture.")

        y_one_hot = to_categorical(y, num_classes=len(train_actions)).astype(int)

        class_counts = np.bincount(y, minlength=len(train_actions))
        can_stratify = bool(np.all(class_counts >= 2))

        if can_stratify and X.shape[0] >= len(train_actions) * 3:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_one_hot,
                test_size=0.2,
                shuffle=True,
                stratify=y,
                random_state=42,
            )
            validation_data = (X_test, y_test)
            callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
            print(f"[OK] Validation enabled. Train={X_train.shape[0]}, Val={X_test.shape[0]}")
        else:
            X_train, y_train = X, y_one_hot
            validation_data = None
            callbacks = []
            print("[WARN] Validation skipped due to limited data")

        new_model = Sequential(
            [
                LSTM(128, return_sequences=True, activation="tanh", input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
                Dropout(0.3),
                LSTM(128, return_sequences=False, activation="tanh"),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dense(len(train_actions), activation="softmax"),
            ]
        )

        new_model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"],
        )

        fit_kwargs = {
            "x": X_train,
            "y": y_train,
            "epochs": 40,
            "batch_size": 16,
            "callbacks": callbacks,
            "verbose": 1,
        }
        if validation_data is not None:
            fit_kwargs["validation_data"] = validation_data

        new_model.fit(**fit_kwargs)
        new_model.save(MODEL_PATH)
        save_model_labels(train_actions)

        loaded_model = load_model(MODEL_PATH)
        with state_lock:
            model = loaded_model
            model_output_dim = len(train_actions)
            model_actions = train_actions
            model_last_trained = time.strftime("%Y-%m-%d %H:%M:%S")
            model_needs_retrain = False

        refresh_runtime_config()
        print("[OK] Training complete and model reloaded")

    except Exception as exc:
        with state_lock:
            model_training_error = str(exc)
        print(f"[ERROR] Training failed: {exc}")
    finally:
        with state_lock:
            model_training = False

def extract_keypoints(results, face_landmarks_to_use=50):
    """
    ========================================
    แตกเอาจุด keypoints จากผลการตรวจจับ
    ========================================
    - Pose: 33 จุด × 3 (x, y, z) = 99 ค่า
    - Face: variable (def. 50) × 3 ค่า
    - Left Hand: 21 จุด × 3 (x, y, z) = 63 ค่า
    - Right Hand: 21 จุด × 3 (x, y, z) = 63 ค่า
    """
    data = []

    # ===== เก็บจุด Pose แบบโฟกัสครึ่งตัว =====
    if results.pose_landmarks:
        for idx in POSE_KEYPOINT_IDS:
            lm = results.pose_landmarks[idx]
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * (len(POSE_KEYPOINT_IDS) * 3))

    # ===== เก็บจุด Face เฉพาะสำคัญ (รองรับท่ามือแตะหน้าผาก/เหนือหัว) =====
    face_features = len(FACE_KEYPOINT_IDS) * 3
    if results.face_landmarks:
        for idx in FACE_KEYPOINT_IDS:
            lm = results.face_landmarks[idx]
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0] * face_features)

    # ===== เก็บจุด Left Hand (มือซ้าย 21 จุด) =====
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*63)

    # ===== เก็บจุด Right Hand (มือขวา 21 จุด) =====
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*63)

    return np.array(data)


def is_upper_body_ready(results):
    """เช็กว่าจุดคอ-ไหล่-เอวมีพอสำหรับโหมดครึ่งตัว"""
    if not results.pose_landmarks:
        return False

    pose = results.pose_landmarks
    required = [11, 12, 23, 24]
    for idx in required:
        vis = getattr(pose[idx], 'visibility', None)
        if vis is not None and vis < 0.35:
            return False
    return True


def is_head_level_gesture(results):
    """รองรับท่าที่มือขึ้นเหนือหัวหรือแตะหน้าผาก (เช่น สวัสดี)"""
    if not results.pose_landmarks:
        return False

    pose = results.pose_landmarks
    nose_y = pose[0].y
    left_wrist_y = pose[15].y
    right_wrist_y = pose[16].y

    # มืออยู่ระดับศีรษะหรือสูงกว่า
    above_head = (left_wrist_y < nose_y + 0.04) or (right_wrist_y < nose_y + 0.04)
    if above_head:
        return True

    # มือแตะใกล้หน้าผาก (ใช้ face landmark #10)
    if results.face_landmarks and results.left_hand_landmarks:
        forehead = results.face_landmarks[10]
        lw = results.left_hand_landmarks[8]  # left index tip
        if abs(lw.x - forehead.x) < 0.08 and abs(lw.y - forehead.y) < 0.08:
            return True

    if results.face_landmarks and results.right_hand_landmarks:
        forehead = results.face_landmarks[10]
        rw = results.right_hand_landmarks[8]  # right index tip
        if abs(rw.x - forehead.x) < 0.08 and abs(rw.y - forehead.y) < 0.08:
            return True

    return False

def camera_loop():
    """
    ========================================
    หลักการทำงานของกล้อง:
    1. อ่านเฟรมจากกล้อง
    2. ใช้ Holistic ตรวจจับท่าทาง
    3. แตกเอา keypoints มาเก็บในลำดับ
    4. เมื่อเก็บ 30 เฟรมพอแล้ว ให้ AI ทำนาย
    5. แสดงผลบนหน้าจอ
    ========================================
    """
    global frame_global, prediction, prediction_confidence
    global sequence, last_pred_time, last_word_time, sentence, frame_count

    print("[CAMERA] Camera thread started")
    
    target_fps = 30
    _last_ts_ms = 0
    inactivity_threshold = 2.5  # วิินาทีที่ถือว่าไม่มีท่าต่อเนื่อง
    last_hand_time = time.time()

    print(f"🔧 FEATURE_SIZE={FEATURE_SIZE} (upper-body hybrid mode)")

    while True:
        # ===== 1. อ่านเฟรมจากกล้อง =====
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame, retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
            
        frame = cv2.flip(frame, 1)  # พลิกภาพให้เป็นภาพเงา (ตรงกับที่ผู้ใช้เห็น)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB สำหรับ MediaPipe

        # ===== 2. ใช้ Holistic ตรวจจับท่าทาง =====
        try:
            _ts_ms = max(_last_ts_ms + 1, int(time.time() * 1000))
            _last_ts_ms = _ts_ms
            _mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = holistic.detect_for_video(_mp_img, _ts_ms)
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            results = None
            continue

        # ===== 3. แตกเอา keypoints มาเก็บ =====
        data = extract_keypoints(results)
        hands_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        upper_body_ready = is_upper_body_ready(results)
        head_level_gesture = is_head_level_gesture(results)
        should_predict = hands_detected and (upper_body_ready or head_level_gesture)

        sequence.append(data)
        if hands_detected:
            last_hand_time = time.time()  # อัปเดตเวลาการเห็นมือล่าสุด

        # ===== 4. เมื่อเก็บ 30 เฟรมพอแล้ว ให้ AI ทำนาย =====
        # เราจะทำนายก็ต่อเมื่อมีการตรวจจับมืออย่างน้อยหนึ่งข้าง เพราะ
        # ถ้าไม่เห็นมือเลย โมเดลของเราถูกฝึกด้วยข้อมูลที่มีมืออยู่เสมอ
        # (และเราอยากหลีกเลี่ยงผลลัพธ์เพี้ยนเวลาเว้นช่วง)
        if model is not None and len(sequence) == 30 and should_predict and time.time() - last_pred_time > 0.8:
            # ตรวจสอบจำนวน feature เทียบกับโมเดล
            current_feat = sequence[0].shape[0] if sequence else 0
            if current_feat != FEATURE_SIZE:
                print(f"⚠️ Feature mismatch: expected {FEATURE_SIZE}, got {current_feat}")
            else:
                try:
                    res = model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
                    local_labels = model_actions if model_actions else actions
                    if not local_labels:
                        raise ValueError("No labels available for prediction")
                    pred_idx = int(np.argmax(res))
                    if pred_idx >= len(local_labels):
                        raise ValueError(f"Prediction index {pred_idx} out of range for labels")
                    word = local_labels[pred_idx]
                    conf = float(np.max(res))
                    sorted_idx = np.argsort(res)[::-1]
                    top2_conf = float(res[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
                    margin = conf - top2_conf

                    # Debug: แสดงความน่าจะเป็นทั้งหมด
                    print("🔍 probabilities:", {a: float(p) for a, p in zip(local_labels, res)})

                    prediction_history.append((word, conf))

                    # คำนวณ consensus จากหน้าต่างทำนายล่าสุด
                    scores = {}
                    best_conf = {}
                    for hist_word, hist_conf in prediction_history:
                        scores[hist_word] = scores.get(hist_word, 0.0) + hist_conf
                        best_conf[hist_word] = max(best_conf.get(hist_word, 0.0), hist_conf)

                    consensus_word = max(scores, key=scores.get)
                    consensus_conf = float(best_conf[consensus_word])
                    support_count = action_sample_counts.get(consensus_word, 0)
                    conf_bonus = LOW_SAMPLE_CONFIDENCE_BONUS if support_count < LOW_SAMPLE_SUPPORT_THRESHOLD else 0.0
                    display_threshold = DISPLAY_CONFIDENCE_THRESHOLD + conf_bonus
                    sentence_threshold = SENTENCE_CONFIDENCE_THRESHOLD + conf_bonus
                    margin_ok = margin >= PREDICTION_MARGIN_THRESHOLD

                    if consensus_conf >= display_threshold and margin_ok:
                        prediction = consensus_word
                        prediction_confidence = consensus_conf
                    else:
                        prediction = ""
                        prediction_confidence = 0.0

                    if consensus_conf >= sentence_threshold and margin_ok:
                        if len(sentence) == 0 or sentence[-1] != consensus_word:
                            sentence.append(consensus_word)
                            last_word_time = time.time()
                            print(f"✅ Detected: {consensus_word} ({consensus_conf:.2f})")
                    else:
                        print(
                            f"⚠️ Low confidence ({consensus_conf:.2f})/margin ({margin:.2f}) "
                            f"for {consensus_word} (support={support_count})"
                        )

                    last_pred_time = time.time()

                except Exception as e:
                    print(f"⚠️ Model prediction error: {e}")
                    prediction = "Error"
                    prediction_confidence = 0

        # ===== 5. รีเซ็ตและจัดการ inactivity =====
        if prediction and time.time() - last_pred_time > prediction_timeout:
            prediction = ""
            prediction_confidence = 0.0

        # ===== 5.5. บันทึก sequence ตามคำสั่งจากหน้าเว็บ =====
        with state_lock:
            recording_active = recording_state["active"]

        if recording_active and should_predict:
            should_save = False
            action_to_save = ""
            captured_frames = None

            with state_lock:
                recording_state["buffer"].append(data)
                if len(recording_state["buffer"]) >= SEQUENCE_LENGTH:
                    should_save = True
                    action_to_save = recording_state["action"]
                    captured_frames = recording_state["buffer"][:SEQUENCE_LENGTH]
                    recording_state["active"] = False
                    recording_state["buffer"] = []

            if should_save:
                try:
                    seq_id = save_recorded_sequence(action_to_save, captured_frames)
                    refresh_runtime_config()
                    with state_lock:
                        recording_state["last_saved"] = {
                            "action": action_to_save,
                            "sequence": seq_id,
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        recording_state["error"] = None
                    print(f"[OK] Recorded sequence saved: {action_to_save}/{seq_id}")
                except Exception as exc:
                    with state_lock:
                        recording_state["error"] = str(exc)
                    print(f"[ERROR] Failed to save recorded sequence: {exc}")

        # ถ้าไม่มีมือนานกว่ากำหนด ให้ถือว่าเป็นจุดสิ้นสุดประโยค
        if time.time() - last_hand_time > inactivity_threshold and sentence:
            phrase = " ".join(sentence)
            print(f"[END] Inactivity, final phrase: {phrase}")
            sentence = []
            sequence.clear()
            # รีเซ็ตสถานะการทำนายด้วย
            prediction = ""
            prediction_confidence = 0.0
            last_word_time = time.time()

        # ===== 6. วาด keypoints บนภาพ =====
        if hands_detected:
            _draw_landmarks(frame, results.left_hand_landmarks, _HAND_CONNECTIONS, dot_color=(0, 255, 0))
            _draw_landmarks(frame, results.right_hand_landmarks, _HAND_CONNECTIONS, dot_color=(0, 200, 255))

        # วาดท่าทาง (ครึ่งตัว)
        if results.pose_landmarks:
            _draw_landmarks(frame, results.pose_landmarks, _POSE_CONNECTIONS_UPPER, dot_color=(255, 100, 100))
            # วาดจุดคอ (midpoint ของหัวไหล่ซ้าย-ขวา)
            try:
                lm = results.pose_landmarks
                left_sh = lm[11]
                right_sh = lm[12]
                neck_x = int((left_sh.x + right_sh.x) / 2 * frame.shape[1])
                neck_y = int((left_sh.y + right_sh.y) / 2 * frame.shape[0])
                cv2.circle(frame, (neck_x, neck_y), 5, (255, 0, 255), -1)
                cv2.putText(frame, 'neck', (neck_x+5, neck_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255),1)
            except Exception:
                pass

        # ===== 7. แสดงข้อความผลลัพธ์ =====
        color = (0, 255, 0) if prediction_confidence > SENTENCE_CONFIDENCE_THRESHOLD else (0, 0, 255)
        display_text = f"Word: {prediction} ({prediction_confidence:.2f})" if prediction else "Word: -"
        cv2.putText(frame, display_text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        with state_lock:
            meaning_text = gesture_meanings.get(prediction, "-") if prediction else "-"
            local_needs_retrain = model_needs_retrain
            local_training = model_training

        cv2.putText(frame, f"Meaning: {meaning_text}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.putText(frame, "Sentence: " + " ".join(sentence[-10:]), (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"FPS: ~ {target_fps}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if local_training:
            cv2.putText(frame, "Model: training in background", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        elif local_needs_retrain:
            cv2.putText(frame, "Model: retrain recommended", (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ===== 8. เก็บเฟรมไว้เพื่อส่งไปแสดงผลใน web =====
        with state_lock:
            frame_global = frame.copy()

def gen():
    """
    ========================================
    ส่งเฟรมไปยัง Web Browser แบบ Live Streaming
    - อ่านเฟรมจาก frame_global
    - แปลงเป็น JPEG
    - ส่งออกแบบ multipart
    ========================================
    """
    global frame_global
    while True:
        with state_lock:
            if frame_global is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame_global)
            frame = buffer.tobytes()

        if frame_global is None:
            time.sleep(0.01)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """========== หน้าแรกของ Web App =========="""
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    """========== API สำหรับส่งสตรีมวิดีโอ =========="""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API รับค่าทำนายล่าสุดเป็น JSON"""
    # ใช้ล็อคเพื่อให้ข้อมูลไม่ขัดกันกับเธรดกล้อง
    with state_lock:
        meaning_text = gesture_meanings.get(prediction, "-") if prediction else "-"
        return {
            "prediction": prediction,
            "meaning": meaning_text,
            "confidence": float(prediction_confidence),
            "sentence": " ".join(sentence[-10:])
        }


@app.route('/api/gestures', methods=['GET'])
def api_gestures_get():
    with state_lock:
        summary = summarize_dataset()
    return jsonify({"gestures": summary})


@app.route('/api/gestures', methods=['POST'])
def api_gestures_post():
    payload = request.get_json(silent=True) or {}
    raw_name = payload.get("name", "")
    raw_meaning = payload.get("meaning", "")

    action_name = sanitize_action_name(str(raw_name))
    if not action_name:
        return jsonify({"error": "Invalid gesture name"}), 400

    meaning = str(raw_meaning).strip() if raw_meaning else action_name

    action_path = os.path.join(DATA_PATH, action_name)
    os.makedirs(action_path, exist_ok=True)

    with state_lock:
        gesture_meanings[action_name] = meaning
        save_gesture_meanings()
        refresh_runtime_config()

    return jsonify({"ok": True, "action": action_name, "meaning": meaning})


@app.route('/api/record/start', methods=['POST'])
def api_record_start():
    payload = request.get_json(silent=True) or {}
    action_name = sanitize_action_name(str(payload.get("action", "")))

    if not action_name:
        return jsonify({"error": "Invalid action"}), 400

    action_path = os.path.join(DATA_PATH, action_name)
    if not os.path.isdir(action_path):
        return jsonify({"error": "Gesture not found. Please add it first."}), 404

    with state_lock:
        if recording_state["active"]:
            return jsonify({"error": "Recording already in progress"}), 409

        recording_state["active"] = True
        recording_state["action"] = action_name
        recording_state["buffer"] = []
        recording_state["error"] = None

    return jsonify({"ok": True, "action": action_name, "target": SEQUENCE_LENGTH})


@app.route('/api/record/status', methods=['GET'])
def api_record_status():
    with state_lock:
        return jsonify(
            {
                "active": recording_state["active"],
                "action": recording_state["action"],
                "progress": len(recording_state["buffer"]),
                "target": SEQUENCE_LENGTH,
                "last_saved": recording_state["last_saved"],
                "error": recording_state["error"],
            }
        )


@app.route('/api/model/status', methods=['GET'])
def api_model_status():
    with state_lock:
        return jsonify(
            {
                "loaded": model is not None,
                "training": model_training,
                "training_error": model_training_error,
                "needs_retrain": model_needs_retrain,
                "trained_actions": model_actions,
                "dataset_actions": actions,
                "sample_counts": action_sample_counts,
                "low_support_actions": [
                    a for a, c in action_sample_counts.items() if c < LOW_SAMPLE_SUPPORT_THRESHOLD
                ],
                "last_trained": model_last_trained,
            }
        )


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    with state_lock:
        if model_training:
            return jsonify({"error": "Training already in progress"}), 409

    t = threading.Thread(target=train_model_job, daemon=True)
    t.start()
    return jsonify({"ok": True, "message": "Training started"})

if __name__ == "__main__":
    """
    ========================================
    เริ่มต้นโปรแกรม:
    1. สร้าง Thread สำหรับอ่านกล้อง (daemon=True)
    2. เริ่ม Web Server (Flask)
    3. เมื่อปิดโปรแกรม ให้ปิดกล้องอย่างปลอดภัย
    ========================================
    """
    load_runtime_model()

    try:
        t = threading.Thread(target=camera_loop, daemon=True)
        t.start()
        print("[OK] Starting Flask Web Server...")
        print("[INFO] Open browser at: http://localhost:5000")
        app.run(debug=False, threaded=True, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")
    finally:
        try:
            holistic.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Camera released")