from flask import Flask, render_template, Response, request, jsonify
import json
import os
import re
import cv2
import mediapipe as mp
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

DATA_PATH = "dataset"
MODEL_PATH = "model.h5"
MODEL_LABELS_PATH = "model_labels.json"
GESTURE_META_PATH = os.path.join(DATA_PATH, "gestures.json")

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 375
PREDICTION_INTERVAL = 0.8
CONFIDENCE_THRESHOLD = 0.6

state_lock = threading.Lock()

model = None
model_output_dim = 0
model_actions = []
actions = []
gesture_meanings = {}
model_needs_retrain = False
model_training = False
model_training_error = ""
model_last_trained = None

# ============================================
# 3. โหลด MediaPipe Holistic (รู้จำใบหน้า, คอ, แขน, ตัว)
# ============================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

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
        print(f"⚠️ Failed to load model labels: {exc}")
    return None


def refresh_runtime_config():
    global actions, model_needs_retrain

    actions = list_action_dirs()
    load_gesture_meanings()

    if model is None:
        model_needs_retrain = len(actions) > 0
        return

    model_needs_retrain = set(actions) != set(model_actions)


def load_runtime_model():
    global model, model_output_dim, model_actions, model_last_trained

    model = None
    model_output_dim = 0
    model_actions = []
    model_last_trained = None

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

        print(f"✅ Model loaded successfully ({model_output_dim} classes)")
        print(f"📝 Model actions: {model_actions}")
    except Exception as exc:
        model = None
        model_output_dim = 0
        model_actions = []
        print(f"❌ Failed to load model: {exc}")

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
        print(f"📦 Training samples loaded: {X.shape[0]}")
        if skipped:
            print(f"⚠️ Skipped sequences: {len(skipped)}")

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
            print(f"🧪 Validation enabled. Train={X_train.shape[0]}, Val={X_test.shape[0]}")
        else:
            X_train, y_train = X, y_one_hot
            validation_data = None
            callbacks = []
            print("⚠️ Validation skipped due to limited data")

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
        print("✅ Training complete and model reloaded")

    except Exception as exc:
        with state_lock:
            model_training_error = str(exc)
        print(f"❌ Training failed: {exc}")
    finally:
        with state_lock:
            model_training = False

def extract_keypoints(results):
    """
    ========================================
    แตกเอาจุด keypoints จากผลการตรวจจับ
    ========================================
    - Pose: 33 จุด × 3 (x, y, z) = 99 ค่า
    - Face: 50 จุด × 3 (x, y, z) = 150 ค่า  
    - Left Hand: 21 จุด × 3 (x, y, z) = 63 ค่า
    - Right Hand: 21 จุด × 3 (x, y, z) = 63 ค่า
    ============= รวมทั้งหมด = 375 ค่า =============
    """
    data = []

    # ===== เก็บจุด Pose (ครึ่งตัว, คอ, ศรีษะแบบ 33 จุด) =====
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*99)  # ถ้าไม่ตรวจจับได้ ใส่ 0

    # ===== เก็บจุด Face (ใบหน้า 468 จุดแต่เราใช้ 50 จุด) =====
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark[:50]:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*150)  # ถ้าไม่ตรวจจับได้ ใส่ 0

    # ===== เก็บจุด Left Hand (มือซ้าย 21 จุด) =====
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*63)

    # ===== เก็บจุด Right Hand (มือขวา 21 จุด) =====
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*63)

    return np.array(data)

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

    print("📷 Camera thread started")
    
    target_fps = 30
    inactivity_threshold = 2.5
    last_hand_time = time.time()

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
            results = holistic.process(rgb)
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            results = None
            continue

        # ===== 3. แตกเอา keypoints มาเก็บ =====
        data = extract_keypoints(results)
        hands_detected = results.left_hand_landmarks or results.right_hand_landmarks

        sequence.append(data)
        if hands_detected:
            last_hand_time = time.time()

        with state_lock:
            rec_active = recording_state["active"]
            rec_action = recording_state["action"]

        if rec_active:
            with state_lock:
                recording_state["buffer"].append(data.copy())
                rec_progress = len(recording_state["buffer"])

            cv2.putText(
                frame,
                f"Recording {rec_action}: {rec_progress}/{SEQUENCE_LENGTH}",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

            if rec_progress >= SEQUENCE_LENGTH:
                try:
                    with state_lock:
                        captured = list(recording_state["buffer"][:SEQUENCE_LENGTH])
                    seq_id = save_recorded_sequence(rec_action, captured)
                    with state_lock:
                        recording_state["active"] = False
                        recording_state["buffer"] = []
                        recording_state["error"] = None
                        recording_state["last_saved"] = {
                            "action": rec_action,
                            "sequence": seq_id,
                            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    refresh_runtime_config()
                    print(f"💾 Saved sequence {rec_action}/{seq_id}")
                except Exception as exc:
                    with state_lock:
                        recording_state["active"] = False
                        recording_state["buffer"] = []
                        recording_state["error"] = str(exc)
                    print(f"❌ Failed to save recording: {exc}")

        with state_lock:
            local_model = model
            local_model_actions = list(model_actions)

        if (
            local_model is not None
            and len(local_model_actions) > 0
            and len(sequence) == SEQUENCE_LENGTH
            and hands_detected
            and time.time() - last_pred_time > PREDICTION_INTERVAL
        ):
            current_feat = sequence[0].shape[0] if sequence else 0
            if current_feat != FEATURE_SIZE:
                print(f"⚠️ Feature mismatch: expected {FEATURE_SIZE}, got {current_feat}")
            else:
                try:
                    res = local_model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
                    if len(res) != len(local_model_actions):
                        raise ValueError(
                            f"Output mismatch: model={len(res)} labels={len(local_model_actions)}"
                        )

                    best_idx = int(np.argmax(res))
                    word = local_model_actions[best_idx]
                    conf = float(np.max(res))

                    if conf > CONFIDENCE_THRESHOLD:
                        if len(sentence) == 0 or sentence[-1] != word:
                            sentence.append(word)
                            last_word_time = time.time()
                            print(f"✅ Detected: {word} ({conf:.2f})")

                    prediction = word
                    prediction_confidence = conf
                    last_pred_time = time.time()

                except Exception as exc:
                    print(f"⚠️ Model prediction error: {exc}")
                    prediction = ""
                    prediction_confidence = 0.0

        # ===== 5. รีเซ็ตและจัดการ inactivity =====
        if time.time() - last_word_time > prediction_timeout:
            prediction = ""
            prediction_confidence = 0.0

        # ถ้าไม่มีมือนานกว่ากำหนด ให้ถือว่าเป็นจุดสิ้นสุดประโยค
        if time.time() - last_hand_time > inactivity_threshold and sentence:
            phrase = " ".join(sentence)
            print(f"🛑 Inactivity, final phrase: {phrase}")
            sentence = []
            sequence.clear()
            # รีเซ็ตสถานะการทำนายด้วย
            prediction = ""
            prediction_confidence = 0.0
            last_word_time = time.time()

        # ===== 6. วาด keypoints บนภาพ =====
        if hands_detected:
            # วาดมือซ้าย
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )
            # วาดมือขวา
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

        # วาดท่าทาง (ครึ่งตัว คอ)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )
            # 🔍 วาดจุดคอ (midpoint ของหัวไหล่ซ้าย-ขวา)
            try:
                lm = results.pose_landmarks.landmark
                left_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_sh = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                neck_x = int((left_sh.x + right_sh.x) / 2 * frame.shape[1])
                neck_y = int((left_sh.y + right_sh.y) / 2 * frame.shape[0])
                cv2.circle(frame, (neck_x, neck_y), 5, (255, 0, 255), -1)
                cv2.putText(frame, 'neck', (neck_x+5, neck_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255),1)
            except Exception:
                pass

        # ===== 7. แสดงข้อความผลลัพธ์ =====
        color = (0, 255, 0) if prediction_confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
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
                time.sleep(0.01)
                continue
            ret, buffer = cv2.imencode('.jpg', frame_global)
            frame = buffer.tobytes()

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
    with state_lock:
        meaning = gesture_meanings.get(prediction, "") if prediction else ""
        return jsonify(
            {
            "prediction": prediction,
            "meaning": meaning,
            "confidence": prediction_confidence,
            "sentence": " ".join(sentence[-10:]),
            "needs_retrain": model_needs_retrain,
            "training": model_training,
            }
        )


@app.route('/api/gestures', methods=['GET', 'POST'])
def gestures_api():
    if request.method == 'GET':
        with state_lock:
            summary = summarize_dataset()
        return jsonify({"gestures": summary})

    payload = request.get_json(silent=True) or {}
    name_raw = str(payload.get("name", "")).strip()
    meaning_raw = str(payload.get("meaning", "")).strip()

    if not name_raw:
        return jsonify({"error": "name is required"}), 400

    action_name = sanitize_action_name(name_raw)
    if not action_name:
        return jsonify({"error": "name must contain letters or numbers"}), 400

    action_path = os.path.join(DATA_PATH, action_name)
    created = False

    if not os.path.exists(action_path):
        os.makedirs(action_path, exist_ok=True)
        created = True

    with state_lock:
        if action_name not in gesture_meanings:
            gesture_meanings[action_name] = meaning_raw or action_name
        elif meaning_raw:
            gesture_meanings[action_name] = meaning_raw
        save_gesture_meanings()

    refresh_runtime_config()

    return (
        jsonify(
            {
                "action": action_name,
                "meaning": gesture_meanings.get(action_name, action_name),
                "created": created,
                "needs_retrain": model_needs_retrain,
            }
        ),
        201 if created else 200,
    )


@app.route('/api/record/start', methods=['POST'])
def start_record():
    payload = request.get_json(silent=True) or {}
    action_name = str(payload.get("action", "")).strip()

    if not action_name:
        return jsonify({"error": "action is required"}), 400

    if action_name not in list_action_dirs():
        return jsonify({"error": "action not found"}), 404

    with state_lock:
        if recording_state["active"]:
            return jsonify({"error": "recording already active"}), 409
        recording_state["active"] = True
        recording_state["action"] = action_name
        recording_state["buffer"] = []
        recording_state["error"] = None
        recording_state["last_saved"] = None

    return jsonify({"status": "recording_started", "action": action_name})


@app.route('/api/record/status')
def record_status():
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


@app.route('/api/retrain', methods=['POST'])
def retrain_api():
    with state_lock:
        if model_training:
            return jsonify({"error": "training already in progress"}), 409

    t = threading.Thread(target=train_model_job, daemon=True)
    t.start()
    return jsonify({"status": "training_started"})


@app.route('/api/model/status')
def model_status():
    with state_lock:
        return jsonify(
            {
                "loaded": model is not None,
                "output_dim": model_output_dim,
                "trained_actions": model_actions,
                "dataset_actions": actions,
                "needs_retrain": model_needs_retrain,
                "training": model_training,
                "training_error": model_training_error,
                "last_trained": model_last_trained,
            }
        )

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
        print("✅ Starting Flask Web Server...")
        print("📱 Open browser at: http://localhost:5000")
        app.run(debug=False, threaded=True, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n❌ Shutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")