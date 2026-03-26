from flask import Flask, Response, jsonify, render_template, request
import json
import os
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
from keras.models import load_model

from feature_utils import FEATURE_SIZE, LandmarkFeatureExtractor, landmark_list

app = Flask(__name__)

DATA_PATH = "dataset"
MODEL_PATH = "model.h5"
MODEL_LABELS_PATH = os.path.join(DATA_PATH, "model_labels.json")
GESTURE_META_PATH = os.path.join(DATA_PATH, "gestures.json")
HOLISTIC_MODEL_PATH = "holistic_landmarker.task"
CALIBRATION_PATH = os.path.join(DATA_PATH, "calibration.json")

SEQUENCE_LENGTH = 30
CONSENSUS_WINDOW = 7
DISPLAY_CONFIDENCE_THRESHOLD = 0.45
SENTENCE_CONFIDENCE_THRESHOLD = 0.60
PREDICTION_MARGIN_THRESHOLD = 0.18
PREDICTION_TIMEOUT = 1.0
MIN_PREDICTION_INTERVAL = 0.20
INACTIVITY_THRESHOLD = 1.8
NO_HAND_RESET_FRAMES = 8
MIN_VOTE_RATIO = 0.60
VISUAL_FALLBACK_CONFIDENCE = 0.22
VISUAL_FALLBACK_MARGIN = 0.06
VISUAL_FALLBACK_VOTE_RATIO = 0.40

USE_TASK_HOLISTIC = hasattr(_mp_vision, "HolisticLandmarkerOptions")
SHOW_LANDMARKS = True

_HAND_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 5),
        (5, 9),
        (9, 13),
        (13, 17),
        (0, 17),
    ]
)
_POSE_CONNECTIONS_UPPER = frozenset(
    [
        (0, 11),
        (0, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
    ]
)

state_lock = threading.Lock()

model = None
model_actions = []
gesture_meanings = {}

frame_global = None
prediction = ""
prediction_confidence = 0.0
sequence = deque(maxlen=SEQUENCE_LENGTH)
probability_history = deque(maxlen=CONSENSUS_WINDOW)
sentence_actions = []
last_prediction_time = 0.0
last_final_phrase = ""
last_final_phrase_at = 0.0
camera_mirror = False
calibration_config = {}

feature_extractor = LandmarkFeatureExtractor()


def load_gesture_meanings():
    if not os.path.exists(GESTURE_META_PATH):
        return {}

    try:
        with open(GESTURE_META_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:
        print(f"[WARN] Failed to read gestures metadata: {exc}")

    return {}


def load_model_labels(expected_len):
    if os.path.exists(MODEL_LABELS_PATH):
        try:
            with open(MODEL_LABELS_PATH, "r", encoding="utf-8") as f:
                labels = json.load(f)
            if isinstance(labels, list) and len(labels) == expected_len:
                return [str(x) for x in labels]
        except Exception as exc:
            print(f"[WARN] Failed to read model labels: {exc}")

    if os.path.isdir(DATA_PATH):
        fallback = sorted(
            [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
        )
        if len(fallback) == expected_len:
            return fallback

    return []


def load_runtime_model():
    global model, model_actions, gesture_meanings

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    loaded_model = load_model(MODEL_PATH)
    model_output_dim = int(loaded_model.output_shape[-1])
    labels = load_model_labels(model_output_dim)

    if not labels:
        raise ValueError("Unable to load model labels")

    input_features = int(loaded_model.input_shape[-1])
    if input_features != FEATURE_SIZE:
        raise ValueError(
            f"Model input mismatch: model expects {input_features}, extractor produces {FEATURE_SIZE}"
        )

    model = loaded_model
    model_actions = labels
    gesture_meanings = load_gesture_meanings()

    print(f"[OK] Model loaded: classes={len(model_actions)}, feature_size={FEATURE_SIZE}")


def _default_calibration():
    return {
        "camera_mirror": False,
        "global": {
            "display_confidence": DISPLAY_CONFIDENCE_THRESHOLD,
            "sentence_confidence": SENTENCE_CONFIDENCE_THRESHOLD,
            "margin": PREDICTION_MARGIN_THRESHOLD,
            "vote_ratio": MIN_VOTE_RATIO,
        },
        "class_thresholds": {},
    }


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _clamp(value, low, high):
    return max(low, min(high, value))


def load_calibration_config():
    global calibration_config, camera_mirror

    cfg = _default_calibration()
    if os.path.exists(CALIBRATION_PATH):
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cfg.update({k: v for k, v in loaded.items() if k in cfg})
        except Exception as exc:
            print(f"[WARN] Failed to load calibration config: {exc}")

    global_cfg = cfg.get("global", {})
    cfg["global"] = {
        "display_confidence": _clamp(_to_float(global_cfg.get("display_confidence"), DISPLAY_CONFIDENCE_THRESHOLD), 0.05, 0.99),
        "sentence_confidence": _clamp(_to_float(global_cfg.get("sentence_confidence"), SENTENCE_CONFIDENCE_THRESHOLD), 0.05, 0.99),
        "margin": _clamp(_to_float(global_cfg.get("margin"), PREDICTION_MARGIN_THRESHOLD), 0.01, 0.8),
        "vote_ratio": _clamp(_to_float(global_cfg.get("vote_ratio"), MIN_VOTE_RATIO), 0.3, 1.0),
    }

    class_cfg = cfg.get("class_thresholds", {})
    if not isinstance(class_cfg, dict):
        class_cfg = {}
    sanitized_class_cfg = {}
    for label, item in class_cfg.items():
        if not isinstance(item, dict):
            continue
        sanitized_class_cfg[str(label)] = {
            "display_confidence": _clamp(_to_float(item.get("display_confidence"), cfg["global"]["display_confidence"]), 0.05, 0.99),
            "sentence_confidence": _clamp(_to_float(item.get("sentence_confidence"), cfg["global"]["sentence_confidence"]), 0.05, 0.99),
            "margin": _clamp(_to_float(item.get("margin"), cfg["global"]["margin"]), 0.01, 0.8),
            "vote_ratio": _clamp(_to_float(item.get("vote_ratio"), cfg["global"]["vote_ratio"]), 0.3, 1.0),
        }

    cfg["class_thresholds"] = sanitized_class_cfg
    cfg["camera_mirror"] = bool(cfg.get("camera_mirror", False))

    calibration_config = cfg
    camera_mirror = cfg["camera_mirror"]


def save_calibration_config():
    os.makedirs(DATA_PATH, exist_ok=True)
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(calibration_config, f, ensure_ascii=False, indent=2)


def get_action_thresholds(action_name):
    global_cfg = calibration_config.get("global", _default_calibration()["global"])
    class_cfg = calibration_config.get("class_thresholds", {}).get(action_name, {})
    return {
        "display_confidence": _to_float(class_cfg.get("display_confidence"), global_cfg["display_confidence"]),
        "sentence_confidence": _to_float(class_cfg.get("sentence_confidence"), global_cfg["sentence_confidence"]),
        "margin": _to_float(class_cfg.get("margin"), global_cfg["margin"]),
        "vote_ratio": _to_float(class_cfg.get("vote_ratio"), global_cfg["vote_ratio"]),
    }


def setup_holistic_detector():
    if USE_TASK_HOLISTIC:
        if not os.path.exists(HOLISTIC_MODEL_PATH):
            import urllib.request

            dl_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
            )
            print(f"[INFO] Downloading {HOLISTIC_MODEL_PATH}")
            urllib.request.urlretrieve(dl_url, HOLISTIC_MODEL_PATH)

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

    print("[WARN] HolisticLandmarker Tasks API unavailable. Using mp.solutions.holistic fallback")
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )


def detect_holistic(holistic, rgb_frame, timestamp_ms):
    if USE_TASK_HOLISTIC:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return holistic.detect_for_video(mp_image, timestamp_ms)
    return holistic.process(rgb_frame)


def sentence_text_from_actions(action_items):
    return " ".join(gesture_meanings.get(a, a) for a in action_items)


def is_upper_body_ready(results):
    pose = landmark_list(results.pose_landmarks if results else None)
    if not pose:
        return False

    for idx in (11, 12, 23, 24):
        vis = getattr(pose[idx], "visibility", None)
        if vis is not None and vis < 0.35:
            return False
    return True


def is_head_level_gesture(results):
    pose = landmark_list(results.pose_landmarks if results else None)
    if not pose:
        return False

    face = landmark_list(results.face_landmarks if results else None)
    left = landmark_list(results.left_hand_landmarks if results else None)
    right = landmark_list(results.right_hand_landmarks if results else None)

    nose_y = pose[0].y
    if pose[15].y < nose_y + 0.04 or pose[16].y < nose_y + 0.04:
        return True

    if face and left:
        forehead = face[10]
        index_tip = left[8]
        if abs(index_tip.x - forehead.x) < 0.08 and abs(index_tip.y - forehead.y) < 0.08:
            return True

    if face and right:
        forehead = face[10]
        index_tip = right[8]
        if abs(index_tip.x - forehead.x) < 0.08 and abs(index_tip.y - forehead.y) < 0.08:
            return True

    return False


def _draw_landmarks(frame, landmarks, connections, dot_color, line_color=(180, 180, 180)):
    points = landmark_list(landmarks)
    if not points:
        return

    h, w = frame.shape[:2]
    for start, end in connections:
        if start < len(points) and end < len(points):
            x1, y1 = int(points[start].x * w), int(points[start].y * h)
            x2, y2 = int(points[end].x * w), int(points[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 1)

    for lm in points:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, dot_color, -1)


def camera_loop():
    global frame_global, prediction, prediction_confidence
    global last_prediction_time, sentence_actions, last_final_phrase, last_final_phrase_at

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    holistic = setup_holistic_detector()
    ts_ms = 0
    last_hand_time = time.time()
    no_hand_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            with state_lock:
                local_mirror = camera_mirror

            if local_mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = max(ts_ms + 1, int(time.time() * 1000))
            results = detect_holistic(holistic, rgb, ts_ms)

            left_hand = landmark_list(results.left_hand_landmarks if results else None)
            right_hand = landmark_list(results.right_hand_landmarks if results else None)
            pose_lm = landmark_list(results.pose_landmarks if results else None)

            hands_detected = bool(left_hand or right_hand)
            # Keep prediction alive when hands are visible; strict filtering is applied later via thresholds.
            should_predict = hands_detected

            if hands_detected:
                last_hand_time = time.time()
                no_hand_frames = 0
            else:
                no_hand_frames += 1

            if no_hand_frames >= NO_HAND_RESET_FRAMES:
                sequence.clear()
                probability_history.clear()
                feature_extractor.reset()

            if should_predict:
                features = feature_extractor.extract(results)
                sequence.append(features)

            if model is not None and len(sequence) == SEQUENCE_LENGTH and should_predict:
                if time.time() - last_prediction_time >= MIN_PREDICTION_INTERVAL:
                    probs = model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
                    probability_history.append(np.asarray(probs, dtype=np.float32))

                    smooth_probs = np.mean(np.stack(list(probability_history), axis=0), axis=0)
                    top_idx = int(np.argmax(smooth_probs))
                    sorted_idx = np.argsort(smooth_probs)[::-1]
                    top_conf = float(smooth_probs[top_idx])
                    second_conf = float(smooth_probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
                    margin = top_conf - second_conf
                    vote_ratio = float(
                        np.mean(
                            [
                                1.0 if int(np.argmax(prev_probs)) == top_idx else 0.0
                                for prev_probs in probability_history
                            ]
                        )
                    )

                    if top_idx < len(model_actions):
                        top_action = model_actions[top_idx]
                    else:
                        top_action = ""

                    thresholds = get_action_thresholds(top_action) if top_action else get_action_thresholds("")
                    display_thr = thresholds["display_confidence"]
                    sentence_thr = thresholds["sentence_confidence"]
                    margin_thr = thresholds["margin"]
                    vote_thr = thresholds["vote_ratio"]

                    visual_conf_thr = min(display_thr, max(VISUAL_FALLBACK_CONFIDENCE, display_thr * 0.65))
                    visual_margin_thr = min(margin_thr, max(VISUAL_FALLBACK_MARGIN, margin_thr * 0.65))
                    visual_vote_thr = min(vote_thr, max(VISUAL_FALLBACK_VOTE_RATIO, vote_thr * 0.70))

                    if top_conf >= visual_conf_thr and margin >= visual_margin_thr and vote_ratio >= visual_vote_thr:
                        prediction = top_action
                        prediction_confidence = top_conf
                    else:
                        prediction = ""
                        prediction_confidence = 0.0

                    if (
                        top_action
                        and top_conf >= sentence_thr
                        and margin >= margin_thr
                        and vote_ratio >= vote_thr
                    ):
                        if not sentence_actions or sentence_actions[-1] != top_action:
                            sentence_actions.append(top_action)
                            sentence_actions = sentence_actions[-20:]

                    last_prediction_time = time.time()

            if prediction and time.time() - last_prediction_time > PREDICTION_TIMEOUT:
                prediction = ""
                prediction_confidence = 0.0

            if time.time() - last_hand_time > INACTIVITY_THRESHOLD and sentence_actions:
                final_phrase = sentence_text_from_actions(sentence_actions)
                last_final_phrase = final_phrase
                last_final_phrase_at = time.time()
                sentence_actions = []
                sequence.clear()
                probability_history.clear()
                feature_extractor.reset()
                prediction = ""
                prediction_confidence = 0.0

            if SHOW_LANDMARKS:
                if left_hand:
                    _draw_landmarks(frame, left_hand, _HAND_CONNECTIONS, dot_color=(0, 255, 0))
                if right_hand:
                    _draw_landmarks(frame, right_hand, _HAND_CONNECTIONS, dot_color=(0, 200, 255))
                if pose_lm:
                    _draw_landmarks(frame, pose_lm, _POSE_CONNECTIONS_UPPER, dot_color=(255, 120, 120))

            color = (0, 220, 0) if prediction_confidence >= SENTENCE_CONFIDENCE_THRESHOLD else (0, 180, 255)
            word_text = f"Word: {prediction} ({prediction_confidence:.2f})" if prediction else "Word: -"
            meaning_text = gesture_meanings.get(prediction, "-") if prediction else "-"
            sentence_text = sentence_text_from_actions(sentence_actions[-8:])

            cv2.putText(frame, word_text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Meaning: {meaning_text}", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 60), 2)
            cv2.putText(frame, f"Sentence: {sentence_text}", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2)

            with state_lock:
                frame_global = frame.copy()

    finally:
        try:
            holistic.close()
        except Exception:
            pass
        cap.release()


def gen():
    global frame_global

    while True:
        with state_lock:
            frame = None if frame_global is None else frame_global.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        payload = buffer.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calibration")
def calibration_page():
    return render_template("calibration.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with state_lock:
        meaning_text = gesture_meanings.get(prediction, "-") if prediction else "-"
        sentence_text = sentence_text_from_actions(sentence_actions[-8:])
        return {
            "prediction": prediction,
            "meaning": meaning_text,
            "confidence": float(prediction_confidence),
            "sentence": sentence_text,
            "final_phrase": last_final_phrase,
            "final_phrase_at": float(last_final_phrase_at),
            "camera_mirror": camera_mirror,
        }


@app.route("/api/calibration", methods=["GET"])
def api_calibration_get():
    with state_lock:
        return {
            "config": calibration_config,
            "actions": model_actions,
        }


@app.route("/api/calibration", methods=["POST"])
def api_calibration_post():
    global camera_mirror

    payload = request.get_json(silent=True) or {}
    incoming_cfg = payload.get("config", {})
    if not isinstance(incoming_cfg, dict):
        return jsonify({"error": "Invalid config payload"}), 400

    with state_lock:
        current = _default_calibration()
        current.update(calibration_config)

        global_in = incoming_cfg.get("global", {})
        if isinstance(global_in, dict):
            current["global"] = {
                "display_confidence": _clamp(_to_float(global_in.get("display_confidence"), current["global"]["display_confidence"]), 0.05, 0.99),
                "sentence_confidence": _clamp(_to_float(global_in.get("sentence_confidence"), current["global"]["sentence_confidence"]), 0.05, 0.99),
                "margin": _clamp(_to_float(global_in.get("margin"), current["global"]["margin"]), 0.01, 0.8),
                "vote_ratio": _clamp(_to_float(global_in.get("vote_ratio"), current["global"]["vote_ratio"]), 0.3, 1.0),
            }

        class_in = incoming_cfg.get("class_thresholds", {})
        if isinstance(class_in, dict):
            sanitized = {}
            for action_name, item in class_in.items():
                if action_name not in model_actions or not isinstance(item, dict):
                    continue
                sanitized[action_name] = {
                    "display_confidence": _clamp(_to_float(item.get("display_confidence"), current["global"]["display_confidence"]), 0.05, 0.99),
                    "sentence_confidence": _clamp(_to_float(item.get("sentence_confidence"), current["global"]["sentence_confidence"]), 0.05, 0.99),
                    "margin": _clamp(_to_float(item.get("margin"), current["global"]["margin"]), 0.01, 0.8),
                    "vote_ratio": _clamp(_to_float(item.get("vote_ratio"), current["global"]["vote_ratio"]), 0.3, 1.0),
                }
            current["class_thresholds"] = sanitized

        current["camera_mirror"] = bool(incoming_cfg.get("camera_mirror", current.get("camera_mirror", False)))

        calibration_config.clear()
        calibration_config.update(current)
        camera_mirror = current["camera_mirror"]
        save_calibration_config()

    return {"ok": True, "config": calibration_config}


if __name__ == "__main__":
    load_runtime_model()
    load_calibration_config()

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print("[OK] Translation server started at http://localhost:5000")
    app.run(debug=False, threaded=True, host="0.0.0.0")
