from flask import Flask, render_template, Response
import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
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

# ============================================
# 5. ตัวแปรทั่วโลกสำหรับเก็บข้อมูล
# ============================================
frame_global = None
prediction = ""
prediction_confidence = 0
sequence = deque(maxlen=30)  # เก็บ 30 frames ล่าสุด
sentence = []
last_pred_time = 0
last_word_time = time.time()
frame_count = 0
prediction_timeout = 2  # รีเซ็ตการทำนายหลังจาก 2 วินาที

# ดึงขนาด feature ที่โมเดลคาดหวัง
expected_features = None
if model is not None:
    try:
        _, seq_len, feat = model.input_shape
        expected_features = feat
        print(f"🔧 Model expects {seq_len} timesteps and {feat} features per step")
    except Exception:
        pass

lock = threading.Lock()  # ล็อคเพื่อความปลอดภัยการเข้าถึงข้อมูลพร้อมกัน

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
    global frame_global, prediction, prediction_confidence, sequence, last_pred_time, last_word_time, sentence, frame_count

    print("📷 Camera thread started")
    
    target_fps = 30
    
    inactivity_threshold = 2.5  # วิินาทีที่ถือว่าไม่มีท่าต่อเนื่อง
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

        # append ทุกกรณีเพื่อให้ sequence ยาว 30 เฟรมเสมอ
        sequence.append(data)
        if hands_detected:
            last_hand_time = time.time()  # อัปเดตเวลาการเห็นมือล่าสุด

        # ===== 4. เมื่อเก็บ 30 เฟรมพอแล้ว ให้ AI ทำนาย =====
        # เราจะทำนายก็ต่อเมื่อมีการตรวจจับมืออย่างน้อยหนึ่งข้าง เพราะ
        # ถ้าไม่เห็นมือเลย โมเดลของเราถูกฝึกด้วยข้อมูลที่มีมืออยู่เสมอ
        # (และเราอยากหลีกเลี่ยงผลลัพธ์เพี้ยนเวลาเว้นช่วง)
        if len(sequence) == 30 and hands_detected and time.time() - last_pred_time > 0.8:
            # ตรวจสอบจำนวน feature เทียบกับโมเดล
            current_feat = sequence[0].shape[0] if sequence else 0
            if expected_features and current_feat != expected_features:
                print(f"⚠️ Feature mismatch: model expects {expected_features}, got {current_feat}")
                # ไม่ทำนายเพื่อตัดปัญหา
            else:
                try:
                    res = model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
                    word = actions[np.argmax(res)]
                    conf = float(np.max(res))

                # Debug: แสดงความน่าจะเป็นทั้งหมด
                print("🔍 probabilities:", {a: float(p) for a,p in zip(actions, res)})

                if conf > 0.6:
                    if len(sentence) == 0 or sentence[-1] != word:
                        sentence.append(word)
                        last_word_time = time.time()
                        print(f"✅ Detected: {word} ({conf:.2f})")
                else:
                    print(f"⚠️ Low confidence ({conf:.2f}) for sequence")

                prediction = word
                prediction_confidence = conf
                last_pred_time = time.time()
                
            except Exception as e:
                print(f"⚠️ Model prediction error: {e}")
                prediction = "Error"
                prediction_confidence = 0

        # ===== 5. รีเซ็ตและจัดการ inactivity =====
        if time.time() - last_word_time > prediction_timeout:
            prediction = ""
            prediction_confidence = 0

        # ถ้าไม่มีมือนานกว่ากำหนด ให้ถือว่าเป็นจุดสิ้นสุดประโยค
        if time.time() - last_hand_time > inactivity_threshold and sentence:
            phrase = " ".join(sentence)
            print(f"🛑 Inactivity, final phrase: {phrase}")
            sentence = []
            sequence.clear()
            # รีเซ็ตสถานะการทำนายด้วย
            prediction = ""
            prediction_confidence = 0
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
        color = (0, 255, 0) if prediction_confidence > 0.6 else (0, 0, 255)
        display_text = f"Word: {prediction} ({prediction_confidence:.2f})" if prediction else "Word: -"
        cv2.putText(frame, display_text, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, "Sentence: " + " ".join(sentence[-10:]), (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: ~ {target_fps}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ===== 8. เก็บเฟรมไว้เพื่อส่งไปแสดงผลใน web =====
        with lock:
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
        with lock:
            if frame_global is None:
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
    """API รับค่าทำนายล่าสุดเป็น JSON"""
    # ใช้ล็อคเพื่อให้ข้อมูลไม่ขัดกันกับเธรดกล้อง
    with lock:
        return {
            "prediction": prediction,
            "sentence": " ".join(sentence[-10:])
        }

if __name__ == "__main__":
    """
    ========================================
    เริ่มต้นโปรแกรม:
    1. สร้าง Thread สำหรับอ่านกล้อง (daemon=True)
    2. เริ่ม Web Server (Flask)
    3. เมื่อปิดโปรแกรม ให้ปิดกล้องอย่างปลอดภัย
    ========================================
    """
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