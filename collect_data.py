import cv2
import mediapipe as mp
import numpy as np
import os

# ===== กำหนดค่า =====
actions = ["hello", "hungry", "one", "two", "three", "four", "five"]
no_sequences = 5  # จำนวน video ต่อ action (เพิ่มขึ้นเพื่อให้ได้ ~100+ sequences)
sequence_length = 30  # จำนวน frames ต่อ sequence
DATA_PATH = "dataset"

# ===== ตั้งค่า MediaPipe Holistic =====
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===== เปิดกล้อง =====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้อง")
    exit()

def extract_keypoints(results):
    """
    แตกเอา keypoints จาก Holistic results
    - Pose: 33 จุด × 3 = 99
    - Face: 50 จุด × 3 = 150
    - LeftHand: 21 จุด × 3 = 63
    - RightHand: 21 จุด × 3 = 63
    ===================== รวม = 375
    """
    data = []

    # Pose
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*99)

    # Face
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark[:50]:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*150)

    # Left Hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*63)

    # Right Hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*63)

    return np.array(data)

# ===== สร้างโครงสร้าง dataset/action/seq_num/ =====
for action in actions:
    for seq in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

print("✅ โครงสร้าง dataset สร้างเรียบร้อย")
print(f"📝 Actions: {actions}")
print(f"📝 Sequences per action: {no_sequences}")
print(f"📝 Frames per sequence: {sequence_length}")

# ===== เก็บข้อมูล =====
for action in actions:
    for seq in range(no_sequences):
        print(f"\n🎥 Recording: {action} - Sequence {seq}")
        
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️  Failed to read frame {frame_num}")
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ===== ตรวจจับท่าทาง =====
            try:
                results = holistic.process(rgb)
            except Exception as e:
                print(f"⚠️  Detection error: {e}")
                continue
            
            # ===== แตกเอา keypoints =====
            keypoints = extract_keypoints(results)
            
            # ===== บันทึก =====
            file_path = os.path.join(DATA_PATH, action, str(seq), str(frame_num))
            np.save(file_path, keypoints)
            
            # ===== แสดงบนหน้าจอ =====
            cv2.putText(frame, f"{action} {seq}/{no_sequences-1}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_num}/{sequence_length-1}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # วาด landmarks
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, 
                                      mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, 
                                      mp_holistic.HAND_CONNECTIONS)
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, 
                                      mp_holistic.POSE_CONNECTIONS)
            
            cv2.imshow("Collecting Data", frame)
            
            # Press 'q' to quit, 'escape' to skip this action
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n❌ Quitting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == 27:  # Escape
                print(f"\n⏭️  Skipping {action}")
                break

cap.release()
cv2.destroyAllWindows()
print("\n✅ เก็บข้อมูลเสร็จแล้ว!")
