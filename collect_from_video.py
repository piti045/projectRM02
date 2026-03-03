import cv2
import mediapipe as mp
import numpy as np
import os

VIDEO_PATH = "videos"     # โฟลเดอร์วิดีโอ
DATA_PATH = "dataset"    # โฟลเดอร์เก็บข้อมูล
SEQUENCE_LENGTH = 30     # 30 frame ต่อ 1 sample

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands()
face = mp_face.FaceMesh()
pose = mp_pose.Pose()

def extract_landmarks(h, f, p):
    data = []

    # มือ (21 จุด * 3)
    if h.multi_hand_landmarks:
        for lm in h.multi_hand_landmarks[0].landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*63)

    # หน้า (50 จุดแรก)
    if f.multi_face_landmarks:
        for lm in f.multi_face_landmarks[0].landmark[:50]:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*150)

    # ท่าทาง (33 จุด)
    if p.pose_landmarks:
        for lm in p.pose_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
    else:
        data.extend([0]*99)

    return np.array(data)


for action in os.listdir(VIDEO_PATH):
    action_path = os.path.join(VIDEO_PATH, action)
    save_path = os.path.join(DATA_PATH, action)
    os.makedirs(save_path, exist_ok=True)

    print("📂 Action:", action)

    for video_file in os.listdir(action_path):
        video_path = os.path.join(action_path, video_file)
        print("🎬 Opening:", video_path)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Cannot open video:", video_file)
            continue

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h = hands.process(rgb)
            f = face.process(rgb)
            p = pose.process(rgb)

            landmarks = extract_landmarks(h, f, p)
            frames.append(landmarks)

        cap.release()
        print("🧠 Total frames:", len(frames))

        if len(frames) < SEQUENCE_LENGTH:
            print("⚠ Not enough frames, skipped:", video_file)
            continue

        for i in range(0, len(frames) - SEQUENCE_LENGTH, SEQUENCE_LENGTH):
            seq = frames[i:i + SEQUENCE_LENGTH]
            file_name = f"{video_file}_{i}.npy"
            save_file = os.path.join(save_path, file_name)
            np.save(save_file, seq)
            print("✅ Saved:", save_file)

        print("✔ Finished:", video_file)