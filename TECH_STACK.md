# TECH STACK AND SYSTEM OVERVIEW

เอกสารนี้สรุปว่าโปรเจกต์นี้ใช้ระบบอะไรบ้างในการสร้างระบบแปลภาษามือแบบเรียลไทม์

## 1) Core Runtime Stack
- Language: Python 3.10
- Web Framework: Flask
- CV and Camera I/O: OpenCV
- Landmark Detection: MediaPipe Holistic
  - ใช้ Tasks API เมื่อมี
  - fallback เป็น `mp.solutions.holistic` เมื่อ Tasks API ไม่มี
- Deep Learning Inference/Training: Keras (TensorFlow backend)
- Data Processing: NumPy

## 2) AI Pipeline (Inference)
- Input: กล้องเว็บแคม
- Vision Stage:
  - ตรวจจับ landmark ของ pose + face + มือซ้าย/ขวา
- Feature Engineering Stage:
  - Normalized coordinates (body and hand)
  - Hand velocity (delta ระหว่างเฟรม)
  - Hand distance features (คู่จุดนิ้วสำคัญ)
- Temporal Stage:
  - เก็บเป็น sequence 30 เฟรม
- Classification Stage:
  - LSTM-based model (`model.h5`)
- Post-processing Stage:
  - confidence threshold
  - margin threshold (top1-top2)
  - vote ratio จากประวัติหลายเฟรม
  - smoothing ด้วย probability history

## 3) Calibration System
- มีระบบปรับค่าพารามิเตอร์แบบ runtime ผ่านหน้า calibration
- รองรับทั้ง:
  - Global thresholds
  - Per-class thresholds
- ปรับ camera mirror ได้
- ตั้งค่าเก็บใน `dataset/calibration.json`

## 4) Frontend/UI
- หน้าแปลหลัก: แสดงภาพกล้อง, คำทำนาย, ความหมาย, ประโยค, confidence
- หน้า calibration: ปรับ threshold และ camera mirror
- Text-to-Speech (TTS): ใช้ Web Speech API (`speechSynthesis`) ฝั่งเบราว์เซอร์

## 5) Training and Dataset Stack
- Dataset format:
  - `dataset/<action>/<sequence_id>/<frame_id>.npy`
  - 1 sequence = 30 เฟรม
- Video-to-dataset conversion:
  - ใช้ `collect_from_video.py`
- Training:
  - ใช้ `train_model.py`
  - บันทึก label mapping ที่ `dataset/model_labels.json`
- Dataset tools:
  - `check_dataset_balance.py` (ตรวจสมดุล)
  - `debug_dataset.py`, `diagnose.py` (ตรวจโครงสร้าง/feature)
  - `augment_dataset.py` (เพิ่มข้อมูลเชิงสถิติ)

## 6) Model and Feature Compatibility
- Runtime ตรวจว่า model input feature size ตรงกับ feature extractor
- ถ้าไม่ตรง จะหยุดเพื่อกัน inference เพี้ยน

## 7) Project Files (Key)
- Runtime server: `app.py`
- Feature extractor: `feature_utils.py`
- Main UI: `templates/index.html`
- Calibration UI: `templates/calibration.html`
- Dataset conversion: `collect_from_video.py`
- Training script: `train_model.py`

## 8) Practical Notes
- คุณภาพโมเดลขึ้นกับความสมดุลของคลาสอย่างมาก
- หากคลาสบางคำมีคลิปเยอะเกินไป โมเดลจะ bias ไปคลาสนั้น
- แนะนำคุมจำนวน sequence ต่อคลาสให้ใกล้เคียงกันก่อนเทรน
