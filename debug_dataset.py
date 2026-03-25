"""
🔍 สคริปต์เพื่อตรวจสอบปัญหา dataset
- โครงสร้างของ dataset
- จำนวนคำและไฟล์
- ตรวจสอบความสม่ำเสมอของ keypoints
"""

import os
import numpy as np

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 189

print("=" * 60)
print("📋 ตรวจสอบ Dataset Structure")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    print(f"❌ Folder '{DATA_PATH}' ไม่มี!")
    exit()

# ===== อ่านรายชื่อ action (โฟลเดอร์)
actions = sorted([d for d in os.listdir(DATA_PATH) 
                  if os.path.isdir(os.path.join(DATA_PATH, d))])

print(f"\n✅ Actions found: {actions}")
print(f"   Total: {len(actions)} คำ\n")

# ===== ตรวจสอบแต่ละ action
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    print(f"\n{'='*60}")
    print(f"📁 Action: {action}")
    print(f"{'='*60}")
    
    # โครงสร้างที่ 1: dataset/action/*.npy (ระดับเดียว)
    npy_files_level1 = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    
    # โครงสร้างที่ 2: dataset/action/*/frame_num.npy (ลึก 2 ระดับ)
    npy_files_level2 = []
    subdirs = [d for d in os.listdir(action_path) 
               if os.path.isdir(os.path.join(action_path, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(action_path, subdir)
        npy_files_in_subdir = [f for f in os.listdir(subdir_path) if f.endswith('.npy')]
        npy_files_level2.extend(npy_files_in_subdir)
    
    print(f"  ├─ .npy files ที่ระดับ 1 (action/ ): {len(npy_files_level1)}")
    print(f"  ├─ Subdirectories: {len(subdirs)}")
    if subdirs:
        print(f"  │  ├─ Names: {subdirs[:5]}" + ("..." if len(subdirs) > 5 else ""))
    print(f"  └─ .npy files ที่ระดับ 2 (action/*/): {len(npy_files_level2)}")
    
    # ตรวจสอบ keypoints consistency
    if npy_files_level1:
        print(f"\n  ⚠️  พบ .npy ที่ระดับ 1 - ตรวจสอบ shape:")
        for f in npy_files_level1[:3]:
            data = np.load(os.path.join(action_path, f))
            print(f"     - {f}: shape {data.shape}")
    
    if npy_files_level2:
        print(f"\n  ✅ พบ .npy ที่ระดับ 2 - ตรวจสอบ shape:")
        sample_path = os.path.join(action_path, subdirs[0], npy_files_level2[0])
        data = np.load(sample_path)
        keypoint_size = data.shape[0]
        print(f"     - Keypoint size: {keypoint_size}")
        
        # ตรวจสอบตามสเปกปัจจุบันของแอป (189)
        if keypoint_size == FEATURE_SIZE:
            print(f"     ✅ Correct (expected {FEATURE_SIZE})")
        else:
            print(f"     ❌ Unexpected size (expected {FEATURE_SIZE})")

print("\n" + "=" * 60)
print("💾 ตรวจสอบ train_model.py data loading")
print("=" * 60)

# จำลองว่า train_model.py จะโหลดยังไง (แบบเดียวกับไฟล์ที่แก้ไขแล้ว)
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(actions)}

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    # อ่านจาก subdirectories (seq numbers)
    seq_dirs = sorted([d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))])
    for seq in seq_dirs:
        seq_path = os.path.join(action_path, seq)
        frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.npy')],
                             key=lambda x: int(x.replace('.npy', '')))
        if len(frame_files) == SEQUENCE_LENGTH:
            frames = [np.load(os.path.join(seq_path, f)) for f in frame_files]
            sequences.append(np.array(frames))
            labels.append(label_map[action])
        else:
            print(f"⚠️  {action}/{seq} มี {len(frame_files)} frames (ต้อง {SEQUENCE_LENGTH}) - ข้าม")

print(f"\n✅ Sequences loaded: {len(sequences)}")
print(f"✅ Labels loaded: {len(labels)}")

if sequences:
    X = np.array(sequences)
    print(f"✅ X shape: {X.shape} (should be [num_sequences, {SEQUENCE_LENGTH}, {FEATURE_SIZE}])")
    if len(X.shape) >= 2:
        if X.shape[1] != SEQUENCE_LENGTH:
            print(f"   ⚠️  WARNING: 2nd dimension is {X.shape[1]}, expected {SEQUENCE_LENGTH}")
        if len(X.shape) >= 3 and X.shape[2] != FEATURE_SIZE:
            print(f"   ⚠️  WARNING: 3rd dimension is {X.shape[2]}, expected {FEATURE_SIZE}")

print("\n" + "=" * 60)
print("📝 สรุปปัญหา:")
print("=" * 60)
if len(sequences) == 0:
    print("❌ ไม่มี sequences ถูกโหลด!")
    print("   → ตรวจสอบโครงสร้าง dataset อีกครั้ง")
    print("   → collect_data.py บันทึกลึกกว่า train_model.py หรือไม่?")
else:
    print(f"✅ ค่อนข้างออเค - โหลด {len(sequences)} sequences")
