import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "dataset"

# ===== โหลด actions จากชื่อโฟลเดอร์ =====
actions = sorted(np.array([d for d in os.listdir(DATA_PATH) 
                           if os.path.isdir(os.path.join(DATA_PATH, d))]))
print(f"✅ Actions: {actions}")

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# ===== โหลดข้อมูลจาก dataset/action/seq/frame.npy =====
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    
    # อ่าน sequences (subdirectories ภายใน action)
    seq_dirs = sorted([d for d in os.listdir(action_path)
                       if os.path.isdir(os.path.join(action_path, d))])
    
    for seq_dir in seq_dirs:
        seq_path = os.path.join(action_path, seq_dir)
        frames = []
        
        # อ่าน frames (.npy files) ภายใน sequence
        frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.npy')],
                            key=lambda x: int(x.replace('.npy', '')))
        
        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)
            frame_data = np.load(frame_path)
            frames.append(frame_data)
        
        # ต้องมี 30 frames พอดี
        if len(frames) == 30:
            sequence = np.array(frames)  # Shape: (30, 375)
            sequences.append(sequence)
            labels.append(label_map[action])
        else:
            print(f"⚠️  {action}/{seq_dir} มี {len(frames)} frames (ต้อง 30) - ข้าม")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

if X.shape[0] == 0:
    print("❌ ไม่มี sequences ถูกโหลด! ตรวจสอบโครงสร้าง dataset")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(30, X.shape[2])),
    Dropout(0.3),
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.summary()

print(f"\n🚀 Training on {X_train.shape[0]} sequences...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

model.save("model.h5")
print("✅ Model saved as model.h5")