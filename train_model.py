import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

DATA_PATH = "dataset"
MODEL_PATH = "model.h5"
MODEL_LABELS_PATH = os.path.join(DATA_PATH, "model_labels.json")
SEQUENCE_LENGTH = 30


def frame_sort_key(name: str) -> int:
    stem, _ = os.path.splitext(name)
    return int(stem) if stem.isdigit() else 10**9


def load_dataset():
    actions = sorted(
        [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    )
    print(f"Actions: {actions}")

    label_map = {label: idx for idx, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        seq_dirs = sorted(
            [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d)) and d.isdigit()],
            key=lambda x: int(x),
        )

        for seq_dir in seq_dirs:
            seq_path = os.path.join(action_path, seq_dir)
            frame_files = sorted(
                [f for f in os.listdir(seq_path) if f.endswith(".npy")],
                key=frame_sort_key,
            )

            if len(frame_files) != SEQUENCE_LENGTH:
                continue

            frames = []
            valid = True
            for frame_file in frame_files:
                arr = np.load(os.path.join(seq_path, frame_file)).reshape(-1)
                frames.append(arr)

            if not valid:
                continue

            sequences.append(np.array(frames))
            labels.append(label_map[action])

    X = np.array(sequences)
    y_idx = np.array(labels)

    return actions, X, y_idx


def main() -> None:
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    actions, X, y_idx = load_dataset()

    if len(actions) < 2:
        raise ValueError("Need at least 2 gesture classes")

    if X.shape[0] == 0:
        raise ValueError("No valid sequence data found")

    print(f"X shape: {X.shape}")
    print(f"Total samples: {X.shape[0]}")

    class_counts = np.bincount(y_idx, minlength=len(actions))
    for i, c in enumerate(class_counts):
        print(f"  {actions[i]}: {int(c)} samples")

    y_one_hot = to_categorical(y_idx, num_classes=len(actions)).astype(int)

    can_stratify = bool(np.all(class_counts >= 2))
    if can_stratify and X.shape[0] >= len(actions) * 3:
        X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
            X,
            y_one_hot,
            y_idx,
            test_size=0.2,
            shuffle=True,
            stratify=y_idx,
            random_state=42,
        )
    else:
        X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
            X,
            y_one_hot,
            y_idx,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        print("[WARN] Stratified split disabled due to limited per-class samples")

    model = Sequential(
        [
            LSTM(128, return_sequences=True, activation="tanh", input_shape=(SEQUENCE_LENGTH, X.shape[2])),
            Dropout(0.3),
            LSTM(128, return_sequences=False, activation="tanh"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(len(actions), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print(f"\nTraining on {X_train.shape[0]} sequences...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(MODEL_PATH)
    with open(MODEL_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump([str(a) for a in actions], f, ensure_ascii=False, indent=2)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved labels: {MODEL_LABELS_PATH}")

    # Evaluate class-wise metrics
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test_idx, y_pred_idx)
    print(f"\nTest accuracy: {acc:.4f}")

    report = classification_report(
        y_test_idx,
        y_pred_idx,
        labels=list(range(len(actions))),
        target_names=actions,
        digits=4,
        zero_division=0,
    )
    print("\nClassification report:")
    print(report)

    cm = confusion_matrix(y_test_idx, y_pred_idx, labels=list(range(len(actions))))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    if "val_categorical_accuracy" in history.history:
        best_val = max(history.history["val_categorical_accuracy"])
        print(f"Best val_categorical_accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()
