import hashlib
import os
import shutil
import numpy as np

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30


def sequence_hash(sequence_dir: str) -> str:
    hasher = hashlib.sha256()
    for idx in range(SEQUENCE_LENGTH):
        frame_path = os.path.join(sequence_dir, f"{idx}.npy")
        arr = np.load(frame_path).reshape(-1).astype(np.float32)
        hasher.update(arr.tobytes())
    return hasher.hexdigest()


def main() -> None:
    if not os.path.isdir(DATA_PATH):
        print(f"Dataset path not found: {DATA_PATH}")
        return

    total_removed = 0

    for action in sorted(os.listdir(DATA_PATH)):
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.isdir(action_path):
            continue

        seen = {}
        removed = 0

        seq_dirs = sorted(
            [d for d in os.listdir(action_path) if d.isdigit() and os.path.isdir(os.path.join(action_path, d))],
            key=lambda x: int(x),
        )

        for seq in seq_dirs:
            seq_path = os.path.join(action_path, seq)
            frame_files = [f for f in os.listdir(seq_path) if f.endswith(".npy")]
            if len(frame_files) != SEQUENCE_LENGTH:
                continue

            try:
                key = sequence_hash(seq_path)
            except Exception as exc:
                print(f"[WARN] Skip {action}/{seq}: {exc}")
                continue

            if key in seen:
                shutil.rmtree(seq_path)
                removed += 1
            else:
                seen[key] = seq

        total_removed += removed
        print(f"{action}: removed {removed} duplicates")

    print(f"\nDone. Total duplicates removed: {total_removed}")


if __name__ == "__main__":
    main()
