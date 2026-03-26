import argparse
import os
import random

import numpy as np

from feature_utils import (
    DISTANCE_FEATURE_SIZE,
    FEATURE_SIZE,
    STATIC_FEATURE_SIZE,
    VELOCITY_FEATURE_SIZE,
)

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30


def frame_sort_key(name: str) -> int:
    stem, _ = os.path.splitext(name)
    return int(stem) if stem.isdigit() else 10**9


def list_actions():
    if not os.path.isdir(DATA_PATH):
        return []
    return sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])


def valid_sequence_dirs(action_path: str):
    valid_dirs = []
    for seq_name in os.listdir(action_path):
        seq_path = os.path.join(action_path, seq_name)
        if not os.path.isdir(seq_path) or not seq_name.isdigit():
            continue
        frame_files = [f for f in os.listdir(seq_path) if f.endswith(".npy")]
        if len(frame_files) == SEQUENCE_LENGTH:
            valid_dirs.append(seq_name)
    return sorted(valid_dirs, key=int)


def load_sequence(seq_path: str):
    frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith(".npy")], key=frame_sort_key)
    if len(frame_files) != SEQUENCE_LENGTH:
        return None

    frames = []
    for frame_file in frame_files:
        arr = np.load(os.path.join(seq_path, frame_file)).reshape(-1).astype(np.float32)
        if arr.shape[0] != FEATURE_SIZE:
            return None
        frames.append(arr)

    return np.array(frames, dtype=np.float32)


def augment_sequence(seq: np.ndarray, rng: np.random.Generator):
    static = seq[:, :STATIC_FEATURE_SIZE].copy()
    velocity = seq[:, STATIC_FEATURE_SIZE:STATIC_FEATURE_SIZE + VELOCITY_FEATURE_SIZE].copy()
    distance = seq[:, STATIC_FEATURE_SIZE + VELOCITY_FEATURE_SIZE:].copy()

    scale = float(rng.uniform(0.94, 1.06))
    shift = rng.normal(0.0, 0.015, size=(1, 1)).astype(np.float32)
    noise_static = rng.normal(0.0, 0.01, size=static.shape).astype(np.float32)
    noise_velocity = rng.normal(0.0, 0.007, size=velocity.shape).astype(np.float32)
    noise_distance = rng.normal(0.0, 0.005, size=distance.shape).astype(np.float32)

    static = (static * scale) + shift + noise_static
    velocity = (velocity * scale) + noise_velocity
    distance = np.clip((distance * abs(scale)) + noise_distance, 0.0, None)

    return np.concatenate([static, velocity, distance], axis=1).astype(np.float32)


def save_sequence(seq: np.ndarray, action_path: str, seq_id: int):
    save_path = os.path.join(action_path, str(seq_id))
    os.makedirs(save_path, exist_ok=True)

    for frame_idx, frame_data in enumerate(seq):
        np.save(os.path.join(save_path, f"{frame_idx}.npy"), frame_data)


def next_sequence_id(action_path: str):
    ids = []
    for name in os.listdir(action_path):
        full_path = os.path.join(action_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            ids.append(int(name))
    return (max(ids) + 1) if ids else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Augment low-support gesture classes")
    parser.add_argument("--target-per-class", type=int, default=10, help="Target sequences per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_per_class < 1:
        raise ValueError("target-per-class must be >= 1")

    actions = list_actions()
    if not actions:
        print("No dataset actions found")
        return

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    print("=== Dataset Augmentation ===")
    print(f"Target per class: {args.target_per_class}")

    total_added = 0

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        seq_dirs = valid_sequence_dirs(action_path)
        current = len(seq_dirs)

        if current == 0:
            print(f"- {action}: 0 sequences, skipped")
            continue

        need = max(0, args.target_per_class - current)
        if need == 0:
            print(f"- {action}: {current} sequences, no augmentation needed")
            continue

        next_id = next_sequence_id(action_path)
        added = 0

        while added < need:
            src_dir = random.choice(seq_dirs)
            src_path = os.path.join(action_path, src_dir)
            seq = load_sequence(src_path)
            if seq is None:
                continue

            aug = augment_sequence(seq, rng)
            save_sequence(aug, action_path, next_id)

            next_id += 1
            added += 1

        total_added += added
        print(f"- {action}: added {added} (from {current} to {current + added})")

    print(f"\nDone. Total augmented sequences added: {total_added}")


if __name__ == "__main__":
    main()
