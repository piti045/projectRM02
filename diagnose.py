import os
import numpy as np

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30
FEATURE_SIZE = 189

print("=== Dataset Summary ===")
total = 0
for action in sorted(os.listdir(DATA_PATH)):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        continue

    seqs = []
    bad = []
    for s in sorted(os.listdir(action_path)):
        sp = os.path.join(action_path, s)
        if not os.path.isdir(sp) or not s.isdigit():
            continue
        frames = [f for f in os.listdir(sp) if f.endswith(".npy")]
        if len(frames) == SEQUENCE_LENGTH:
            arr = np.load(os.path.join(sp, "0.npy"))
            feat = arr.reshape(-1).shape[0]
            if feat == FEATURE_SIZE:
                seqs.append(s)
            else:
                bad.append(f"{s}(feat={feat})")
        else:
            bad.append(f"{s}(frames={len(frames)})")

    bad_str = " | BAD: " + str(bad) if bad else ""
    print(f"  {action}: {len(seqs)} valid seqs{bad_str}")
    total += len(seqs)

print(f"\nTotal: {total} sequences across {len([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])} classes")

print("\n=== Note on top-level .npy files ===")
print("Top-level files like action/video_offset.npy are legacy clips and not used for training.")
print("Training only reads dataset/action/<sequence_id>/<frame>.npy")

# Quick check: deduplicate content (detect double-run sequences)
print("\n=== Checking for duplicate sequences ===")
for action in sorted(os.listdir(DATA_PATH)):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        continue
    seq_hashes = {}
    dups = 0
    for s in sorted(os.listdir(action_path)):
        sp = os.path.join(action_path, s)
        if not os.path.isdir(sp) or not s.isdigit():
            continue
        frames = [f for f in os.listdir(sp) if f.endswith(".npy")]
        if len(frames) != SEQUENCE_LENGTH:
            continue
        # Hash first+last frame sum
        try:
            first = np.load(os.path.join(sp, "0.npy")).sum()
            last = np.load(os.path.join(sp, f"{SEQUENCE_LENGTH-1}.npy")).sum()
            key = (round(float(first), 4), round(float(last), 4))
            if key in seq_hashes:
                dups += 1
            else:
                seq_hashes[key] = s
        except Exception:
            pass
    if dups > 0:
        print(f"  {action}: {dups} duplicate sequences found")
    else:
        print(f"  {action}: no duplicates")
