import os

DATA_PATH = "dataset"
SEQUENCE_LENGTH = 30
MIN_RECOMMENDED_SAMPLES = 6
IMBALANCE_RATIO_WARN = 2.5


def count_valid_sequences(action_path: str) -> int:
    total = 0
    for seq in os.listdir(action_path):
        seq_path = os.path.join(action_path, seq)
        if not os.path.isdir(seq_path):
            continue
        frame_files = [f for f in os.listdir(seq_path) if f.endswith(".npy")]
        if len(frame_files) == SEQUENCE_LENGTH:
            total += 1
    return total


def main() -> None:
    if not os.path.isdir(DATA_PATH):
        print(f"Dataset path not found: {DATA_PATH}")
        return

    actions = sorted(
        [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    )
    if not actions:
        print("No class folders found in dataset")
        return

    counts = {}
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        counts[action] = count_valid_sequences(action_path)

    print("=== Dataset Balance Check ===")
    for action in actions:
        print(f"  {action}: {counts[action]} valid sequences")

    non_zero = [v for v in counts.values() if v > 0]
    if not non_zero:
        print("\nERROR: no valid sequences found")
        return

    min_count = min(non_zero)
    max_count = max(non_zero)
    ratio = (max_count / min_count) if min_count > 0 else float("inf")

    low_support = [k for k, v in counts.items() if v < MIN_RECOMMENDED_SAMPLES]

    print("\n=== Summary ===")
    print(f"Classes: {len(actions)}")
    print(f"Total valid sequences: {sum(counts.values())}")
    print(f"Min non-zero per class: {min_count}")
    print(f"Max per class: {max_count}")
    print(f"Imbalance ratio (max/min): {ratio:.2f}")

    warnings = []
    if low_support:
        warnings.append(
            f"Low-support classes (<{MIN_RECOMMENDED_SAMPLES}): "
            + ", ".join(f"{k}={counts[k]}" for k in low_support)
        )
    if ratio > IMBALANCE_RATIO_WARN:
        warnings.append(
            f"Imbalance ratio {ratio:.2f} is higher than recommended {IMBALANCE_RATIO_WARN:.2f}"
        )

    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings:
            print(f"- {w}")
        print("\nRecommendation: collect more sequences for low-support classes and retrain.")
    else:
        print("\nDataset balance looks good.")


if __name__ == "__main__":
    main()
