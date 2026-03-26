import numpy as np

POSE_KEYPOINT_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
FACE_KEYPOINT_IDS = [10, 9, 8, 6, 4, 1, 33, 263, 61, 291, 13, 14]
HAND_DISTANCE_PAIRS = [
    (0, 4),
    (0, 8),
    (0, 12),
    (0, 16),
    (0, 20),
    (4, 8),
    (4, 12),
    (4, 16),
    (4, 20),
    (8, 12),
    (12, 16),
    (16, 20),
]

STATIC_FEATURE_SIZE = ((len(POSE_KEYPOINT_IDS) + len(FACE_KEYPOINT_IDS) + 21 + 21) * 3)
VELOCITY_FEATURE_SIZE = 21 * 3 * 2
DISTANCE_FEATURE_SIZE = len(HAND_DISTANCE_PAIRS) * 2
FEATURE_SIZE = STATIC_FEATURE_SIZE + VELOCITY_FEATURE_SIZE + DISTANCE_FEATURE_SIZE


def landmark_list(landmarks):
    if landmarks is None:
        return []
    if hasattr(landmarks, "landmark"):
        return landmarks.landmark
    return landmarks


def _extract_coords(landmarks, expected_count, selected_ids=None):
    points = landmark_list(landmarks)
    coords = np.zeros((expected_count, 3), dtype=np.float32)
    if not points:
        return coords, False

    if selected_ids is None:
        if len(points) < expected_count:
            return coords, False
        for idx in range(expected_count):
            coords[idx] = [points[idx].x, points[idx].y, points[idx].z]
        return coords, True

    valid = True
    for out_idx, point_idx in enumerate(selected_ids):
        if point_idx >= len(points):
            valid = False
            break
        point = points[point_idx]
        coords[out_idx] = [point.x, point.y, point.z]

    return coords, valid


def _safe_scale(value):
    return float(value) if value and value > 1e-6 else 1.0


def _normalize_body(pose_coords, face_coords):
    pose_norm = pose_coords.copy()
    face_norm = face_coords.copy()

    if np.any(pose_coords):
        shoulder_mid = (pose_coords[1] + pose_coords[2]) / 2.0
        shoulder_width = np.linalg.norm(pose_coords[1] - pose_coords[2])
        hip_width = np.linalg.norm(pose_coords[7] - pose_coords[8])
        body_scale = _safe_scale(max(float(shoulder_width), float(hip_width)))
        pose_norm = (pose_coords - shoulder_mid) / body_scale
        if np.any(face_coords):
            face_norm = (face_coords - shoulder_mid) / body_scale

    return pose_norm, face_norm


def _normalize_hand(hand_coords):
    if not np.any(hand_coords):
        return hand_coords.copy(), False

    origin = hand_coords[0]
    base_lengths = []
    for idx in (5, 9, 13, 17):
        dist = float(np.linalg.norm(hand_coords[idx] - origin))
        if dist > 1e-6:
            base_lengths.append(dist)

    hand_scale = _safe_scale(np.mean(base_lengths) if base_lengths else 0.0)
    return (hand_coords - origin) / hand_scale, True


def _velocity(current, previous, present):
    if not present or previous is None:
        return np.zeros_like(current)
    return current - previous


def _hand_distances(hand_coords, present):
    distances = np.zeros(len(HAND_DISTANCE_PAIRS), dtype=np.float32)
    if not present:
        return distances

    for idx, (start, end) in enumerate(HAND_DISTANCE_PAIRS):
        distances[idx] = float(np.linalg.norm(hand_coords[start] - hand_coords[end]))
    return distances


class LandmarkFeatureExtractor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_pose = None
        self.prev_face = None
        self.prev_left_hand = None
        self.prev_right_hand = None

    def extract(self, results):
        pose_coords, _ = _extract_coords(
            results.pose_landmarks if results else None,
            len(POSE_KEYPOINT_IDS),
            POSE_KEYPOINT_IDS,
        )
        face_coords, _ = _extract_coords(
            results.face_landmarks if results else None,
            len(FACE_KEYPOINT_IDS),
            FACE_KEYPOINT_IDS,
        )
        left_hand_coords, left_present = _extract_coords(
            results.left_hand_landmarks if results else None,
            21,
        )
        right_hand_coords, right_present = _extract_coords(
            results.right_hand_landmarks if results else None,
            21,
        )

        pose_detected = bool(np.any(pose_coords))
        face_detected = bool(np.any(face_coords))

        pose_norm, face_norm = _normalize_body(pose_coords, face_coords)
        left_norm, left_present = _normalize_hand(left_hand_coords)
        right_norm, right_present = _normalize_hand(right_hand_coords)

        # Missing landmark handling: carry-forward from the last reliable frame.
        if not pose_detected and self.prev_pose is not None:
            pose_norm = self.prev_pose.copy()
        if not face_detected and self.prev_face is not None:
            face_norm = self.prev_face.copy()

        left_carried = False
        right_carried = False
        if not left_present and self.prev_left_hand is not None:
            left_norm = self.prev_left_hand.copy()
            left_carried = True
        if not right_present and self.prev_right_hand is not None:
            right_norm = self.prev_right_hand.copy()
            right_carried = True

        left_velocity = _velocity(left_norm, self.prev_left_hand, left_present)
        right_velocity = _velocity(right_norm, self.prev_right_hand, right_present)

        left_distances = _hand_distances(left_norm, left_present or left_carried)
        right_distances = _hand_distances(right_norm, right_present or right_carried)

        self.prev_pose = pose_norm.copy()
        self.prev_face = face_norm.copy()

        self.prev_left_hand = left_norm.copy() if (left_present or left_carried) else None
        self.prev_right_hand = right_norm.copy() if (right_present or right_carried) else None

        feature_vector = np.concatenate(
            [
                pose_norm.reshape(-1),
                face_norm.reshape(-1),
                left_norm.reshape(-1),
                right_norm.reshape(-1),
                left_velocity.reshape(-1),
                right_velocity.reshape(-1),
                left_distances,
                right_distances,
            ]
        ).astype(np.float32)

        if feature_vector.shape[0] != FEATURE_SIZE:
            raise ValueError(
                f"Feature size mismatch: expected {FEATURE_SIZE}, got {feature_vector.shape[0]}"
            )

        return feature_vector