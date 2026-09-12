from __future__ import annotations

from collections import defaultdict

import numpy as np


def labelset_smote(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    k_neighbors: int = 5,
    target: str | int = "max",
    noise_scale: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Training-only label-set SMOTE for multi-label data.

    Samples are interpolated only within identical label sets, so no synthetic
    sample mixes incompatible syndrome-label combinations.
    """

    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(y):
        groups[tuple(row.astype(int).tolist())].append(idx)

    if target == "max":
        target_count = max(len(v) for v in groups.values())
    else:
        target_count = int(target)

    synthetic_x: list[np.ndarray] = []
    synthetic_y: list[np.ndarray] = []
    for label_key, idxs in groups.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        count = len(idxs)
        if count >= target_count:
            continue
        needed = target_count - count
        for _ in range(needed):
            anchor = int(rng.choice(idxs))
            if count > 1:
                neighbor_pool = idxs[idxs != anchor]
                neighbor = int(rng.choice(neighbor_pool))
                lam = float(rng.random())
                sample = x[anchor] + lam * (x[neighbor] - x[anchor])
            else:
                scale = np.maximum(np.std(x, axis=0), 1e-6)
                sample = x[anchor] + rng.normal(0.0, noise_scale, size=x.shape[1]) * scale
            synthetic_x.append(sample.astype(np.float32))
            synthetic_y.append(np.asarray(label_key, dtype=np.float32))

    if not synthetic_x:
        return x, y
    return (
        np.vstack([x, np.vstack(synthetic_x).astype(np.float32)]),
        np.vstack([y, np.vstack(synthetic_y).astype(np.float32)]),
    )

