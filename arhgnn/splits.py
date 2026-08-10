from __future__ import annotations

from collections import defaultdict

import numpy as np


def split_train_val_test(
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create disjoint patient indices using label-set stratification.

    ``val_size`` is interpreted as the validation fraction of the non-test
    remainder. With defaults this yields 64/16/20 train/val/test.
    """

    labels = np.asarray(labels)
    n = labels.shape[0]
    all_indices = np.arange(n)
    rng = np.random.default_rng(seed)

    test_idx, remaining = _split_by_labelset(all_indices, labels, test_size, rng)
    remaining_labels = labels[remaining]
    val_local, train_local = _split_by_labelset(np.arange(len(remaining)), remaining_labels, val_size, rng)
    val_idx = remaining[val_local]
    train_idx = remaining[train_local]

    result = {
        "train_idx": np.sort(train_idx),
        "val_idx": np.sort(val_idx),
        "test_idx": np.sort(test_idx),
    }
    assert_disjoint(result)
    return result


def make_masks(n_samples: int, splits: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, indices in splits.items():
        mask = np.zeros(n_samples, dtype=bool)
        mask[np.asarray(indices, dtype=np.int64)] = True
        masks[name.replace("_idx", "_mask")] = mask
    return masks


def assert_disjoint(splits: dict[str, np.ndarray]) -> None:
    seen: set[int] = set()
    for name, indices in splits.items():
        current = set(np.asarray(indices, dtype=int).tolist())
        overlap = seen.intersection(current)
        if overlap:
            raise ValueError(f"Split {name} overlaps with another split: {sorted(overlap)[:10]}")
        seen.update(current)


def _split_by_labelset(
    indices: np.ndarray,
    labels: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for local_pos, label_row in enumerate(labels):
        groups[tuple(label_row.astype(int).tolist())].append(int(indices[local_pos]))

    selected: list[int] = []
    retained: list[int] = []
    target = int(round(len(indices) * fraction))

    for group_indices in groups.values():
        group_indices = list(group_indices)
        rng.shuffle(group_indices)
        if len(group_indices) <= 1:
            retained.extend(group_indices)
            continue
        take = int(round(len(group_indices) * fraction))
        take = min(max(take, 1 if fraction > 0 and len(group_indices) >= 4 else 0), len(group_indices) - 1)
        selected.extend(group_indices[:take])
        retained.extend(group_indices[take:])

    if len(selected) < target and retained:
        rng.shuffle(retained)
        need = min(target - len(selected), max(0, len(retained) - 1))
        selected.extend(retained[:need])
        retained = retained[need:]
    elif len(selected) > target:
        rng.shuffle(selected)
        retained.extend(selected[target:])
        selected = selected[:target]

    return np.asarray(selected, dtype=np.int64), np.asarray(retained, dtype=np.int64)

