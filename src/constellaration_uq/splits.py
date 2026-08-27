"""Random, interior-hole and tail splits over the chosen split axis.

Three splits, same model and recipe for each (3.8). Random is the in-domain
baseline. The interior hole tests whether the model can fill a gap surrounded by
training data. The tail is the actual extrapolation condition. The distance
between the hole result and the tail result is the main figure.

Every function takes the axis values and returns (train_mask, test_mask) as
boolean arrays, so the day 1-2 grid can call them interchangeably.
"""

import numpy as np


def random_split(axis, seed, test_fraction=0.2):
    rng = np.random.default_rng(seed)
    n = len(axis)
    test = np.zeros(n, dtype=bool)
    test[rng.choice(n, size=round(n * test_fraction), replace=False)] = True
    return ~test, test


def tail_split(axis, direction, test_fraction=0.2):
    if direction not in ('low', 'high'):
        raise ValueError(f"direction must be 'low' or 'high', got {direction!r}")

    if direction == 'low':
        cutoff = np.quantile(axis, test_fraction)
        test = axis <= cutoff
    else:
        cutoff = np.quantile(axis, 1 - test_fraction)
        test = axis >= cutoff
    return ~test, test


def hole_split(axis, test_fraction=0.2, center_quantile=0.5):
    half = test_fraction / 2
    lo, hi = np.quantile(axis, [center_quantile - half, center_quantile + half])
    test = (axis >= lo) & (axis <= hi)
    return ~test, test


def distance_from_training_region(axis, train_mask, axis_std=None):
    """Distance along the split axis to the nearest training point, in std units (3.11).

    One formula covers all three splits. Tail: distance past the cutoff. Interior
    hole: distance from the nearer edge, peaking at the hole's centre. Random:
    near zero everywhere, which is why the random split gets no distance figure.

    Normalized by a split-independent constant, the axis std over the full
    filtered pool, so degradation at 0.5 in one split is comparable to 0.5 in
    another. Deliberately not the training region's own width, which differs
    between hole and tail and would break exactly that comparison.
    """
    if axis_std is None:
        axis_std = np.std(axis)

    train_sorted = np.sort(axis[train_mask])
    last = len(train_sorted) - 1

    pos = np.searchsorted(train_sorted, axis)
    left = train_sorted[np.clip(pos - 1, 0, last)]
    right = train_sorted[np.clip(pos, 0, last)]

    nearest = np.minimum(np.abs(axis - left), np.abs(axis - right))
    return nearest / axis_std
