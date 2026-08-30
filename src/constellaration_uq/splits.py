"""Random, interior-hole and tail splits over the chosen split axis.

Three splits, same model and recipe for each (3.8). Random is the in-domain
baseline. The interior hole tests whether the model can fill a gap surrounded by
training data. The tail is the actual extrapolation condition. The distance
between the hole result and the tail result is the main figure.

Every function takes the axis values and returns (train_mask, test_mask) as
boolean arrays, so the day 1-2 grid can call them interchangeably.
"""

from itertools import pairwise

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


def nested_subsamples(n_pool, sizes, seed):
    """Training subsets for the N-sweep, each nested inside the next larger one.

    Returns one index array per entry in `sizes`, drawn from a single shuffle of
    range(n_pool), so the N = 2000 set is the N = 1000 set plus 1000 more rows
    rather than an independent draw.

    ⚠️ Nesting is the point, not an implementation convenience. The sweep asks
    whether epistemic uncertainty shrinks as data is added. With independent
    draws per size, the difference between two rungs mixes "more data" with
    "different data", and at the small rungs the second term is large: 1000 rows
    out of 16,793 is a 6% sample, so two draws can differ substantially in which
    part of the training region they cover. Nesting removes that term, leaving
    added data as the only thing that changed. The three seeds then vary the
    whole ladder together, which is the variation the sweep actually wants to
    report.

    Subsampling happens within the training region only, so N varies and
    coverage of the region does not.
    """
    sizes = list(sizes)
    if any(b <= a for a, b in pairwise(sizes)):
        raise ValueError(f'sizes must be strictly ascending, got {sizes}')
    if sizes and sizes[-1] > n_pool:
        raise ValueError(f'largest size {sizes[-1]} exceeds pool of {n_pool}')
    if sizes and sizes[0] < 1:
        raise ValueError(f'sizes must be positive, got {sizes}')

    order = np.random.default_rng(seed).permutation(n_pool)
    return [order[:size] for size in sizes]
