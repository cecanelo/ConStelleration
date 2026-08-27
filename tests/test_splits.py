"""Tests for the three splits (3.8) and the shared distance formula (3.11).

The bug class these exist for is silent: a row landing in both masks is
test-set leakage that raises nothing and makes results look better than they
are. Same category as the empty-array bug in data.py.
"""

import numpy as np
import pytest

from constellaration_uq.splits import (
    distance_from_training_region,
    hole_split,
    random_split,
    tail_split,
)


@pytest.fixture
def axis():
    """A skewed axis, deliberately. Aspect ratio is skewed in the real pool, and
    a uniform fixture would hide a value-space-versus-percentile regression."""
    return np.random.default_rng(0).gamma(shape=9.0, scale=1.0, size=2000)


def all_splits(axis):
    return {
        'random': random_split(axis, seed=0),
        'tail_low': tail_split(axis, 'low'),
        'tail_high': tail_split(axis, 'high'),
        'hole': hole_split(axis),
    }


@pytest.mark.parametrize('name', ['random', 'tail_low', 'tail_high', 'hole'])
def test_masks_are_complementary(axis, name):
    """Every row in exactly one mask. Overlap is silent test-set leakage."""
    train, test = all_splits(axis)[name]

    assert train.dtype == bool and test.dtype == bool
    assert len(train) == len(test) == len(axis)
    assert (train | test).all(), 'some rows are in neither mask'
    assert not (train & test).any(), 'some rows are in both masks'


@pytest.mark.parametrize('name', ['random', 'tail_low', 'tail_high', 'hole'])
def test_held_out_fraction(axis, name):
    train, test = all_splits(axis)[name]
    assert test.sum() == pytest.approx(0.2 * len(axis), rel=0.02)


def test_tail_low_holds_out_the_low_end(axis):
    train, test = tail_split(axis, 'low')
    assert axis[test].max() <= axis[train].min()


def test_tail_high_holds_out_the_high_end(axis):
    train, test = tail_split(axis, 'high')
    assert axis[test].min() >= axis[train].max()


def test_tail_direction_is_validated():
    with pytest.raises(ValueError, match='low'):
        tail_split(np.arange(10.0), 'lo')


def test_hole_is_surrounded_by_training_data(axis):
    """The defining property: training data on BOTH sides of the gap.

    This is what separates the hole from a tail, and the main figure's whole
    claim rests on the difference.
    """
    train, test = hole_split(axis)

    assert (axis[train] < axis[test].min()).any(), 'no training data below the hole'
    assert (axis[train] > axis[test].max()).any(), 'no training data above the hole'


def test_hole_is_contiguous(axis):
    """No training points strictly inside the held-out band."""
    train, test = hole_split(axis)
    lo, hi = axis[test].min(), axis[test].max()

    assert not ((axis[train] > lo) & (axis[train] < hi)).any()


def test_random_split_is_reproducible(axis):
    _, first = random_split(axis, seed=0)
    _, again = random_split(axis, seed=0)
    _, different = random_split(axis, seed=1)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, different)


def test_distance_is_zero_for_training_points(axis):
    train, _ = tail_split(axis, 'low')
    distance = distance_from_training_region(axis, train)

    assert (distance[train] == 0).all()


def test_distance_grows_into_the_tail():
    """Past the cutoff, distance should increase the further out you go."""
    axis = np.arange(100.0)
    train = axis >= 20
    distance = distance_from_training_region(axis, train, axis_std=1.0)

    held_out = distance[:20]
    assert held_out[0] > held_out[-1], 'distance should shrink toward the cutoff'
    assert np.all(np.diff(held_out) < 0), 'distance should be monotone in the tail'
    assert held_out[-1] == 1.0, 'the point adjacent to the cutoff sits one unit out'


def test_distance_peaks_at_the_hole_centre():
    """Inside a gap the nearest training point is whichever edge is closer, so
    distance rises from both edges and peaks in the middle."""
    axis = np.arange(100.0)
    train = (axis < 40) | (axis > 60)
    distance = distance_from_training_region(axis, train, axis_std=1.0)

    inside = distance[41:60]
    peak = inside.argmax()

    assert 0 < peak < len(inside) - 1, 'peak should be interior, not at an edge'
    assert np.all(np.diff(inside[: peak + 1]) > 0)
    assert np.all(np.diff(inside[peak:]) < 0)


def test_distance_is_near_zero_for_a_random_split(axis):
    """Random test points are surrounded by training data (3.11)."""
    train, test = random_split(axis, seed=0)
    distance = distance_from_training_region(axis, train)

    assert distance[test].max() < 0.1


def test_distance_normalization_is_split_independent():
    """Passing an explicit axis_std must override the computed one, which is how
    the grid pins one constant across all three splits."""
    axis = np.arange(100.0)
    train = axis >= 20

    default = distance_from_training_region(axis, train)
    explicit = distance_from_training_region(axis, train, axis_std=1.0)

    assert np.allclose(default, explicit / np.std(axis))


def test_distance_handles_points_beyond_both_ends():
    """Clipping guard: without it, pos - 1 wraps to the far end of the array and
    silently returns a wrong distance."""
    axis = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    train = np.array([False, True, True, True, False])

    distance = distance_from_training_region(axis, train, axis_std=1.0)

    assert distance[0] == 5.0
    assert distance[-1] == 5.0
