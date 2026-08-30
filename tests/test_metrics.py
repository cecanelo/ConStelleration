"""Tests for the shared distance binning.

The bug class these exist for is quiet arithmetic. A dropped point, an off-by-
one in the edge lookup or a bin whose reported edges do not match the points
inside it all still produce a plottable curve with plausible numbers on it. The
furthest point is the one most at risk, because it sits exactly on the top
quantile edge and is also the point the whole extrapolation story rests on.
"""

import numpy as np
import pytest

from constellaration_uq.metrics import distance_bins


@pytest.fixture
def distance():
    """Skewed, like the real thing. Distance from the training region is heavily
    right-tailed, and a uniform fixture would hide a quantile-versus-width bug."""
    return np.random.default_rng(0).gamma(shape=2.0, scale=1.0, size=1000)


def test_rows_and_masks_are_parallel(distance):
    rows, masks = distance_bins(distance, 8)
    assert len(rows) == len(masks)


def test_every_point_lands_in_exactly_one_bin(distance):
    _, masks = distance_bins(distance, 8)
    stacked = np.stack(masks)
    assert (stacked.sum(axis=0) == 1).all()


def test_counts_sum_to_the_input(distance):
    rows, _ = distance_bins(distance, 8)
    assert sum(row['n'] for row in rows) == len(distance)


def test_reported_count_matches_its_mask(distance):
    rows, masks = distance_bins(distance, 8)
    for row, sel in zip(rows, masks, strict=True):
        assert row['n'] == int(sel.sum())


def test_bins_are_equal_count_to_within_one(distance):
    """The reason for quantile edges in the first place. A width-based bug shows
    up here as a top bin holding a handful of points."""
    rows, _ = distance_bins(distance, 8)
    counts = [row['n'] for row in rows]
    assert max(counts) - min(counts) <= 1


def test_points_lie_inside_their_reported_edges(distance):
    rows, masks = distance_bins(distance, 8)
    for row, sel in zip(rows, masks, strict=True):
        inside = distance[sel]
        assert inside.min() >= row['d_lo']
        assert inside.max() < row['d_hi']


def test_furthest_point_is_binned_not_clipped(distance):
    """⚠️ The nextafter guard. searchsorted with side='right' puts the maximum
    past the last edge, and the clip would then fold it back into the top bin
    without it ever being counted as inside. Same answer here, different reason,
    so this asserts the edge brackets it rather than that it merely landed."""
    rows, masks = distance_bins(distance, 8)
    furthest = distance.argmax()
    assert masks[-1][furthest]
    assert distance[furthest] < rows[-1]['d_hi']


def test_median_matches_the_points_in_the_bin(distance):
    rows, masks = distance_bins(distance, 8)
    for row, sel in zip(rows, masks, strict=True):
        assert row['d_median'] == pytest.approx(np.median(distance[sel]))


def test_edges_are_contiguous(distance):
    """No gap between one bin's top and the next one's bottom, so the curve has
    no invisible hole in it."""
    rows, _ = distance_bins(distance, 8)
    for lower, upper in zip(rows, rows[1:], strict=False):
        assert lower['d_hi'] == pytest.approx(upper['d_lo'])


def test_hand_computed_example():
    rows, masks = distance_bins(np.arange(8.0), 4)
    assert [row['n'] for row in rows] == [2, 2, 2, 2]
    assert [row['d_median'] for row in rows] == [0.5, 2.5, 4.5, 6.5]
    assert masks[0].tolist()[:2] == [True, True]


def test_collapsed_bins_are_dropped_not_reported_empty():
    """Duplicate quantile edges. The random split does this, where nearly every
    distance is identical and tiny. An empty bin would be a divide by zero in
    every caller's aggregation, so it is skipped here rather than guarded there."""
    rows, masks = distance_bins(np.zeros(100), 8)
    assert len(rows) == 1
    assert rows[0]['n'] == 100
    assert masks[0].all()


def test_partially_degenerate_input_keeps_the_spread_points():
    """Half the points identical, half spread out. The identical half collapses
    and the rest must still resolve, which is the realistic version of the case
    above."""
    values = np.concatenate([np.zeros(50), np.linspace(1.0, 5.0, 50)])
    rows, _ = distance_bins(values, 8)
    assert 1 < len(rows) <= 8
    assert sum(row['n'] for row in rows) == 100
