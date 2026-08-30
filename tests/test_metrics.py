"""Tests for the scoring functions and the shared distance binning.

The bug class these exist for is quiet arithmetic. A dropped point, an off-by-
one in the edge lookup, a halved argument to ppf or a sign error in the CRPS
closed form all still produce a plottable curve with plausible numbers on it.
Nothing downstream would notice, because there is no reference implementation to
disagree with and the resulting figures would look exactly as expected.

The strategy for the scoring functions is synthetic data with a known answer:
draw truths from the very distribution the model claims to predict, then assert
the diagnostics say "calibrated". Anything that reads miscalibrated on data that
is calibrated by construction is a bug in the diagnostic, not a finding.
"""

from itertools import pairwise

import numpy as np
import pytest

from constellaration_uq.metrics import (
    CRPS_AT_MEAN,
    coverage,
    coverage_curve,
    crps_gaussian,
    distance_bins,
    pit_values,
    rms_uncertainty,
)

LEVELS = np.array([0.5, 0.68, 0.8, 0.9, 0.95, 0.99])


@pytest.fixture
def calibrated():
    """A perfectly calibrated prediction: truths drawn from the claimed Gaussian.

    Heteroscedastic on purpose. A homoscedastic fixture would pass even if a
    function used a single global sigma somewhere instead of the per-point one,
    which is a live risk given every array here is the same length.
    """
    rng = np.random.default_rng(0)
    n = 20000
    mean = rng.normal(0.0, 3.0, n)
    variance = rng.uniform(0.25, 4.0, n)
    y_true = mean + rng.normal(0.0, 1.0, n) * np.sqrt(variance)
    return y_true, mean, variance


def test_rms_uncertainty_equals_rmse_when_calibrated(calibrated):
    """The property the whole calibration ratio rests on, asserted directly.

    On data drawn from the claimed distribution, the aggregated uncertainty must
    equal the realised RMSE. This is the test that would have caught the
    mean(sqrt(variance)) aggregation the project used until 2026-08-30.
    """
    y_true, mean, variance = calibrated
    realised = np.sqrt(np.mean((y_true - mean) ** 2))
    assert rms_uncertainty(variance) == pytest.approx(realised, rel=0.02)


def test_rms_uncertainty_is_not_the_mean_standard_deviation(calibrated):
    """⚠️ Pins the distinction rather than trusting the name. The two agree when
    every point shares one variance and diverge as the spread grows, so a
    homoscedastic-only test would pass on the wrong implementation."""
    _, _, variance = calibrated
    assert rms_uncertainty(variance) > np.mean(np.sqrt(variance))


def test_rms_uncertainty_matches_the_mean_standard_deviation_when_constant():
    """The one case where the two aggregations agree. Bounds the claim above."""
    variance = np.full(500, 2.25)
    assert rms_uncertainty(variance) == pytest.approx(1.5)


def test_rms_uncertainty_adds_the_components_in_variance_space(calibrated):
    """Total is epistemic plus aleatoric as variances, which survives this
    aggregation and would not survive averaging standard deviations."""
    _, _, epistemic = calibrated
    aleatoric = epistemic * 0.5 + 0.1
    total = rms_uncertainty(epistemic + aleatoric)
    expected = np.sqrt(rms_uncertainty(epistemic) ** 2 + rms_uncertainty(aleatoric) ** 2)
    assert total == pytest.approx(expected)


def test_pit_is_uniform_when_calibrated(calibrated):
    pit = pit_values(*calibrated)
    assert pit.mean() == pytest.approx(0.5, abs=0.01)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert np.quantile(pit, q) == pytest.approx(q, abs=0.02)


def test_pit_is_symmetric_about_the_mean(calibrated):
    """Reflecting a truth through its own predicted mean must send its PIT to
    1 minus itself. Catches a sign error that a uniformity check would not, since
    a systematically flipped PIT is still uniform."""
    _, mean, variance = calibrated
    y = mean + 1.7
    mirrored = mean - 1.7
    assert pit_values(y, mean, variance) == pytest.approx(1 - pit_values(mirrored, mean, variance))


def test_pit_piles_at_the_ends_when_overconfident(calibrated):
    """The signature the aggregate PIT histogram exists to show. Quarter the
    variance and the truths start falling outside the intervals."""
    y_true, mean, variance = calibrated
    pit = pit_values(y_true, mean, variance / 4)
    in_tails = np.mean((pit < 0.1) | (pit > 0.9))
    assert in_tails > 0.35


def test_pit_piles_in_the_middle_when_underconfident(calibrated):
    y_true, mean, variance = calibrated
    pit = pit_values(y_true, mean, variance * 9)
    assert np.mean(np.abs(pit - 0.5) < 0.1) > 0.4


def test_coverage_hits_nominal_when_calibrated(calibrated):
    for level, measured in zip(LEVELS, coverage_curve(*calibrated, LEVELS), strict=True):
        assert measured == pytest.approx(level, abs=0.015)


def test_coverage_matches_the_pit_route(calibrated):
    """⚠️ The independent second path. coverage halves inside its ppf argument
    and this route does not, so a dropped or doubled half shows up here as a
    mismatch. Same numbers, two derivations."""
    y_true, mean, variance = calibrated
    pit = pit_values(y_true, mean, variance)
    for level in LEVELS:
        from_pit = np.mean((pit >= (1 - level) / 2) & (pit <= (1 + level) / 2))
        assert coverage(y_true, mean, variance, level) == pytest.approx(from_pit, abs=1e-9)


def test_coverage_is_monotone_in_level(calibrated):
    measured = coverage_curve(*calibrated, LEVELS)
    assert np.all(np.diff(measured) >= 0)


def test_coverage_falls_below_nominal_when_overconfident(calibrated):
    """The project's actual finding, in miniature. Predicted variance too small
    means the reported 90% interval holds far less than 90% of the truth."""
    y_true, mean, variance = calibrated
    assert coverage(y_true, mean, variance / 4, 0.9) < 0.7


def test_crps_at_the_mean_is_the_known_constant(calibrated):
    """Fixed point of the closed form. z = 0 kills the first term and leaves
    sigma * (2 * phi(0) - 1 / sqrt(pi)), about 0.2337 * sigma. A sign error or a
    dropped term moves this."""
    _, mean, variance = calibrated
    assert CRPS_AT_MEAN == pytest.approx(0.2336949, abs=1e-6)
    expected = CRPS_AT_MEAN * np.sqrt(variance)
    assert crps_gaussian(mean, mean, variance) == pytest.approx(expected)


def test_crps_collapses_to_absolute_error_as_variance_vanishes():
    """Confirms the units. CRPS is reported in the target's physical units and
    is directly comparable to MAE, so a point prediction with a vanishing
    interval must score its own absolute error and nothing else."""
    y_true = np.array([1.0, -2.0, 0.5])
    mean = np.array([0.0, 0.0, 0.0])
    tiny = np.full(3, 1e-12)
    assert crps_gaussian(y_true, mean, tiny) == pytest.approx(np.abs(y_true), abs=1e-5)


def test_crps_is_minimised_at_the_true_variance():
    """The property that makes CRPS worth using instead of MAE: it rewards an
    honest error bar rather than a small one. Sweeping the claimed variance with
    the truths fixed must bottom out at the variance that generated them."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(0.0, 1.0, 40000)
    mean = np.zeros_like(y_true)
    claimed = np.array([0.09, 0.25, 0.64, 1.0, 1.44, 4.0, 9.0])
    scores = [crps_gaussian(y_true, mean, np.full_like(y_true, v)).mean() for v in claimed]
    assert int(np.argmin(scores)) == 3


def test_crps_is_never_negative(calibrated):
    assert (crps_gaussian(*calibrated) >= 0).all()


def test_crps_scales_with_the_units_of_the_target(calibrated):
    """Rescaling the target by c rescales CRPS by c, since variance carries c
    squared. ⚠️ This is the guard against the y_std versus y_std squared error
    that nets.py flags as the most likely silent unit bug in the codebase."""
    y_true, mean, variance = calibrated
    c = 7.0
    scaled = crps_gaussian(y_true * c, mean * c, variance * c**2)
    assert scaled == pytest.approx(crps_gaussian(y_true, mean, variance) * c)


def test_scoring_rejects_non_positive_variance():
    y_true, mean = np.array([1.0, 2.0]), np.array([0.0, 0.0])
    for bad in (np.array([1.0, 0.0]), np.array([1.0, -1.0]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError, match='strictly positive'):
            crps_gaussian(y_true, mean, bad)
        with pytest.raises(ValueError, match='strictly positive'):
            pit_values(y_true, mean, bad)


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
    for lower, upper in pairwise(rows):
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
