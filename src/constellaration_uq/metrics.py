"""Scoring functions and the shared distance binning.

Everything here is array in, array out. No file IO, no plotting, no model. This
is the part that can be silently wrong, so it is the part that gets tested.

The binning lives here rather than in a script because two figures depend on the
bins agreeing. The distance-error curve says error grows with distance and the
coverage curve says honesty falls with distance, and a reader is meant to lay
them side by side and see one story told twice. If each script derived its own
quantile edges, the two x axes would drift apart on any change to either
evaluation set, both plots would still render, and the comparison would quietly
stop being a comparison. One definition means they agree by construction.
"""

import numpy as np
from scipy.stats import norm

# 2 * phi(0) - 1 / sqrt(pi). The Gaussian CRPS at zero residual, in units of the
# predictive standard deviation. Named because it is the fixed point the CRPS
# test pins against, and a wrong sign or a dropped term moves it.
CRPS_AT_MEAN = 2 * norm.pdf(0.0) - 1 / np.sqrt(np.pi)


def _standard_deviation(variance):
    """Variance to standard deviation, with the one check worth making.

    Every function here takes VARIANCE, not standard deviation, because that
    is what ensemble.decompose returns and what the points CSVs store. Passing a
    standard deviation instead is undetectable from inside: it is positive, it is
    finite, and it produces coverage and CRPS numbers that look entirely
    plausible while being wrong by a factor that depends on the magnitude. The
    only defence is the naming, so every argument is called `variance` and every
    key it comes from ends in `_variance`.
    """
    variance = np.asarray(variance, dtype=float)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError('variance must be finite and strictly positive')
    return np.sqrt(variance)


def rms_uncertainty(variance):
    """Summarise per-point variances as one standard deviation, on RMSE's scale.

    sqrt(mean(variance)), the root mean square of the per-point standard
    deviations. Reported as a standard deviation because that is the scale the
    target lives on and the only one comparable to RMSE. The averaging happens
    in variance space, where adding epistemic and aleatoric is valid.

    NOT mean(sqrt(variance)). Two scripts computed the average standard
    deviation until 2026-08-30, which biased every calibration ratio and every
    variance-recovery ratio in the project low. The correct aggregation follows
    from what the ratio claims: a calibrated point satisfies E[(y - mu)^2] =
    sigma^2, so averaging over points gives RMSE^2 = mean(sigma^2), and the
    quantity that should equal RMSE is sqrt(mean(sigma^2)). By Jensen's
    inequality mean(sigma) sits below that, by a margin that grows with how much
    sigma varies across points, which is exactly the off-distribution case the
    project is about. It also breaks the variance-check identity outright, since
    injected noise adds in variance and the expected value is a statement about
    mean variances.

    It lives here rather than inline because the bug appeared independently in
    two scripts, which is what a missing shared definition looks like.
    """
    return float(np.sqrt(np.mean(np.asarray(variance, dtype=float))))


def pit_values(y_true, mean, variance):
    """Where each truth landed inside its own predicted distribution, on 0 to 1.

    A perfectly calibrated model spreads these uniformly. Piling up at both ends
    means the intervals are too narrow, since the truth keeps falling outside.
    Piling up in the middle means they are too wide. Leaning to one side means
    the mean is biased and the spread is a side issue, which is a different fault
    with a different fix and the reason this is worth plotting at all: no single
    calibration number separates "too narrow" from "systematically off centre".
    """
    return norm.cdf(y_true, loc=mean, scale=_standard_deviation(variance))


def coverage(y_true, mean, variance, level):
    """Fraction of truths inside the central predictive interval at `level`.

    This is the diagnostic the deferral threshold rests on. At level 0.9 a
    calibrated model returns 0.9, and the interesting number is how far below
    that it falls once the evaluation points leave the training region.

    Identical to the fraction of pit_values inside [(1-level)/2, (1+level)/2].
    Both routes are implemented, this one directly and that one in the tests,
    because the halving in the ppf argument is the easy mistake here and an
    independent second path is what catches it.
    """
    z = norm.ppf((1 + level) / 2)
    return float(np.mean(np.abs(y_true - mean) <= z * _standard_deviation(variance)))


def coverage_curve(y_true, mean, variance, levels):
    """coverage at each nominal level. A calibrated model traces the diagonal."""
    return np.array([coverage(y_true, mean, variance, level) for level in levels])


def crps_gaussian(y_true, mean, variance):
    """Per-point CRPS for a Gaussian predictive distribution, closed form.

    One score per point for the whole predictive distribution rather than for a
    single interval, so it moves when the mean improves and when an honest error
    bar tightens, and it penalises a confident wrong answer harder than a hedged
    one. Reported in the target's physical units, which means it is directly
    comparable to MAE: as the predicted spread goes to zero this collapses to
    the absolute error, so the difference from MAE is exactly the part the
    variance head is contributing.

    Also the deferral curve's y axis (decision log 8.3), which is why it lives
    here in metrics rather than in anything named for calibration.
    """
    sigma = _standard_deviation(variance)
    z = (np.asarray(y_true, dtype=float) - np.asarray(mean, dtype=float)) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def deferral_curve(crps, signal, rates):
    """Mean CRPS over all points when the top fraction by `signal` is deferred.

    Lives here rather than in deferral.py because seed_spread.py recomputes the
    same curve across replication seeds, and two independent implementations of
    a cumulative-sum index is exactly how the aggregation bug happened.

    Deferred points contribute zero, so this is the score of the hybrid system,
    surrogate plus solver, not of the surrogate on what is left. Dividing by the
    full count rather than the retained count is what makes the ends of the
    curve mean something: at rate 1 every point is solved and the score is 0.

    One cumulative sum serves every rate, so the whole curve costs one sort.
    """
    order = np.argsort(-np.asarray(signal), kind='stable')
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(crps)[order])])
    n = len(crps)
    total = cumulative[-1]
    return np.array([(total - cumulative[round(r * n)]) / n for r in rates])


def distance_bins(distance, n_bins):
    """Split points into equal-count quantile bins of distance.

    Returns (rows, masks). Each row carries only the bin geometry, count and
    edges and median distance, with no measured quantity in it. The caller adds
    its own fields using the matching mask, which is what lets the error curve
    and the calibration curves sit in identical bins while reporting different
    things.

    Equal-count rather than equal-width, so every point on a curve rests on the
    same amount of evidence and the thin far tail cannot produce a bin of two
    configurations masquerading as a measurement. Distance is heavily skewed and
    the hole and tail splits differ tenfold in range, so shared uniform edges
    would leave the hole with two usable bins and the tail with one crowded one.
    This is the same trap that flipped the verdict in the stage 2 noise floor
    check before it was rebinned.
    """
    edges = np.quantile(distance, np.linspace(0, 1, n_bins + 1))

    # searchsorted with side='right' puts the maximum past the last edge, so the
    # furthest point would be clipped back into the top bin rather than binned
    # into it. Nudging the top edge up by one float keeps that a measurement.
    edges[-1] = np.nextafter(edges[-1], np.inf)
    which = np.clip(np.searchsorted(edges, distance, side='right') - 1, 0, n_bins - 1)

    rows, masks = [], []
    for b in range(n_bins):
        sel = which == b
        if not sel.any():
            # Duplicate quantile edges collapse a bin. Happens on the random
            # split, where most distances are identical and tiny.
            continue
        rows.append(
            {
                'n': int(sel.sum()),
                'd_lo': float(edges[b]),
                'd_hi': float(edges[b + 1]),
                'd_median': float(np.median(distance[sel])),
            }
        )
        masks.append(sel)

    return rows, masks
