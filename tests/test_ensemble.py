"""Tests for the deep ensemble wrapper.

The failure that matters here is silent and total: if the per-member seed does
not reach the initialisation, every member converges to the same weights, the
spread is exactly zero, and the epistemic term reads as perfect confidence
everywhere. Nothing raises, the numbers look clean, and the entire project's
headline quantity is a constant.

Small ensembles and short runs throughout. These assert wiring, not accuracy.
"""

import numpy as np
import pytest

from constellaration_uq.ensemble import N_MEMBERS, combine, train_ensemble
from constellaration_uq.nets import split_validation

N_INPUTS = 80


@pytest.fixture
def fit_and_val():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(700, N_INPUTS))
    y = 2.0 * X[:, 0] - X[:, 3] + rng.normal(scale=0.05, size=700)
    return split_validation(X, y * 3.0 + 10.0, n_val=200, seed=0)


def quick(fit_and_val, **kwargs):
    X_fit, y_fit, X_val, y_val = fit_and_val
    options = {'n_members': 3, 'max_epochs': 8, 'patience': 8, 'device': 'cpu'}
    return train_ensemble(X_fit, y_fit, X_val, y_val, **{**options, **kwargs})


def test_predict_returns_one_row_per_member(fit_and_val):
    """Shape (n_members, n_points), not a pre-averaged vector. Averaging inside
    the wrapper would discard the spread, which is the epistemic term."""
    _, _, X_val, _ = fit_and_val
    predict, _ = quick(fit_and_val, n_members=4)

    assert predict(X_val).shape == (4, len(X_val))


def test_members_are_not_identical(fit_and_val):
    """The one that matters. Identical members give zero spread, which reads as
    perfect confidence rather than as a broken seed (4.3)."""
    _, _, X_val, _ = fit_and_val
    predict, _ = quick(fit_and_val)

    spread = predict(X_val).std(axis=0, ddof=1)

    assert (spread > 0).all(), 'some points have zero spread across members'
    assert spread.mean() > 1e-4, 'members are nearly identical, check the seeding'


def test_members_still_agree_roughly(fit_and_val):
    """The mirror of the test above. Diversity has to come from initialisation
    and shuffle order, not from members failing to train at all, so the spread
    must stay small next to the signal."""
    _, _, X_val, y_val = fit_and_val
    predict, _ = quick(fit_and_val, max_epochs=40, patience=40)

    spread = predict(X_val).std(axis=0, ddof=1)

    assert spread.mean() < y_val.std()


def test_base_seed_is_reproducible(fit_and_val):
    _, _, X_val, _ = fit_and_val

    first, _ = quick(fit_and_val, base_seed=0)
    again, _ = quick(fit_and_val, base_seed=0)
    other, _ = quick(fit_and_val, base_seed=100)

    assert np.allclose(first(X_val), again(X_val))
    assert not np.allclose(first(X_val), other(X_val))


def test_histories_are_per_member(fit_and_val):
    """Kept rather than summarised, because a member that stopped after two
    epochs is a broken run that an aggregate would hide."""
    _, histories = quick(fit_and_val, n_members=4, max_epochs=6)

    assert len(histories) == 4
    assert all(len(h) == 6 for h in histories)


def test_progress_callback_reports_each_member(fit_and_val):
    seen = []
    quick(fit_and_val, n_members=3, progress=lambda k, epochs, best: seen.append(k))

    assert seen == [0, 1, 2]


def test_default_member_count_matches_the_paper():
    """Appendix A.4 used ten. Changing this silently would break the one
    comparison the MSE ensemble exists to make (4.2)."""
    assert N_MEMBERS == 10


def test_combine_returns_mean_and_sample_spread():
    predictions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    mean, spread = combine(predictions)

    assert np.allclose(mean, [3.0, 4.0])
    assert np.allclose(spread, np.std(predictions, axis=0, ddof=1))


def test_combine_uses_ddof_one():
    """ddof=0 would understate the spread by about 5% at ten members, which is
    small but biased toward looking more confident than the evidence supports."""
    predictions = np.array([[0.0], [2.0]])

    _, spread = combine(predictions)

    assert spread[0] == pytest.approx(np.sqrt(2.0))
