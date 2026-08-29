"""Tests for the network and its training loop.

The bug class these exist for is silent. An ensemble whose members share an
initialisation reports zero spread, which reads as perfect confidence rather
than as a broken seed. A predict function that forgets to un-z-score returns
numbers in the right shape and the wrong units. Early stopping that keeps the
last weights instead of the best returns a model measurably worse than the one
it stopped for, and nothing raises.

Everything here pins device='cpu'. These assert behaviour, not speed, and a
GPU would make them non-deterministic for no gain.
"""

import numpy as np
import pytest
import torch

from constellaration_uq.nets import (
    MAX_EPOCHS,
    MLP,
    VAL_SIZE,
    resolve_device,
    split_validation,
    train_one,
)

N_INPUTS = 80


@pytest.fixture
def data():
    """A small learnable problem. Deliberately not pure noise: several tests
    need the model to actually fit something for the assertion to mean
    anything."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(900, N_INPUTS))
    y = 2.0 * X[:, 0] - X[:, 3] + 0.5 * X[:, 7] ** 2 + rng.normal(scale=0.05, size=900)
    # Shifted and stretched so a missing un-z-scoring cannot pass by accident.
    return X, y * 3.0 + 10.0


@pytest.fixture
def fit_and_val(data):
    X, y = data
    return split_validation(X, y, n_val=200, seed=0)


def quick(X_fit, y_fit, X_val, y_val, **kwargs):
    """Short runs. These tests check wiring, not convergence."""
    options = {'max_epochs': 12, 'patience': 12, 'device': 'cpu'}
    return train_one(X_fit, y_fit, X_val, y_val, **{**options, **kwargs})


# --- architecture ------------------------------------------------------------


def test_mlp_shape_follows_n_outputs():
    """The variance head is an output-width change, so this is the seam that
    has to hold for the mean-variance model to be a config change (4.1)."""
    x = torch.zeros(5, N_INPUTS)

    assert MLP(N_INPUTS, n_outputs=1)(x).shape == (5, 1)
    assert MLP(N_INPUTS, n_outputs=2)(x).shape == (5, 2)


def test_mlp_has_three_tanh_hidden_layers():
    """Appendix A.4 is the whole reason for this architecture, so a silent
    change to depth or activation should fail a test rather than a comparison
    against Table 7 months later."""
    layers = list(MLP(N_INPUTS).net)

    assert sum(isinstance(m, torch.nn.Tanh) for m in layers) == 3
    assert [m.out_features for m in layers if isinstance(m, torch.nn.Linear)] == [
        256,
        256,
        256,
        1,
    ]


# --- the validation split ----------------------------------------------------


def test_split_validation_is_exact_size_and_disjoint(data):
    """Fixed count, not a fraction (4.7). A fraction would make the sweep's
    stopping behaviour vary with N for a reason unrelated to data volume."""
    X, y = data
    X_fit, y_fit, X_val, y_val = split_validation(X, y, n_val=200, seed=0)

    assert len(X_val) == len(y_val) == 200
    assert len(X_fit) == len(y_fit) == len(X) - 200

    fit_rows = {row.tobytes() for row in X_fit}
    assert not any(row.tobytes() in fit_rows for row in X_val), 'validation row also in fit set'


def test_split_validation_default_is_the_frozen_size(data):
    X, y = data
    _, _, X_val, _ = split_validation(X, y)

    assert len(X_val) == VAL_SIZE


def test_split_validation_keeps_rows_paired(data):
    """A permutation applied to X but not y would silently destroy every label,
    and the model would still train, just on noise."""
    X, y = data
    X_fit, y_fit, X_val, y_val = split_validation(X, y, n_val=200, seed=0)

    for Xs, ys in ((X_fit, y_fit), (X_val, y_val)):
        for row, target in zip(Xs[:20], ys[:20], strict=True):
            match = np.flatnonzero((X == row).all(axis=1))
            assert y[match[0]] == target


def test_split_validation_is_reproducible(data):
    X, y = data
    first = split_validation(X, y, n_val=200, seed=0)[2]
    again = split_validation(X, y, n_val=200, seed=0)[2]
    other = split_validation(X, y, n_val=200, seed=1)[2]

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


# --- training ----------------------------------------------------------------


def test_predict_returns_physical_units(fit_and_val):
    """The fixture's target is shifted to a mean near 10, so a predict that
    forgot to un-z-score would answer near 0 with the correct shape."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    predict, _ = quick(X_fit, y_fit, X_val, y_val)

    predicted = predict(X_val)

    assert predicted.shape == (len(X_val),)
    assert abs(predicted.mean() - y_val.mean()) < 0.5 * y_val.std()


def test_training_actually_learns(fit_and_val):
    """Beat the mean predictor. Without this, every other test here would pass
    on a model that learned nothing."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    predict, history = quick(X_fit, y_fit, X_val, y_val, max_epochs=80, patience=80)

    rmse = np.sqrt(np.mean((y_val - predict(X_val)) ** 2))

    assert rmse < 0.6 * y_val.std()
    assert history[-1] < history[0]


def test_seed_is_reproducible(fit_and_val):
    X_fit, y_fit, X_val, y_val = fit_and_val

    first, _ = quick(X_fit, y_fit, X_val, y_val, seed=0)
    again, _ = quick(X_fit, y_fit, X_val, y_val, seed=0)

    assert np.allclose(first(X_val), again(X_val))


def test_different_seeds_give_different_members(fit_and_val):
    """Initialisation and shuffle order are the only diversity mechanism (4.3).
    If the seed does not reach them, every ensemble member is identical, the
    spread is zero, and the epistemic term reads as perfect confidence."""
    X_fit, y_fit, X_val, y_val = fit_and_val

    first, _ = quick(X_fit, y_fit, X_val, y_val, seed=0)
    other, _ = quick(X_fit, y_fit, X_val, y_val, seed=1)

    assert not np.allclose(first(X_val), other(X_val))


def test_history_has_one_entry_per_epoch(fit_and_val):
    """The 4.10 sanity check reads history to pick a learning rate and to judge
    whether 500 validation points give a smooth signal, so it has to be a real
    per-epoch record rather than a summary."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    _, history = quick(X_fit, y_fit, X_val, y_val, max_epochs=9, patience=9)

    assert len(history) == 9
    assert all(np.isfinite(history))


def test_early_stopping_returns_the_best_weights(fit_and_val):
    """Patience means training continues past the best epoch. Returning the
    last weights would hand back a model measurably worse than the one the
    stopping rule chose, and nothing would raise."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    predict, history = quick(X_fit, y_fit, X_val, y_val, max_epochs=60, patience=60)

    y_mean, y_std = y_fit.mean(), y_fit.std()
    achieved = np.mean(((y_val - predict(X_val)) / y_std) ** 2)

    assert achieved == pytest.approx(min(history), rel=0.05)
    assert min(history) <= history[-1]
    assert y_mean == pytest.approx(y_fit.mean())


def test_early_stopping_triggers_before_the_cap(fit_and_val):
    X_fit, y_fit, X_val, y_val = fit_and_val
    _, history = quick(X_fit, y_fit, X_val, y_val, max_epochs=200, patience=3)

    assert len(history) < 200, 'patience of 3 should stop well short of the cap'


def test_max_epochs_default_is_a_cap_not_a_target():
    """Guards against someone lowering the cap to speed a run up and silently
    changing the frozen recipe for every model in the project."""
    assert MAX_EPOCHS >= 200


# --- device ------------------------------------------------------------------


def test_resolve_device_honours_the_override():
    assert resolve_device('cpu').type == 'cpu'


def test_resolve_device_matches_availability():
    expected = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert resolve_device().type == expected
