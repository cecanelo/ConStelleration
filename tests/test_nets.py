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
    LOG_VARIANCE_MAX,
    LOG_VARIANCE_MIN,
    MAX_EPOCHS,
    MLP,
    VAL_SIZE,
    VARIANCE_FLOOR,
    WARMUP_EPOCHS,
    gaussian_nll,
    resolve_device,
    split_validation,
    train_one,
    train_one_mv,
    variance_from_raw,
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


# --- raw output to variance --------------------------------------------------


def test_variance_from_raw_is_the_exponential():
    raw = torch.tensor([-2.0, 0.0, 1.5])

    assert torch.allclose(variance_from_raw(raw), torch.exp(raw))


def test_variance_from_raw_is_always_positive():
    """The head is unconstrained, so anything it emits has to come back as a
    usable variance. A negative or zero one makes the NLL undefined."""
    raw = torch.tensor([-500.0, -13.9, 0.0, 100.0, 1e4])

    assert (variance_from_raw(raw) > 0).all()
    assert torch.isfinite(variance_from_raw(raw)).all()


def test_variance_from_raw_respects_the_floor():
    """The floor stops 1/variance exploding. It must also be low enough that it
    cannot manufacture an aleatoric term the step 3 check would read as real,
    which is why the constant is asserted rather than just the clamping (4.4)."""
    assert variance_from_raw(torch.tensor([-1e6])).item() == pytest.approx(VARIANCE_FLOOR)
    assert VARIANCE_FLOOR == 1e-6
    assert LOG_VARIANCE_MIN == pytest.approx(np.log(1e-6))


def test_variance_from_raw_cannot_overflow():
    """Without the upper clamp, exp of a large raw output is inf, which
    poisons every downstream number silently."""
    assert variance_from_raw(torch.tensor([1e4])).item() == pytest.approx(np.exp(LOG_VARIANCE_MAX))


# --- the loss ----------------------------------------------------------------


def test_nll_prefers_small_variance_when_the_mean_is_right():
    """A confident correct prediction must score better than a hedged one, or
    the model has no reason to ever report low uncertainty."""
    target = torch.zeros(1, 1)
    mean = torch.zeros(1, 1)

    confident = gaussian_nll(mean, torch.full((1, 1), -4.0), target)
    hedged = gaussian_nll(mean, torch.zeros(1, 1), target)

    assert confident < hedged


def test_nll_prefers_large_variance_when_the_mean_is_wrong():
    """The mirror. A badly wrong prediction should be penalised less if the
    model admitted it was uncertain, which is what makes the head learn."""
    target = torch.zeros(1, 1)
    mean = torch.full((1, 1), 5.0)

    confident = gaussian_nll(mean, torch.zeros(1, 1), target)
    hedged = gaussian_nll(mean, torch.full((1, 1), 3.0), target)

    assert hedged < confident


def test_nll_matches_the_closed_form():
    """Guards the algebra itself, up to the dropped 0.5*log(2*pi) constant."""
    mean = torch.tensor([[1.0]])
    raw = torch.tensor([[0.7]])
    target = torch.tensor([[2.5]])

    variance = float(np.exp(0.7))
    expected = 0.5 * (0.7 + (2.5 - 1.0) ** 2 / variance)

    assert gaussian_nll(mean, raw, target).item() == pytest.approx(expected, rel=1e-5)


def test_nll_is_finite_at_the_floor():
    """The pathological case the floor exists for: a confident model that is
    badly wrong. Without a floor this is a division by something near zero."""
    loss = gaussian_nll(torch.zeros(1, 1), torch.full((1, 1), -1e6), torch.full((1, 1), 3.0))

    assert torch.isfinite(loss)


# --- mean-variance training --------------------------------------------------


def quick_mv(X_fit, y_fit, X_val, y_val, **kwargs):
    options = {'max_epochs': 20, 'patience': 20, 'warmup_epochs': 8, 'device': 'cpu'}
    return train_one_mv(X_fit, y_fit, X_val, y_val, **{**options, **kwargs})


def test_mv_predict_returns_a_mean_and_a_variance(fit_and_val):
    X_fit, y_fit, X_val, y_val = fit_and_val
    predict, _ = quick_mv(X_fit, y_fit, X_val, y_val)

    mean, variance = predict(X_val)

    assert mean.shape == variance.shape == (len(X_val),)
    assert (variance > 0).all()


def test_mv_mean_is_in_physical_units(fit_and_val):
    """The fixture's target sits near 10, so a forgotten un-z-scoring would
    answer near 0 with a perfectly correct shape."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    predict, _ = quick_mv(X_fit, y_fit, X_val, y_val)

    mean, _ = predict(X_val)

    assert abs(mean.mean() - y_val.mean()) < 0.5 * y_val.std()


def test_mv_variance_scales_with_y_std_squared(fit_and_val):
    """The likeliest silent error in the codebase. A variance rescales by the
    square of a linear change to the target, so multiplying the target by 10
    must multiply the reported variance by 100, not 10.

    Both models see identical data up to that scaling and identical seeds, so
    the ratio isolates the un-scaling rather than any training difference.
    """
    X_fit, y_fit, X_val, y_val = fit_and_val

    plain, _ = quick_mv(X_fit, y_fit, X_val, y_val, seed=0)
    scaled, _ = quick_mv(X_fit, y_fit * 10.0, X_val, y_val * 10.0, seed=0)

    _, variance = plain(X_val)
    _, variance_scaled = scaled(X_val)

    assert np.allclose(variance_scaled, variance * 100.0, rtol=1e-3)


def test_mv_history_records_both_phases(fit_and_val):
    """The two losses are on different scales, so the phase has to travel with
    the number or a later reader will plot them as one curve."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    _, history = quick_mv(X_fit, y_fit, X_val, y_val, max_epochs=12, warmup_epochs=5)

    phases = [phase for phase, _ in history]

    assert phases == ['warmup'] * 5 + ['nll'] * 7
    assert all(np.isfinite(value) for _, value in history)


def test_mv_warmup_runs_its_full_length(fit_and_val):
    """Warm-up must not be cut short by patience. Its job is to run a fixed
    number of epochs so the NLL phase starts from a comparable place at every
    N in the sweep (4.7), and early stopping would make that vary."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    _, history = quick_mv(X_fit, y_fit, X_val, y_val, max_epochs=30, warmup_epochs=12, patience=1)

    phases = [phase for phase, _ in history]

    assert phases.count('warmup') == 12
    assert 'nll' in phases, 'stopped before the NLL phase ever started'


def test_mv_best_weights_come_from_the_nll_phase(fit_and_val):
    """Warm-up losses are MSE and typically much smaller than NLL values. A
    best-so-far carried across the switch would lock in the warm-up weights and
    the variance head would never train at all."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    predict, history = quick_mv(X_fit, y_fit, X_val, y_val, max_epochs=25, warmup_epochs=8)

    nll_losses = [value for phase, value in history if phase == 'nll']
    _, variance = predict(X_val)

    assert len(nll_losses) > 1
    assert not np.allclose(variance, variance[0]), 'variance is constant, head did not train'


def test_mv_is_reproducible(fit_and_val):
    X_fit, y_fit, X_val, y_val = fit_and_val

    first, _ = quick_mv(X_fit, y_fit, X_val, y_val, seed=0)
    again, _ = quick_mv(X_fit, y_fit, X_val, y_val, seed=0)
    other, _ = quick_mv(X_fit, y_fit, X_val, y_val, seed=1)

    assert np.allclose(first(X_val)[0], again(X_val)[0])
    assert not np.allclose(first(X_val)[0], other(X_val)[0])


def test_warmup_default_is_the_frozen_value():
    """Changing this silently would alter the frozen recipe for every model in
    the project (4.4)."""
    assert WARMUP_EPOCHS == 25


def test_mv_refuses_a_max_epochs_that_skips_the_nll_phase(fit_and_val):
    """⚠️ The silent-failure guard. Best-weight tracking resets at the warm-up
    boundary, so a run that never reaches it restores the best MSE-phase
    checkpoint and returns variances from a head that never saw the NLL. Every
    downstream number would look ordinary and mean nothing. Unreachable with the
    frozen constants, reachable through ensemble.py's **train_kwargs."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    for max_epochs in (5, 10):
        with pytest.raises(ValueError, match='must exceed warmup_epochs'):
            train_one_mv(X_fit, y_fit, X_val, y_val, max_epochs=max_epochs, warmup_epochs=10)


def test_mv_accepts_one_epoch_past_warmup(fit_and_val):
    """Bounds the guard above: the check is <=, not <, so the smallest workable
    setting must still run rather than being rejected by an off-by-one."""
    X_fit, y_fit, X_val, y_val = fit_and_val
    predict, history = train_one_mv(
        X_fit, y_fit, X_val, y_val, max_epochs=3, warmup_epochs=2, patience=99
    )
    assert len(history) == 3
    _, variance = predict(X_val)
    assert (variance > 0).all()
