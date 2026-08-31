"""The network and its training loop.

sklearn's MLPRegressor took the project as far as the split decisions and
cannot go further: no two-output variance head, no custom NLL loss, no GPU.
This is the replacement, built so the variance head is an output-width
argument rather than a rewrite.

Batching deliberately avoids DataLoader. The whole pool is 27k rows of 80
columns, about 17 MB, so it fits on the device once and every epoch is an
index permutation. DataLoader would add worker processes and per-batch host to
device copies to move data that never needed to leave.
"""

import copy

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from constellaration_uq.baseline import HIDDEN_LAYER_SIZES

# FROZEN 2026-08-29 by the 4.10 sanity check, recorded in 4.7. Do not change
# these to make a run finish faster: the N-sweep's whole claim is that only the
# amount of data varied between its runs.
VAL_SIZE = 500
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 500
PATIENCE = 20

# Mean-variance only (4.4). Epochs of plain MSE before the NLL term is allowed
# to act: the NLL gradient on the mean is scaled by 1/variance, so a model that
# has not learned the mean yet labels its hard points as noisy and then stops
# learning from them. 25 comes from the step 0 history, where the frozen recipe
# reaches 1.5x its best loss by epoch 18 and 1.2x by epoch 32.
WARMUP_EPOCHS = 25

# Floor on the predicted variance, in z-scored space where the target has unit
# variance. 1e-6 is six orders of magnitude below the signal and safely above
# float32 noise, so it stops 1/variance exploding without manufacturing an
# aleatoric term that the step 5 check would then read as real.
VARIANCE_FLOOR = 1e-6


def resolve_device(override=None):
    """GPU when there is one, CPU otherwise.

    The override exists so tests can pin CPU and so a misbehaving GPU run can
    be forced back without editing the module.
    """
    if override is not None:
        return torch.device(override)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    """Three hidden layers of 256 with tanh, following Appendix A.4.

    n_outputs is the only difference between the plain MSE model and the
    mean-variance one: 1 for a point prediction, 2 for a mean and a raw
    log-variance.
    """

    def __init__(self, n_inputs, n_outputs=1, hidden=HIDDEN_LAYER_SIZES):
        super().__init__()
        layers, width = [], n_inputs
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        layers.append(nn.Linear(width, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def split_validation(X, y, n_val=VAL_SIZE, seed=0):
    """Carve a fixed-size early-stopping set. Returns (X_fit, y_fit, X_val, y_val).

    Fixed size, never a fraction (4.7, the recipe freeze). A 10% validation set
    would be 100 points at the sweep's smallest N and 1,800 at its largest, so
    models would stop training at different points for a reason unrelated to
    data volume, which is the confound that decision exists to prevent.

    Call this once, before any subsampling, and hand the same validation set to
    every fit in a sweep.
    """
    order = np.random.default_rng(seed).permutation(len(X))
    val, fit = order[:n_val], order[n_val:]
    return X[fit], y[fit], X[val], y[val]


def train_one(
    X_fit,
    y_fit,
    X_val,
    y_val,
    seed=0,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    device=None,
):
    """Train one MSE network. Returns (predict, history).

    predict answers in physical units, so no caller handles a z-score. history
    is the per-epoch validation loss, which is what the 4.10 sanity check reads
    to choose a learning rate and to confirm 500 validation points give a
    smooth enough stopping signal.

    seed is the only source of diversity between ensemble members, and it
    drives both the weight initialisation and the shuffle order (4.3).
    """
    device = resolve_device(device)

    # Scaler and target statistics come from the fit set alone (3.10). The
    # validation set is transformed with them and never used to fit them.
    scaler = StandardScaler().fit(X_fit)
    y_mean, y_std = y_fit.mean(), y_fit.std()

    def tensor(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    x_fit = tensor(scaler.transform(X_fit))
    t_fit = tensor((y_fit - y_mean) / y_std).unsqueeze(1)
    x_val = tensor(scaler.transform(X_val))
    t_val = tensor((y_val - y_mean) / y_std).unsqueeze(1)

    torch.manual_seed(seed)
    model = MLP(x_fit.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    shuffle = torch.Generator().manual_seed(seed)
    best_loss, best_state, waited = float('inf'), None, 0
    history = []

    for _ in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_fit), generator=shuffle).to(device)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(model(x_fit[batch]), t_fit[batch]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), t_val).item()
        history.append(val_loss)

        # Keep the best weights, not the last. Without this, early stopping
        # returns a model `patience` epochs worse than the one it stopped for.
        if val_loss < best_loss:
            best_loss, waited = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            out = model(tensor(scaler.transform(X)))
        return out.cpu().numpy().squeeze(-1) * y_std + y_mean

    return predict, history


# The clamp is applied to the raw output before exponentiating, which enforces
# the floor and stops exp overflowing in one step. log(1e-6) is about -13.8.
LOG_VARIANCE_MIN = float(np.log(VARIANCE_FLOOR))
LOG_VARIANCE_MAX = 13.8


def variance_from_raw(raw):
    """Turn the network's second output into a positive variance (4.5).

    The output is an unconstrained real number, declared to be the natural log
    of the variance, so exp of it is positive for any value the network can
    produce and no constraint is needed on the head itself.

    Log-variance rather than softplus because the loss needs log(variance),
    which is then the raw value itself with no exp-then-log round trip. Kept in
    one function so a softplus swap is a single edit rather than a search
    through the codebase.
    """
    return torch.exp(raw.clamp(LOG_VARIANCE_MIN, LOG_VARIANCE_MAX))


def gaussian_nll(raw_mean, raw_log_variance, target):
    """Negative log likelihood of a Gaussian, up to an additive constant.

    Two terms pulling against each other. The squared error divided by the
    variance rewards claiming high uncertainty, since it shrinks the penalty.
    The log-variance term punishes claiming it. Their balance is what makes the
    model report calibrated uncertainty rather than saying "I do not know"
    everywhere.

    The clamped raw value is used directly as log(variance) rather than taking
    a log of the exponentiated one, which is both cheaper and exact.

    ⚠️ **The clamp has zero derivative outside its bounds, in BOTH directions,
    and that is a latent trap rather than only a safety rail.** The floor
    correctly blocks a point being pulled further down, but it equally blocks it
    being pulled back up: a point whose raw output dives below log(1e-6)
    contributes no gradient to the variance head at all, so it cannot un-pin
    itself through its own NLL term, only through shared-weight updates driven
    by other points. A head that collapses early in the NLL phase would stay
    collapsed and aleatoric would silently read 1e-6 * y_std^2 everywhere, which
    looks like a confident model rather than a broken one.

    **Nothing catches this except the monitor.** `mv_ensemble.py` computes and
    prints the fraction of member predictions sitting on the floor, and 4.4 made
    that the observable beta-NLL trigger. It has read 0.00% in every run so far,
    so the hazard has never fired, but the monitor is the only thing standing
    between it and a wrong number that raises nothing. Do not remove it.
    """
    clamped = raw_log_variance.clamp(LOG_VARIANCE_MIN, LOG_VARIANCE_MAX)
    variance = torch.exp(clamped)
    return torch.mean(0.5 * (clamped + (target - raw_mean) ** 2 / variance))


def train_one_mv(
    X_fit,
    y_fit,
    X_val,
    y_val,
    seed=0,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
    warmup_epochs=WARMUP_EPOCHS,
    device=None,
):
    """Train one mean-variance network. Returns (predict, history).

    predict(X) returns (mean, variance), both in physical units.

    ⚠️ The variance un-scales by y_std squared while the mean un-scales by
    y_std. Getting that wrong produces entirely plausible numbers that are off
    by the square of the target's spread, and nothing raises. It is the most
    likely silent error in this codebase, which is why this is a separate
    function rather than a flag on train_one.

    The loss switches from MSE to Gaussian NLL at warmup_epochs (4.4). The NLL
    gradient on the mean is scaled by 1/variance, so a model that has not
    learned the mean yet marks its hard points as noisy and then stops learning
    from them. Warming up on MSE gets the mean roughly right first.

    history entries are (phase, validation loss). The two phases are on
    different scales and are not comparable, which is why the phase is recorded
    and why best-weight tracking resets at the switch.
    """
    # ⚠️ Without this the failure is silent and total. Best-weight tracking
    # resets at the warm-up boundary, so a loop that never reaches it restores
    # the best MSE-phase checkpoint instead, and `predict` then returns
    # variances from a head that was never trained on the NLL at all. Every
    # number downstream would be plausible and meaningless. Unreachable with
    # the frozen constants (500 against 25), but both are reachable through the
    # **train_kwargs chain in ensemble.py, so the guard is cheap insurance.
    if max_epochs <= warmup_epochs:
        raise ValueError(
            f'max_epochs ({max_epochs}) must exceed warmup_epochs ({warmup_epochs}), '
            'or the variance head is never trained and its output is meaningless'
        )

    device = resolve_device(device)

    scaler = StandardScaler().fit(X_fit)
    y_mean, y_std = y_fit.mean(), y_fit.std()

    def tensor(a):
        return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

    x_fit = tensor(scaler.transform(X_fit))
    t_fit = tensor((y_fit - y_mean) / y_std).unsqueeze(1)
    x_val = tensor(scaler.transform(X_val))
    t_val = tensor((y_val - y_mean) / y_std).unsqueeze(1)

    torch.manual_seed(seed)
    model = MLP(x_fit.shape[1], n_outputs=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    def loss_for(epoch, output, target):
        """MSE on the mean during warm-up, NLL after. The variance head is
        untrained during warm-up and its output is simply ignored."""
        if epoch < warmup_epochs:
            return mse(output[:, :1], target)
        return gaussian_nll(output[:, :1], output[:, 1:], target)

    shuffle = torch.Generator().manual_seed(seed)
    best_loss, best_state, waited = float('inf'), None, 0
    history = []

    for epoch in range(max_epochs):
        # The two losses are on different scales, so a best-so-far carried
        # across the switch would compare two different quantities and would
        # usually freeze the warm-up weights forever. Reset at the boundary.
        if epoch == warmup_epochs:
            best_loss, best_state, waited = float('inf'), None, 0

        model.train()
        order = torch.randperm(len(x_fit), generator=shuffle).to(device)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_for(epoch, model(x_fit[batch]), t_fit[batch]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_for(epoch, model(x_val), t_val).item()
        phase = 'warmup' if epoch < warmup_epochs else 'nll'
        history.append((phase, val_loss))

        if val_loss < best_loss:
            best_loss, waited = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            # Never stop during warm-up. Its job is to run for a fixed number
            # of epochs so the NLL phase starts from a comparable place at
            # every N (4.7), and stopping early would make that vary.
            if waited >= patience and epoch >= warmup_epochs:
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            out = model(tensor(scaler.transform(X)))
            mean = out[:, 0].cpu().numpy() * y_std + y_mean
            # Squared, because a variance scales with the square of a linear
            # rescaling of the target. This is the trap the docstring names.
            variance = variance_from_raw(out[:, 1]).cpu().numpy() * y_std**2
        return mean, variance

    return predict, history
