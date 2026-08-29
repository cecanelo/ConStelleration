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
