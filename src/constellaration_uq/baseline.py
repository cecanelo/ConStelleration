"""The shared single-MLP baseline: one pool loader, one recipe, one metric.

Three scripts fit the same model on different splits (the day 1-2 grid, the hole
placement sweep, the distance-error curves). "Same model and recipe for every
split" is the premise that makes their numbers comparable at all, and three
copy-pasted definitions is exactly where that premise quietly stops being true.
So the recipe lives here and nowhere else.

Not the project's model. This is the cheap single network used to make decisions
before the ensemble exists.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from constellaration_uq.data import filter_valid, load_raw
from constellaration_uq.results import REPO_ROOT

DATA_DIR = REPO_ROOT / 'data_raw' / 'data'

# Architecture follows Appendix A.4, Proxima's ten-MLP ensemble baseline: three
# layers, 256 units, tanh. Frozen constants, deliberately not config fields, so
# that "more data" is never confounded with "more optimization".
HIDDEN_LAYER_SIZES = (256, 256, 256)
ACTIVATION = 'tanh'
MAX_ITER = 500
N_ITER_NO_CHANGE = 20
ARCHITECTURE = '(256, 256, 256) tanh, early stopping'


def load_pool(data_dir=None):
    """Run the filter chain to exhaustion and return the surviving rows.

    filter_valid yields after every step so callers can print the count table.
    Nothing here needs the intermediate steps, only the last frame.
    """
    for _, df in filter_valid(load_raw(data_dir or DATA_DIR)):
        pass
    return df


def fit_mlp(X_fit, y_fit, seed=0):
    """Fit once, return a predict function that answers in physical units.

    Inputs are scaled on the fit set only and the target is z-scored with fit
    statistics, then un-z-scored on the way out, so no caller ever handles a
    z-score. Out-of-region inputs will fall outside the fitted scaler range;
    that is intended and they are not clipped.
    """
    scaler = StandardScaler().fit(X_fit)
    y_mean, y_std = y_fit.mean(), y_fit.std()

    model = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        random_state=seed,
        max_iter=MAX_ITER,
        early_stopping=True,
        n_iter_no_change=N_ITER_NO_CHANGE,
    ).fit(scaler.transform(X_fit), (y_fit - y_mean) / y_std)

    def predict(X):
        return model.predict(scaler.transform(X)) * y_std + y_mean

    return predict


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
