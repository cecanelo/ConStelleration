"""Deep ensembles over the networks in nets.py.

Ten members differing only in initialisation and shuffle order, no bootstrap
(4.3). One seed per member drives both, so member k is reproducible on its own.

The one design rule here: predict returns every member's answer, never their
average. Averaging inside would throw away the spread, and the spread is the
epistemic term the whole project is built on. Callers combine.
"""

import numpy as np

from constellaration_uq.nets import train_one

# Matching Appendix A.4, Proxima's ten-MLP ensemble baseline (4.2).
N_MEMBERS = 10


def train_ensemble(
    X_fit,
    y_fit,
    X_val,
    y_val,
    n_members=N_MEMBERS,
    base_seed=0,
    progress=None,
    **train_kwargs,
):
    """Train n_members networks. Returns (predict, histories).

    predict(X) returns an array of shape (n_members, len(X)) in physical units.
    histories is the per-member list of per-epoch validation losses, kept
    because a member that stopped after two epochs is a broken run that a
    summary statistic would hide.

    Every member sees the same fit and validation sets. Diversity comes from
    the seed alone (4.3), so resampling data per member would add a third
    mechanism that was never authorised.

    progress, if given, is called with (member_index, epochs, best_val) after
    each member, so a long run can report rather than look hung.
    """
    members, histories = [], []

    for k in range(n_members):
        predict, history = train_one(X_fit, y_fit, X_val, y_val, seed=base_seed + k, **train_kwargs)
        members.append(predict)
        histories.append(history)
        if progress is not None:
            progress(k, len(history), min(history))

    def predict_all(X):
        return np.stack([member(X) for member in members])

    return predict_all, histories


def combine(member_predictions):
    """Ensemble mean and epistemic standard deviation from member means.

    Epistemic is the spread of the member means (5.1). For the plain MSE
    ensemble that is the only uncertainty available, since there is no variance
    head to supply an aleatoric term.

    ddof=1 because the members are a sample of the model class, not the whole
    population of it. With ten members the difference from ddof=0 is about 5%
    on the standard deviation, which is small but is a bias in the direction of
    looking more confident than the evidence supports.
    """
    return member_predictions.mean(axis=0), member_predictions.std(axis=0, ddof=1)
