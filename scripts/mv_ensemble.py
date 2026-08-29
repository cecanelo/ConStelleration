"""Steps 2, 4 and 5: a mean-variance ensemble on one split.

One script, split name as an argument, because "same model and recipe for every
split" is the premise that makes the three results comparable at all (3.8).
Three scripts would be three places for that premise to quietly stop holding.

    python3 scripts/mv_ensemble.py random
    python3 scripts/mv_ensemble.py hole
    python3 scripts/mv_ensemble.py tail

Four disjoint sets come out of the pool, and keeping them straight is most of
what this script does:

    fit          trains the ten members
    validation   500 fixed points, early stopping only, never reported
    in-region    held out from the training region, the "normal accuracy" number
    out-region   the split's held-out set, near-identical to in-region for the
                 random split and the actual subject for hole and tail

Per-point predictions are written to CSV so the calibration figures and the
deferral curve can be built without refitting.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from constellaration_uq.baseline import load_pool, rmse
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.ensemble import N_MEMBERS, decompose, train_mv_ensemble
from constellaration_uq.nets import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    VAL_SIZE,
    VARIANCE_FLOOR,
    WARMUP_EPOCHS,
    resolve_device,
    split_validation,
)
from constellaration_uq.results import save_results, save_table
from constellaration_uq.splits import (
    distance_from_training_region,
    hole_split,
    random_split,
    tail_split,
)

SEED = 0
TEST_FRACTION = 0.2
IN_REGION_TEST_FRACTION = 0.2
HOLE_CENTRE = 0.30
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
AXIS_COL = 'metrics.aspect_ratio'

# The plain MSE ensemble's held-out RMSE on the random split (4.6). The
# mean-variance model should land near it; a large gap means the variance head
# or the NLL loss cost accuracy, which is what step 1 exists to let us say.
MSE_ENSEMBLE_RMSE = 0.01052

# A member variance within this factor of the floor counts as pinned. The
# fraction of pinned predictions is the observable beta-NLL trigger from 4.4.
FLOOR_TOLERANCE = 1.01


@dataclass(frozen=True)
class Config:
    """One typed object threaded through the entry point (4.8). Architecture and
    training hyperparameters are deliberately not fields: they are frozen
    constants in nets.py, and making them configurable would reopen 4.7."""

    split: str
    seed: int = SEED
    target_col: str = TARGET_COL
    axis_col: str = AXIS_COL


def build_split(config, axis):
    if config.split == 'random':
        return random_split(axis, config.seed, TEST_FRACTION)
    if config.split == 'hole':
        return hole_split(axis, TEST_FRACTION, HOLE_CENTRE)
    if config.split == 'tail':
        return tail_split(axis, 'low', TEST_FRACTION)
    raise ValueError(f"split must be 'random', 'hole' or 'tail', got {config.split!r}")


def summarise(label, y_true, parts, selection):
    """Point accuracy and the uncertainty decomposition over one set of rows.

    Reported as standard deviations rather than variances, because that is the
    scale the target lives on and the only one a reader can compare against
    RMSE. The addition happened in decompose, in variance space, where it is
    valid.
    """
    mean = parts['mean'][selection]
    epistemic = np.sqrt(parts['epistemic_variance'][selection])
    aleatoric = np.sqrt(parts['aleatoric_variance'][selection])
    total = np.sqrt(parts['total_variance'][selection])

    return {
        'set': label,
        'n': int(selection.sum()) if selection.dtype == bool else len(selection),
        'rmse': rmse(y_true, mean),
        'epistemic': float(epistemic.mean()),
        'aleatoric': float(aleatoric.mean()),
        'total': float(total.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('split', choices=['random', 'hole', 'tail'])
    config = Config(split=parser.parse_args().split)

    started = time.time()
    print(f'device: {resolve_device()}   split: {config.split}')

    df = load_pool()
    trimmed = trim_target_tails(df, config.target_col)
    X = extract_input_features(trimmed)
    y = trimmed[config.target_col].to_numpy()
    axis = trimmed[config.axis_col].to_numpy()
    print(f'pool: {len(trimmed):,} rows after target trim')

    train_mask, out_mask = build_split(config, axis)

    # The in-region slice comes out first, then the validation set out of what
    # remains, so the three never overlap. Early stopping must not see the rows
    # the in-region number is computed on, and neither may see out-of-region
    # data (3.9), or the extrapolation condition leaks into training.
    train_idx = np.flatnonzero(train_mask)
    fit_pool, in_idx = train_test_split(
        train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=config.seed
    )
    X_fit, y_fit, X_val, y_val = split_validation(X[fit_pool], y[fit_pool], VAL_SIZE, config.seed)
    out_idx = np.flatnonzero(out_mask)

    print(
        f'fit {len(X_fit):,}  validation {len(X_val):,}  '
        f'in-region {len(in_idx):,}  out-of-region {len(out_idx):,}\n'
        f'recipe: lr {LEARNING_RATE:g}, batch {BATCH_SIZE}, warmup {WARMUP_EPOCHS}, '
        f'cap {MAX_EPOCHS}, patience {PATIENCE}\n'
    )

    header = f'{"member":>7s} {"epochs":>7s} {"best_nll":>9s} {"time":>7s}'
    print(header)
    print('-' * len(header))
    previous = [time.time()]

    def report(k, epochs, best):
        now = time.time()
        print(f'{k:7d} {epochs:7d} {best:9.4f} {now - previous[0]:6.1f}s', flush=True)
        previous[0] = now

    predict_all, histories = train_mv_ensemble(
        X_fit, y_fit, X_val, y_val, base_seed=config.seed, progress=report
    )

    eval_idx = np.concatenate([in_idx, out_idx])
    member_means, member_variances = predict_all(X[eval_idx])
    parts = decompose(member_means, member_variances)

    n_in = len(in_idx)
    rows = [
        summarise('in-region', y[in_idx], parts, np.arange(n_in)),
        summarise('out-of-region', y[out_idx], parts, np.arange(n_in, len(eval_idx))),
    ]

    print(
        f'\n{"set":>14s} {"n":>7s} {"rmse":>9s} {"epistemic":>10s} {"aleatoric":>10s} {"total":>9s}'
    )
    for row in rows:
        print(
            f'{row["set"]:>14s} {row["n"]:7,d} {row["rmse"]:9.5f} '
            f'{row["epistemic"]:10.5f} {row["aleatoric"]:10.5f} {row["total"]:9.5f}'
        )

    # The beta-NLL trigger from 4.4, made observable. A head pinned on its floor
    # is not measuring anything, it is saturated, and that is the symptom that
    # says warm-up plus floor was not enough.
    floor_physical = VARIANCE_FLOOR * y_fit.std() ** 2
    pinned = float((member_variances <= floor_physical * FLOOR_TOLERANCE).mean())
    print(f'\nvariance floor in physical units: {floor_physical:.3e}')
    print(f'member predictions pinned on it:  {pinned * 100:.2f}%')
    if pinned > 0.5:
        print('  ⚠️  over half pinned. This is the 4.4 trigger for beta-NLL.')

    if config.split == 'random':
        print(
            f'\nplain MSE ensemble was {MSE_ENSEMBLE_RMSE:.5f} on this split; '
            f'variance head costs {rows[0]["rmse"] / MSE_ENSEMBLE_RMSE - 1:+.1%} on in-region RMSE'
        )

    # Distance measured against the fit set, not train_mask. The mask holds the
    # in-region slice, which would then read distance 0 by construction rather
    # than by measurement.
    fit_mask = np.zeros(len(axis), dtype=bool)
    fit_mask[fit_pool] = True
    distance = distance_from_training_region(axis, fit_mask)[eval_idx]

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')

    constants = {
        'SEED': config.seed,
        'SPLIT': config.split,
        'TEST_FRACTION': TEST_FRACTION,
        'IN_REGION_TEST_FRACTION': IN_REGION_TEST_FRACTION,
        'HOLE_CENTRE': HOLE_CENTRE,
        'TARGET_COL': config.target_col,
        'AXIS_COL': config.axis_col,
        'N_MEMBERS': N_MEMBERS,
        'VAL_SIZE': VAL_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'BATCH_SIZE': BATCH_SIZE,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'MAX_EPOCHS': MAX_EPOCHS,
        'PATIENCE': PATIENCE,
        'VARIANCE_FLOOR': VARIANCE_FLOOR,
        'architecture': '(256, 256, 256) tanh, Adam, MSE warmup then Gaussian NLL',
    }
    payload = {
        'sets': rows,
        'pinned_fraction': pinned,
        'floor_physical': floor_physical,
        'member_epochs': [len(h) for h in histories],
        'target_std': float(y.std()),
        'total_seconds': round(total_seconds, 1),
    }
    save_results(f'mv_ensemble_{config.split}', payload, constants=constants)

    # Per-point rows so the calibration figures and the deferral curve need no
    # refit. Everything a Gaussian predictive distribution needs is here.
    save_table(
        f'mv_ensemble_{config.split}_points',
        [
            {
                'region': 'in' if i < n_in else 'out',
                'distance': round(float(distance[i]), 6),
                'axis': round(float(axis[eval_idx[i]]), 6),
                'y_true': round(float(y[eval_idx[i]]), 8),
                'mean': round(float(parts['mean'][i]), 8),
                'epistemic_variance': float(parts['epistemic_variance'][i]),
                'aleatoric_variance': float(parts['aleatoric_variance'][i]),
                'total_variance': float(parts['total_variance'][i]),
            }
            for i in range(len(eval_idx))
        ],
    )
    print(f'saved results/mv_ensemble_{config.split}.json and _points.csv')


if __name__ == '__main__':
    main()
