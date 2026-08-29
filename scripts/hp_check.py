"""Step 0: the bounded hyperparameter sanity check (4.10).

Not hyperparameter optimization. Appendix A.4 gives architecture, loss, target
scaling and ensemble size and nothing else, and its training code is not in the
public repo, so optimizer, learning rate, batch size and stopping rule have to
be chosen here and declared as ours. A bad learning rate costs an order of
magnitude, which would leave the Table 7 comparison ambiguous between a
pipeline bug and a bad optimizer setting, and disambiguating that is the only
reason the plain MSE ensemble exists.

Six fits, three learning rates by two batch sizes, single networks rather than
ensembles.

⚠️ Random split only, selected on validation loss only. Choosing
hyperparameters by tail-split performance would leak the extrapolation
condition into the model and invalidate the study in a way no reviewer could
detect from the results. This script never constructs a tail split, which is
the cheapest way to make that impossible rather than merely discouraged.

A seventh fit repeats the winner at the N-sweep's smallest training size to
check whether 500 validation points still give a usable stopping signal there
(4.7). That is the tightest case the frozen recipe has to survive.
"""

import time

import numpy as np

from constellaration_uq.baseline import load_pool, rmse
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.nets import (
    MAX_EPOCHS,
    PATIENCE,
    VAL_SIZE,
    resolve_device,
    split_validation,
    train_one,
)
from constellaration_uq.results import save_results, save_table
from constellaration_uq.splits import random_split

SEED = 0
TEST_FRACTION = 0.2
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
AXIS_COL = 'metrics.aspect_ratio'

# Cheapest first, so a pathological configuration shows up in the first row
# rather than after half an hour. Large batches mean fewer optimizer steps per
# epoch, so batch 512 runs faster per epoch than batch 128.
LEARNING_RATES = [3e-3, 1e-3, 3e-4]
BATCH_SIZES = [512, 128]

# The N-sweep's smallest training size (6.1). The recipe has to survive it.
SMALLEST_N = 1000


def main():
    started = time.time()
    print(f'device: {resolve_device()}')

    df = load_pool()
    trimmed = trim_target_tails(df, TARGET_COL)
    X = extract_input_features(trimmed)
    y = trimmed[TARGET_COL].to_numpy()
    axis = trimmed[AXIS_COL].to_numpy()
    print(f'pool: {len(trimmed):,} rows after target trim')

    # Random split, and deliberately nothing else. The held-out portion is not
    # touched by this script at all; selection happens on validation loss.
    train_mask, _ = random_split(axis, SEED, TEST_FRACTION)
    X_train, y_train = X[train_mask], y[train_mask]

    # Carved once, before anything is subsampled, and reused by every fit so
    # the six configurations are compared on identical data (4.7).
    X_fit, y_fit, X_val, y_val = split_validation(X_train, y_train, VAL_SIZE, SEED)
    print(f'fit {len(X_fit):,}  validation {len(X_val):,}  cap {MAX_EPOCHS} epochs\n')

    header = (
        f'{"lr":>8s} {"batch":>6s} {"n_fit":>7s} {"epochs":>7s} '
        f'{"best_val":>9s} {"val_rmse":>9s} {"jitter":>7s} {"time":>7s}'
    )
    print(header)
    print('-' * len(header))

    rows = []
    for batch_size in BATCH_SIZES:
        for learning_rate in LEARNING_RATES:
            rows.append(
                run_one(X_fit, y_fit, X_val, y_val, learning_rate, batch_size, label='full')
            )

    best = min(rows, key=lambda r: r['best_val'])
    print(f'\nbest so far: lr {best["learning_rate"]:g}, batch {best["batch_size"]}')

    # Repeat the winner at the sweep's smallest N. The validation set is the
    # same 500 points, so this asks the one question that matters there: is the
    # stopping signal still smooth when the model has very little to learn from.
    print(f'\nrepeating the winner at N = {SMALLEST_N:,} (sweep floor)\n')
    print(header)
    print('-' * len(header))
    subsample = np.random.default_rng(SEED).choice(len(X_fit), SMALLEST_N, replace=False)
    rows.append(
        run_one(
            X_fit[subsample],
            y_fit[subsample],
            X_val,
            y_val,
            best['learning_rate'],
            best['batch_size'],
            label='smallest_n',
        )
    )

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')
    print('\nPick by best_val. Then read jitter, the mean absolute epoch-to-epoch')
    print('change in validation loss over the last 20 epochs, relative to best_val.')
    print('A small value means 500 validation points give a smooth stopping signal.')
    print('If the smallest_n row is much jitterier than the full rows, raise VAL_SIZE.')

    constants = {
        'SEED': SEED,
        'TEST_FRACTION': TEST_FRACTION,
        'TARGET_COL': TARGET_COL,
        'AXIS_COL': AXIS_COL,
        'VAL_SIZE': VAL_SIZE,
        'MAX_EPOCHS': MAX_EPOCHS,
        'PATIENCE': PATIENCE,
        'LEARNING_RATES': LEARNING_RATES,
        'BATCH_SIZES': BATCH_SIZES,
        'SMALLEST_N': SMALLEST_N,
        'architecture': '(256, 256, 256) tanh, Adam, MSE',
    }
    payload = {'total_seconds': round(total_seconds, 1), 'rows': rows}

    save_results('hp_check', payload, constants=constants)
    save_table('hp_check', [{k: v for k, v in r.items() if k != 'history'} for r in rows])
    print('saved results/hp_check.json and .csv')


def run_one(X_fit, y_fit, X_val, y_val, learning_rate, batch_size, label):
    """One fit, printed before and after so a long run shows what is running."""
    print(f'{learning_rate:8.1e} {batch_size:6d} {len(X_fit):7,d} ', end='', flush=True)

    t0 = time.time()
    predict, history = train_one(
        X_fit,
        y_fit,
        X_val,
        y_val,
        seed=SEED,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0

    best_val = float(min(history))
    val_rmse = rmse(y_val, predict(X_val))

    # Mean absolute epoch-to-epoch change over the tail of the curve, scaled by
    # the best loss. This is the number that says whether VAL_SIZE is big enough
    # for the stopping rule to be picking a real minimum rather than noise.
    tail = np.asarray(history[-20:])
    jitter = float(np.mean(np.abs(np.diff(tail))) / best_val) if len(tail) > 1 else float('nan')

    print(f'{len(history):7d} {best_val:9.5f} {val_rmse:9.5f} {jitter:7.3f} {elapsed:6.1f}s')

    return {
        'label': label,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_fit': len(X_fit),
        'epochs': len(history),
        'best_val': best_val,
        'val_rmse': val_rmse,
        'jitter': jitter,
        'fit_seconds': round(elapsed, 1),
        'history': history,
    }


if __name__ == '__main__':
    main()
