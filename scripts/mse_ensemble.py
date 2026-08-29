"""Step 1: the plain MSE ensemble, checked against Table 7 (4.6).

This is what Appendix A.4 actually built, ten MLPs minimising mean squared
error, so it is the honest like-for-like number against their Table 7.
Comparing a mean-variance ensemble's RMSE to theirs would compare two different
training objectives.

Its real job is isolating failure. If it lands near Table 7, the loader, the
five-flag error filter, the NFP and pathway filters, the target trim, the
scaling, the architecture and the new PyTorch training loop are all verified in
one run. Any later degradation is then attributable to the variance head or the
NLL loss rather than to the pipeline, instead of having two possible causes at
once.

Pass condition, pre-registered in 4.6 before this was ever run: ensemble RMSE
below 0.0105 on the held-out set, and better than any individual member. Table 7
is 0.006, the sklearn single MLP was 0.0138, and one network on the frozen
recipe gives 0.01149.

Random split only. The three-split experiment is steps 2 to 4 and gets its own
script; mixing them here would make a pipeline check depend on the very
generalization question it is supposed to be neutral about.
"""

import time

import numpy as np

from constellaration_uq.baseline import load_pool, rmse
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.ensemble import N_MEMBERS, combine, train_ensemble
from constellaration_uq.nets import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    VAL_SIZE,
    resolve_device,
    split_validation,
)
from constellaration_uq.results import save_results, save_table
from constellaration_uq.splits import random_split

SEED = 0
TEST_FRACTION = 0.2
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
AXIS_COL = 'metrics.aspect_ratio'

# Pre-registered in 4.6, before the run. Judging afterwards always produces a
# pass, which is the lesson gate 3.5 paid for.
RMSE_BAR = 0.0105

# Table 7's row for this metric, for the comparison this run exists to make.
TABLE_7_RMSE = 0.006
TABLE_7_R2 = 0.997


def r_squared(y_true, y_pred):
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - residual / total)


def main():
    started = time.time()
    print(f'device: {resolve_device()}')

    df = load_pool()
    trimmed = trim_target_tails(df, TARGET_COL)
    X = extract_input_features(trimmed)
    y = trimmed[TARGET_COL].to_numpy()
    axis = trimmed[AXIS_COL].to_numpy()
    print(f'pool: {len(trimmed):,} rows after target trim')

    train_mask, test_mask = random_split(axis, SEED, TEST_FRACTION)
    X_fit, y_fit, X_val, y_val = split_validation(X[train_mask], y[train_mask], VAL_SIZE, SEED)
    X_test, y_test = X[test_mask], y[test_mask]

    print(
        f'fit {len(X_fit):,}  validation {len(X_val):,}  held out {len(X_test):,}\n'
        f'recipe: lr {LEARNING_RATE:g}, batch {BATCH_SIZE}, '
        f'cap {MAX_EPOCHS} epochs, patience {PATIENCE}\n'
    )

    header = f'{"member":>7s} {"epochs":>7s} {"best_val":>9s} {"time":>7s}'
    print(header)
    print('-' * len(header))

    # Wall clock at the end of the previous member, so each row reports its own
    # duration rather than the elapsed total.
    previous = [time.time()]

    def report(k, epochs, best_val):
        now = time.time()
        print(f'{k:7d} {epochs:7d} {best_val:9.5f} {now - previous[0]:6.1f}s', flush=True)
        previous[0] = now

    predict_all, histories = train_ensemble(
        X_fit, y_fit, X_val, y_val, base_seed=SEED, progress=report
    )

    member_predictions = predict_all(X_test)
    mean, epistemic = combine(member_predictions)

    member_rmse = [rmse(y_test, p) for p in member_predictions]
    ensemble_rmse = rmse(y_test, mean)
    ensemble_r2 = r_squared(y_test, mean)

    print('\nper-member held-out RMSE:')
    print('  ' + '  '.join(f'{v:.5f}' for v in member_rmse))

    # Two independent conditions, both pre-registered. The second guards against
    # an ensemble that only looks good because one lucky member carries it.
    beats_bar = ensemble_rmse < RMSE_BAR
    beats_members = ensemble_rmse < min(member_rmse)
    verdict = 'PASS' if beats_bar and beats_members else 'FAIL'

    print(
        f'\nensemble RMSE  {ensemble_rmse:.5f}   bar {RMSE_BAR:.5f}   '
        f'{"ok" if beats_bar else "MISS"}'
    )
    print(f'best member    {min(member_rmse):.5f}   {"ok" if beats_members else "MISS"}')
    print(f'ensemble R2    {ensemble_r2:.5f}')
    print(f'\nTable 7:  RMSE {TABLE_7_RMSE:.5f}  R2 {TABLE_7_R2:.5f}')
    print(f'ratio to Table 7: {ensemble_rmse / TABLE_7_RMSE:.2f}x on RMSE')

    # Not a result, a sanity reading. On a random split every held-out point sits
    # next to training data, so member disagreement should be small here. It is
    # the number that has to grow in steps 3 and 4 for the project to have a
    # subject at all.
    print(
        f'\nmean epistemic spread in region: {epistemic.mean():.5f}'
        f'  ({epistemic.mean() / y.std() * 100:.1f}% of the target std)'
    )

    total_seconds = time.time() - started
    print(f'\nverdict: {verdict}      total {total_seconds:.1f}s')

    constants = {
        'SEED': SEED,
        'TEST_FRACTION': TEST_FRACTION,
        'TARGET_COL': TARGET_COL,
        'AXIS_COL': AXIS_COL,
        'N_MEMBERS': N_MEMBERS,
        'VAL_SIZE': VAL_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'BATCH_SIZE': BATCH_SIZE,
        'MAX_EPOCHS': MAX_EPOCHS,
        'PATIENCE': PATIENCE,
        'RMSE_BAR': RMSE_BAR,
        'architecture': '(256, 256, 256) tanh, Adam, MSE, no variance head',
    }
    payload = {
        'verdict': verdict,
        'ensemble_rmse': ensemble_rmse,
        'ensemble_r2': ensemble_r2,
        'member_rmse': member_rmse,
        'member_epochs': [len(h) for h in histories],
        'best_val': [float(min(h)) for h in histories],
        'mean_epistemic_in_region': float(epistemic.mean()),
        'target_std': float(y.std()),
        'table_7_rmse': TABLE_7_RMSE,
        'table_7_r2': TABLE_7_R2,
        'total_seconds': round(total_seconds, 1),
    }

    save_results('mse_ensemble', payload, constants=constants)
    save_table(
        'mse_ensemble_members',
        [
            {
                'member': k,
                'epochs': len(h),
                'best_val': float(min(h)),
                'test_rmse': member_rmse[k],
            }
            for k, h in enumerate(histories)
        ],
    )
    print('saved results/mse_ensemble.json and results/mse_ensemble_members.csv')


if __name__ == '__main__':
    main()
