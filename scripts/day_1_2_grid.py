"""Day 1-2 grid: pick the split axis, direction, and primary target.

2 axes x 3 cuts x 2 targets = 12 cheap fits. Decides 3.4 (axis), 3.6
(direction), evidence for 3.1 and 3.2 (targets), 3.7 (the coverage tolerance
delta) and 7.1 (diagnostic set).

Pass condition: some (axis, direction) pair shows a real in/out error gap for
the primary target AND that axis clears the coverage threshold in both a
mid-range band and a tail band, since the main figure needs both results.

A single MLP, not the ensemble and deliberately not gradient boosting. Trees
return the boundary value outside the training range, so a tail split would
show a large gap by construction and measure the model class rather than the
axis. The eventual model is an MLP ensemble, so an MLP baseline is also what
the real extrapolation behaviour will look like.

The axis is never trimmed (1.4). Only the target is.
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split

from constellaration_uq.baseline import ARCHITECTURE, fit_mlp, load_pool, rmse
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.results import save_results, save_table
from constellaration_uq.splits import (
    distance_from_training_region,
    hole_split,
    tail_split,
)

SEED = 0
TEST_FRACTION = 0.2
IN_REGION_TEST_FRACTION = 0.2
N_DISTANCE_BINS = 8

AXES = ['metrics.aspect_ratio', 'metrics.max_elongation']
CUTS = ['tail_low', 'tail_high', 'hole']
TARGETS = {
    'edge_rot_transform': ('metrics.edge_rotational_transform_over_n_field_periods', None),
    'log10_qi': ('metrics.qi', np.log10),
}


def run_combination(axis, X, y, cut):
    if cut == 'hole':
        train_mask, oor_mask = hole_split(axis, TEST_FRACTION)
    else:
        train_mask, oor_mask = tail_split(axis, cut.removeprefix('tail_'), TEST_FRACTION)

    # An in-region held-out slice, carved from the training region only (3.9).
    # Without it the in-region number would be training error and the gap would
    # be measuring overfitting rather than extrapolation.
    train_idx = np.flatnonzero(train_mask)
    fit_idx, in_idx = train_test_split(
        train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=SEED
    )

    predict = fit_mlp(X[fit_idx], y[fit_idx], SEED)
    rmse_in = rmse(y[in_idx], predict(X[in_idx]))
    rmse_out = rmse(y[oor_mask], predict(X[oor_mask]))

    # Equal-width distance bins, since the calibration figure bins by distance
    # and every band needs enough points to estimate a coverage rate (3.7).
    distance = distance_from_training_region(axis, train_mask)
    d_oor = distance[oor_mask]
    counts, _ = np.histogram(d_oor, bins=np.linspace(0, d_oor.max(), N_DISTANCE_BINS + 1))
    min_count = int(counts.min())

    # n = z^2 p(1-p) / delta^2 with z=1.96, p=0.5, solved for delta.
    delta = 0.98 / np.sqrt(min_count) if min_count else float('inf')

    return {
        'n_fit': len(fit_idx),
        'n_oor': int(oor_mask.sum()),
        'rmse_in': rmse_in,
        'rmse_out': rmse_out,
        'ratio': rmse_out / rmse_in,
        'd_max': float(d_oor.max()),
        'min_bin': min_count,
        'delta': delta,
    }


def main():
    started = time.time()
    df = load_pool()
    print(f'pool: {len(df):,} rows')

    total = len(TARGETS) * len(AXES) * len(CUTS)
    print(f'{total} combinations, one MLP fit each\n')

    header = (
        f'{"":8s} {"axis":16s} {"cut":10s} {"target":18s} '
        f'{"rmse_in":>9s} {"rmse_out":>9s} {"ratio":>7s} '
        f'{"d_max":>7s} {"min_bin":>8s} {"delta":>7s} {"time":>7s}'
    )
    print(header)
    print('-' * len(header))

    rows = []
    step = 0
    for target_name, (target_col, transform) in TARGETS.items():
        trimmed = trim_target_tails(df, target_col)
        X = extract_input_features(trimmed)
        y = trimmed[target_col].to_numpy()
        if transform is not None:
            y = transform(y)

        for axis_col in AXES:
            axis = trimmed[axis_col].to_numpy()
            for cut in CUTS:
                step += 1
                # Printed before the fit and flushed, so a long fit shows which
                # combination is running rather than looking hung.
                print(
                    f'[{step:2d}/{total}] {axis_col.removeprefix("metrics."):16s} '
                    f'{cut:10s} {target_name:18s} ',
                    end='',
                    flush=True,
                )

                t0 = time.time()
                r = run_combination(axis, X, y, cut)
                elapsed = time.time() - t0
                print(
                    f'{r["rmse_in"]:9.5f} {r["rmse_out"]:9.5f} {r["ratio"]:7.2f} '
                    f'{r["d_max"]:7.2f} {r["min_bin"]:8,d} {r["delta"]:7.3f} '
                    f'{elapsed:6.1f}s'
                )

                # Identifiers first so the CSV column order reads naturally.
                rows.append(
                    {
                        'axis': axis_col.removeprefix('metrics.'),
                        'cut': cut,
                        'target': target_name,
                        **r,
                        'fit_seconds': round(elapsed, 1),
                    }
                )

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')

    constants = {
        'SEED': SEED,
        'TEST_FRACTION': TEST_FRACTION,
        'IN_REGION_TEST_FRACTION': IN_REGION_TEST_FRACTION,
        'N_DISTANCE_BINS': N_DISTANCE_BINS,
        'AXES': AXES,
        'CUTS': CUTS,
        'TARGETS': {k: v[0] for k, v in TARGETS.items()},
        'architecture': ARCHITECTURE,
    }
    payload = {'pool_rows': len(df), 'total_seconds': round(total_seconds, 1), 'rows': rows}

    save_results('day_1_2_grid', payload, constants=constants)
    save_table('day_1_2_grid', rows)
    print('saved results/day_1_2_grid.json and .csv')


if __name__ == '__main__':
    main()
