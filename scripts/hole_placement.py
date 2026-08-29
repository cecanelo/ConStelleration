"""Where should the interior hole go?

Reopens one piece of 3.7 (the held-out fraction and coverage tolerance entry),
which fixed the hole at 20% of the data centred on the median. The day 1-2 grid
then showed that hole reaches only 0.20 standard deviations from the training
region while the tail reaches 2.13, so a "at matched distance" comparison has
almost no range to work with.

The sweep in notebooks/figures.ipynb already showed moving the hole buys reach
at no cost in held-out size. What it cannot show is whether the *result* holds:
if a hole in a sparser region still costs nothing, the finding is robust to
placement and moving is free. If the ratio climbs, the "model fills gaps for
free" result depends on sitting in the dense middle, which is worth knowing.

Primary target and aspect ratio only, since this is one question about hole
placement rather than a rerun of the grid. Same model and recipe as
scripts/day_1_2_grid.py so the numbers are directly comparable to it.

Centres to read carefully:
  0.50  the current choice, percentiles 40 to 60
  0.30  percentiles 20 to 40, immediately above the tail without touching it
  0.20  percentiles 10 to 30, which overlaps the tail's percentiles 0 to 20,
        so the same configurations would be held out in both experiments
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
TEST_FRACTION = 0.1
IN_REGION_TEST_FRACTION = 0.2
N_DISTANCE_BINS = 8

AXIS_COL = 'metrics.aspect_ratio'
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
CENTRES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def score_split(axis, X, y, train_mask, oor_mask):
    train_idx = np.flatnonzero(train_mask)
    fit_idx, in_idx = train_test_split(
        train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=SEED
    )

    predict = fit_mlp(X[fit_idx], y[fit_idx], SEED)
    rmse_in = rmse(y[in_idx], predict(X[in_idx]))
    rmse_out = rmse(y[oor_mask], predict(X[oor_mask]))

    d_oor = distance_from_training_region(axis, train_mask)[oor_mask]
    counts, _ = np.histogram(d_oor, bins=np.linspace(0, d_oor.max(), N_DISTANCE_BINS + 1))
    min_count = int(counts.min())

    # n = z^2 p(1-p) / delta^2 with z=1.96, p=0.5, solved for delta.
    delta = 0.98 / np.sqrt(min_count) if min_count else float('inf')

    return {
        'n_held_out': int(oor_mask.sum()),
        'rmse_in': rmse_in,
        'rmse_out': rmse_out,
        'ratio': rmse_out / rmse_in,
        'reach': float(d_oor.max()),
        'min_bin': min_count,
        'delta': delta,
    }


def main():
    started = time.time()
    df = load_pool()
    trimmed = trim_target_tails(df, TARGET_COL)
    X = extract_input_features(trimmed)
    y = trimmed[TARGET_COL].to_numpy()
    axis = trimmed[AXIS_COL].to_numpy()
    print(f'pool: {len(trimmed):,} rows after target trim\n')

    header = (
        f'{"":9s} {"centre":>7s} {"percentiles":>13s} {"ratio":>7s} {"reach":>7s} '
        f'{"held out":>9s} {"min_bin":>8s} {"delta":>7s} {"time":>7s}'
    )
    print(header)
    print('-' * len(header))

    rows = []
    half = TEST_FRACTION / 2

    # The tail runs first as the reference the holes are being compared against.
    for label, centre in [('tail-low', None), *[('hole', c) for c in CENTRES]]:
        if centre is None:
            train_mask, oor_mask = tail_split(axis, 'low', TEST_FRACTION)
            span = '0 to 20'
        else:
            train_mask, oor_mask = hole_split(axis, TEST_FRACTION, centre)
            span = f'{(centre - half) * 100:.0f} to {(centre + half) * 100:.0f}'

        print(f'{label:9s} {centre if centre else 0:7.2f} {span:>13s} ', end='', flush=True)

        t0 = time.time()
        r = score_split(axis, X, y, train_mask, oor_mask)
        elapsed = time.time() - t0
        print(
            f'{r["ratio"]:7.2f} {r["reach"]:7.2f} {r["n_held_out"]:9,d} '
            f'{r["min_bin"]:8,d} {r["delta"]:7.3f} {elapsed:6.1f}s'
        )

        rows.append({'split': label, 'centre': centre, 'percentiles': span, **r})

    print(f'\ntotal {time.time() - started:.1f}s')
    print("\nRead the ratio column first. If it stays near the grid's 0.95 across")
    print('centres, the no-gap-penalty result is robust to placement and moving the')
    print('hole is free. If it climbs, the result depends on the dense middle.')

    constants = {
        'SEED': SEED,
        'TEST_FRACTION': TEST_FRACTION,
        'IN_REGION_TEST_FRACTION': IN_REGION_TEST_FRACTION,
        'N_DISTANCE_BINS': N_DISTANCE_BINS,
        'AXIS_COL': AXIS_COL,
        'TARGET_COL': TARGET_COL,
        'CENTRES': CENTRES,
        'architecture': ARCHITECTURE,
    }
    save_results('hole_placement_narrow', {'rows': rows}, constants=constants)
    save_table('hole_placement_narrow', rows)
    print('saved results/hole_placement.json and .csv')


if __name__ == '__main__':
    main()
