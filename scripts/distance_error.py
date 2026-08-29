"""Per-point error against distance from the training region.

Deliverable one in single-model form. Every earlier figure summarised a whole
experiment as one number, which is why they kept being misread: a handful of
points on a plot, each of them an entire model fit, in a coordinate system no
reader could hold in their head. This one plots configurations rather than
experiments. x is how far a shape sits from the nearest training point along the
split axis, y is what the model's error actually was there.

Three splits, one MLP each, same recipe as the day 1-2 grid so the numbers stay
comparable to it. The ensemble version later swaps in ten members and adds the
uncertainty bands. The figure does not change shape.

Two choices worth knowing about, both recorded in the code below:

  Distance is measured against the points the model actually saw, not against
  the whole training region, so the in-region slice earns its near-zero distance
  rather than being handed it.

  Bins are quantile edges per split, not shared uniform edges. Distance is
  heavily skewed and the two splits differ tenfold in range, so uniform edges
  would leave the hole with two usable bins and the tail with one crowded one.
  This is the same trap that flipped the verdict in the stage 2 noise floor check
  before it was rebinned.
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split

from constellaration_uq.baseline import ARCHITECTURE, fit_mlp, load_pool
from constellaration_uq.data import extract_input_features, trim_target_tails
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
N_BINS = 8

AXIS_COL = 'metrics.aspect_ratio'
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'

# Percentile 30 puts the hole immediately above the tail's 0 to 20 without
# overlapping it, which is what makes the two experiments independent.
SPLITS = {
    'random': lambda axis: random_split(axis, SEED, TEST_FRACTION),
    'hole_p30': lambda axis: hole_split(axis, TEST_FRACTION, HOLE_CENTRE),
    'tail_low': lambda axis: tail_split(axis, 'low', TEST_FRACTION),
}


def bin_by_distance(distance, abs_error, n_bins):
    """Summarise error in quantile bins of distance.

    Equal-count bins rather than equal-width, so every point on the curve rests
    on the same amount of evidence and the thin far tail cannot produce a bin of
    two configurations masquerading as a measurement.
    """
    edges = np.quantile(distance, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    which = np.clip(np.searchsorted(edges, distance, side='right') - 1, 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        sel = which == b
        if not sel.any():
            # Duplicate quantile edges collapse a bin. Happens on the random
            # split, where most distances are identical and tiny.
            continue
        bins.append(
            {
                'n': int(sel.sum()),
                'd_lo': float(edges[b]),
                'd_hi': float(edges[b + 1]),
                'd_median': float(np.median(distance[sel])),
                'rmse': float(np.sqrt(np.mean(abs_error[sel] ** 2))),
                'mae': float(np.mean(abs_error[sel])),
            }
        )
    return bins


def run_split(axis, X, y, make_split):
    train_mask, oor_mask = make_split(axis)

    # An in-region held-out slice carved from the training region only, so the
    # near-zero end of the curve is honest held-out error rather than training
    # error, and the early-stopping validation set never sees out-of-region data.
    train_idx = np.flatnonzero(train_mask)
    fit_idx, in_idx = train_test_split(
        train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=SEED
    )

    predict = fit_mlp(X[fit_idx], y[fit_idx], SEED)

    # Distance is measured against the fit set, not train_mask. Using train_mask
    # would include the in-region slice in its own reference set and hand those
    # points a distance of zero by construction rather than by measurement.
    fit_mask = np.zeros(len(axis), dtype=bool)
    fit_mask[fit_idx] = True
    distance = distance_from_training_region(axis, fit_mask)

    oor_idx = np.flatnonzero(oor_mask)
    eval_idx = np.concatenate([in_idx, oor_idx])
    predicted = predict(X[eval_idx])
    abs_error = np.abs(y[eval_idx] - predicted)

    n_in = len(in_idx)
    d_eval = distance[eval_idx]

    # The in-region slice is one anchor at distance zero, not part of the binned
    # curve. Nearly half the evaluation points sit at exactly zero, so quantile
    # binning over the combined set spends three of eight bins stacking them in a
    # vertical smear on the y axis and leaves the actual curve under-resolved.
    summary = {
        'n_fit': len(fit_idx),
        'n_in': n_in,
        'n_out': len(oor_idx),
        'rmse_in': float(np.sqrt(np.mean(abs_error[:n_in] ** 2))),
        'rmse_out': float(np.sqrt(np.mean(abs_error[n_in:] ** 2))),
        'reach': float(distance[oor_idx].max()),
        'in_region': {
            'n': n_in,
            'd_median': float(np.median(d_eval[:n_in])),
            'rmse': float(np.sqrt(np.mean(abs_error[:n_in] ** 2))),
            'mae': float(np.mean(abs_error[:n_in])),
        },
        'bins': bin_by_distance(d_eval[n_in:], abs_error[n_in:], N_BINS),
    }
    summary['ratio'] = summary['rmse_out'] / summary['rmse_in']

    points = [
        {
            'region': 'in' if i < n_in else 'out',
            'distance': round(float(d), 6),
            'axis': round(float(a), 6),
            'y_true': round(float(t), 8),
            'y_pred': round(float(p), 8),
            'abs_error': round(float(e), 8),
        }
        for i, (d, a, t, p, e) in enumerate(
            zip(
                distance[eval_idx],
                axis[eval_idx],
                y[eval_idx],
                predicted,
                abs_error,
                strict=True,
            )
        )
    ]
    return summary, points


def main():
    started = time.time()
    df = load_pool()
    trimmed = trim_target_tails(df, TARGET_COL)
    X = extract_input_features(trimmed)
    y = trimmed[TARGET_COL].to_numpy()
    axis = trimmed[AXIS_COL].to_numpy()
    print(f'pool: {len(trimmed):,} rows after target trim\n')

    header = (
        f'{"split":10s} {"rmse_in":>9s} {"rmse_out":>9s} {"ratio":>7s} '
        f'{"reach":>7s} {"bins":>5s} {"min_bin":>8s} {"time":>7s}'
    )
    print(header)
    print('-' * len(header))

    results = {}
    all_points = []
    for name, make_split in SPLITS.items():
        print(f'{name:10s} ', end='', flush=True)
        t0 = time.time()
        summary, points = run_split(axis, X, y, make_split)
        elapsed = time.time() - t0

        min_bin = min(b['n'] for b in summary['bins'])
        print(
            f'{summary["rmse_in"]:9.5f} {summary["rmse_out"]:9.5f} '
            f'{summary["ratio"]:7.2f} {summary["reach"]:7.2f} '
            f'{len(summary["bins"]):5d} {min_bin:8,d} {elapsed:6.1f}s'
        )

        results[name] = summary
        all_points.extend({'split': name, **p} for p in points)

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')
    print('\nRead the bins, not the ratio. The curve is the deliverable: error')
    print('against distance, with the hole and the tail overlaid over the range')
    print('they share, and the tail continuing alone past it.')

    constants = {
        'SEED': SEED,
        'TEST_FRACTION': TEST_FRACTION,
        'IN_REGION_TEST_FRACTION': IN_REGION_TEST_FRACTION,
        'HOLE_CENTRE': HOLE_CENTRE,
        'N_BINS': N_BINS,
        'AXIS_COL': AXIS_COL,
        'TARGET_COL': TARGET_COL,
        'architecture': ARCHITECTURE,
    }
    payload = {'total_seconds': round(total_seconds, 1), 'splits': results}

    save_results('distance_error', payload, constants=constants)
    # Per-point rows so the bins can be redrawn without refitting. Roughly 29k
    # rows; the stage 2 rebinning is the precedent for wanting this on disk.
    save_table('distance_error_points', all_points)
    print('saved results/distance_error.json and results/distance_error_points.csv')


if __name__ == '__main__':
    main()
