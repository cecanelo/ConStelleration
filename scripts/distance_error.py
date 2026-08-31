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

import argparse
import time
from itertools import pairwise

import numpy as np
from sklearn.model_selection import train_test_split

from constellaration_uq.baseline import ARCHITECTURE, fit_mlp, load_pool
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.metrics import distance_bins
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

# Width of the fixed windows the matched-distance comparison uses. 0.1 std keeps
# the thinnest window near 300 tail points over the 0 to 0.45 shared range.
WINDOW_WIDTH = 0.1

AXIS_COL = 'metrics.aspect_ratio'
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'

# Percentile 30 puts the hole immediately above the tail's 0 to 20 without
# overlapping it, which is what makes the two experiments independent.
SPLITS = {
    'random': lambda axis, seed=SEED: random_split(axis, seed, TEST_FRACTION),
    'hole_p30': lambda axis, seed=SEED: hole_split(axis, TEST_FRACTION, HOLE_CENTRE),
    'tail_low': lambda axis, seed=SEED: tail_split(axis, 'low', TEST_FRACTION),
}


def bin_by_distance(distance, abs_error, n_bins):
    """Summarise error in the shared quantile bins of distance.

    The bin geometry comes from metrics.distance_bins so the calibration curves
    land on exactly these edges. Only the two error columns are added here.
    """
    rows, masks = distance_bins(distance, n_bins)
    for row, sel in zip(rows, masks, strict=True):
        row['rmse'] = float(np.sqrt(np.mean(abs_error[sel] ** 2)))
        row['mae'] = float(np.mean(abs_error[sel]))
    return rows


def matched_windows(all_points, width=WINDOW_WIDTH):
    """Hole against tail inside identical distance windows.

    ⚠️ **This exists because reading the two binned curves at their bin medians
    does not compare like with like, and a headline number was wrong for exactly
    that reason.** The bins are equal-count, so their widths differ wherever the
    two splits differ in density: the tail's second bin spans 0.13 to 0.31 while
    the hole's are about 0.05 wide there. RMSE inside a wide bin is dominated by
    its far edge, so quoting that bin's value "at 0.2" reports something closer
    to the error at 0.3. The premium was reported as a flat 15% at both 0.2 and
    0.4 std; measured in shared windows it is near zero at 0.2 and about 27% at
    0.4.

    Fixed-width windows here, not quantile bins, because the whole point is that
    both splits are asked about the same interval of distance. Equal counts and
    equal windows cannot both hold when the two splits have different densities,
    and for this comparison the window is what has to match.

    Only the shared range is covered. The hole cannot reach past its own widest
    gap, so beyond that there is nothing to compare the tail against and the
    question stops having an answer.
    """
    rows = []
    for name in ('hole_p30', 'tail_low'):
        rows.append([p for p in all_points if p['split'] == name and p['region'] == 'out'])
    hole, tail = rows

    shared = min(max(p['distance'] for p in hole), max(p['distance'] for p in tail))
    edges = np.arange(0.0, shared + width, width)

    def stats(points, lo, hi):
        errs = np.array([p['abs_error'] for p in points if lo <= p['distance'] < hi])
        dist = np.array([p['distance'] for p in points if lo <= p['distance'] < hi])
        if not len(errs):
            return None
        return {
            'n': len(errs),
            'd_median': float(np.median(dist)),
            'rmse': float(np.sqrt(np.mean(errs**2))),
        }

    out = []
    for lo, hi in pairwise(edges):
        h, t = stats(hole, lo, hi), stats(tail, lo, hi)
        if h is None or t is None:
            continue
        out.append(
            {
                'window_lo': round(float(lo), 4),
                'window_hi': round(float(hi), 4),
                'n_hole': h['n'],
                'n_tail': t['n'],
                'd_median_hole': h['d_median'],
                'd_median_tail': t['d_median'],
                'rmse_hole': h['rmse'],
                'rmse_tail': t['rmse'],
                'premium': t['rmse'] / h['rmse'],
            }
        )

    # The whole shared range as one row, which is the number to quote if a
    # single one is wanted. Labelled so it cannot be mistaken for a window.
    h, t = stats(hole, 0.0, shared), stats(tail, 0.0, shared)
    out.append(
        {
            'window_lo': 0.0,
            'window_hi': round(float(shared), 4),
            'n_hole': h['n'],
            'n_tail': t['n'],
            'd_median_hole': h['d_median'],
            'd_median_tail': t['d_median'],
            'rmse_hole': h['rmse'],
            'rmse_tail': t['rmse'],
            'premium': t['rmse'] / h['rmse'],
        }
    )
    return out


def run_split(axis, X, y, make_split, seed=SEED):
    train_mask, oor_mask = make_split(axis)

    # An in-region held-out slice carved from the training region only, so the
    # near-zero end of the curve is honest held-out error rather than training
    # error, and the early-stopping validation set never sees out-of-region data.
    train_idx = np.flatnonzero(train_mask)
    fit_idx, in_idx = train_test_split(
        train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=seed
    )

    predict = fit_mlp(X[fit_idx], y[fit_idx], seed)

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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help='replication seed. Varies the MLP init, the in-region slice and, on '
        'the random split, which rows are held out. The hole and tail held-out '
        'sets are quantile cutoffs on the axis and do not move.',
    )
    args = parser.parse_args()
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
        summary, points = run_split(axis, X, y, lambda a, f=make_split: f(a, args.seed), args.seed)
        elapsed = time.time() - t0

        min_bin = min(b['n'] for b in summary['bins'])
        print(
            f'{summary["rmse_in"]:9.5f} {summary["rmse_out"]:9.5f} '
            f'{summary["ratio"]:7.2f} {summary["reach"]:7.2f} '
            f'{len(summary["bins"]):5d} {min_bin:8,d} {elapsed:6.1f}s'
        )

        results[name] = summary
        all_points.extend({'split': name, **p} for p in points)

    windows = matched_windows(all_points)

    print('\nmatched distance: hole against tail in identical windows')
    header = (
        f'{"window":>13s} {"n_hole":>7s} {"n_tail":>7s} {"med_h":>7s} {"med_t":>7s} '
        f'{"rmse_hole":>10s} {"rmse_tail":>10s} {"premium":>8s}'
    )
    print(header)
    print('-' * len(header))
    for row in windows[:-1]:
        label = f'{row["window_lo"]:.2f} to {row["window_hi"]:.2f}'
        print(
            f'{label:>13s} {row["n_hole"]:7,d} {row["n_tail"]:7,d} '
            f'{row["d_median_hole"]:7.3f} {row["d_median_tail"]:7.3f} '
            f'{row["rmse_hole"]:10.5f} {row["rmse_tail"]:10.5f} {row["premium"]:8.3f}'
        )
    whole = windows[-1]
    print(
        f'{"shared range":>13s} {whole["n_hole"]:7,d} {whole["n_tail"]:7,d} '
        f'{whole["d_median_hole"]:7.3f} {whole["d_median_tail"]:7.3f} '
        f'{whole["rmse_hole"]:10.5f} {whole["rmse_tail"]:10.5f} {whole["premium"]:8.3f}'
    )

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')
    print('\nRead the bins, not the ratio. The curve is the deliverable: error')
    print('against distance, with the hole and the tail overlaid over the range')
    print('they share, and the tail continuing alone past it.')
    print('\nThe premium column is the edge cost with distance controlled for.')
    print('It is not flat: at short range a gap and an edge cost the same, and')
    print('the edge penalty appears as the distance grows.')

    constants = {
        'SEED': args.seed,
        'TEST_FRACTION': TEST_FRACTION,
        'IN_REGION_TEST_FRACTION': IN_REGION_TEST_FRACTION,
        'HOLE_CENTRE': HOLE_CENTRE,
        'N_BINS': N_BINS,
        'WINDOW_WIDTH': WINDOW_WIDTH,
        'AXIS_COL': AXIS_COL,
        'TARGET_COL': TARGET_COL,
        'architecture': ARCHITECTURE,
    }
    payload = {
        'total_seconds': round(total_seconds, 1),
        'splits': results,
        'matched_windows': windows,
    }

    # Seed 0 keeps the unsuffixed name so replication runs are additive and
    # nothing downstream has to learn about seeds.
    out_name = 'distance_error' + (f'_s{args.seed}' if args.seed != SEED else '')
    save_results(out_name, payload, constants=constants)
    # Per-point rows so the bins can be redrawn without refitting. Roughly 29k
    # rows; the stage 2 rebinning is the precedent for wanting this on disk.
    save_table(out_name + '_points', all_points)
    save_table(out_name + '_matched', windows)
    print(f'saved results/{out_name}.json, _points.csv and _matched.csv')


if __name__ == '__main__':
    main()
