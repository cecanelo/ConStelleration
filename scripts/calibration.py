"""Step 6: calibration diagnostics for the three headline ensembles.

Deliverable one. Steps 2, 4 and 5 already reported that predicted uncertainty is
1.04 to 1.18 times the error it predicts in region and 0.66 to 0.69 out of it.
That is an aggregate ratio, and nobody can act on a ratio. This turns it into
"the 90% interval holds the truth X% of the time, and X falls off past here",
which is a sentence a deferral threshold can rest on.

No training. Reads the three results/mv_ensemble_*_points.csv written by step 5,
which already carry region, distance, truth, mean and all three variance terms
per point. Seconds, not minutes.

Three diagnostics, frozen in 7.1: coverage versus nominal, CRPS, and PIT. Coverage and CRPS bin by distance using metrics.distance_bins,
the same edges the distance-error figure uses, so the two curves overlay.

⚠️ Everything here scores TOTAL predicted variance, which amends 7.5's original
"epistemic alone". Coverage is an interval question and an interval needs the
full spread: in-region epistemic is 0.00887 against an RMSE of 0.01256, so an
epistemic-only 90% interval would read as badly broken on the random split,
where coverage puts the model at 0.966 against a nominal 0.9. That is arithmetic from
using 60% of the standard deviation, not a finding. The epistemic and aleatoric
terms are reported beside the intervals as components instead.

    python3 scripts/calibration.py
"""

import numpy as np
import pandas as pd

from constellaration_uq.metrics import (
    coverage,
    coverage_curve,
    crps_gaussian,
    distance_bins,
    pit_values,
    rms_uncertainty,
)
from constellaration_uq.results import RESULTS_DIR, save_results, save_table

N_BINS = 8

# Six levels is enough to see a coverage curve sag without crowding the figure.
# 0.68 is one sigma, included because it is the level people read intervals at
# by habit even when the plot says something else.
LEVELS = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]

# The one reported in prose. Every bin stores the full curve anyway, so changing
# this later costs a re-plot and no recomputation.
HEADLINE_LEVEL = 0.9

SPLITS = ['random', 'hole', 'tail']


def load_points(split):
    """One step 5 output, with PIT and CRPS added per point.

    Both are scored on total_variance. The two component columns are carried
    through untouched so the per-point file stays self-contained.
    """
    df = pd.read_csv(RESULTS_DIR / f'mv_ensemble_{split}_points.csv')
    df['pit'] = pit_values(df['y_true'], df['mean'], df['total_variance'])
    df['crps'] = crps_gaussian(df['y_true'], df['mean'], df['total_variance'])
    return df


def score(df):
    """Every diagnostic for one set of points, as a flat dict.

    `total_over_rmse` is the aggregate ratio the three ensembles already report,
    recomputed here so the coverage numbers sit next to the quantity they are
    replacing rather than in a different file.
    """
    error = df['y_true'] - df['mean']
    rmse = float(np.sqrt(np.mean(error**2)))
    total = rms_uncertainty(df['total_variance'])

    return {
        'n': len(df),
        'rmse': rmse,
        'crps': float(df['crps'].mean()),
        'epistemic': rms_uncertainty(df['epistemic_variance']),
        'aleatoric': rms_uncertainty(df['aleatoric_variance']),
        'total': total,
        'total_over_rmse': total / rmse,
        'coverage': {
            f'{level:g}': coverage(df['y_true'], df['mean'], df['total_variance'], level)
            for level in LEVELS
        },
        f'coverage_{HEADLINE_LEVEL:g}': coverage(
            df['y_true'], df['mean'], df['total_variance'], HEADLINE_LEVEL
        ),
        'pit_mean': float(df['pit'].mean()),
    }


def bin_out_of_region(df):
    """Coverage and CRPS in the shared quantile bins of distance.

    ⚠️ Out-of-region points only, and the in-region set is a separate anchor at
    distance zero rather than the first bin. Nearly half the evaluation points
    sit at exactly zero distance, so binning the combined set spends three of
    eight bins stacking them against the y axis. Same convention as
    distance_error.py, which is what lets the two figures share an x axis.

    ⚠️ **PIT mean is binned here, and 7.1's cut does not forbid it.** That entry
    cut the per-bin PIT *histogram*, where ten bars rest on about sixty points
    each and jump around from sampling noise alone. A per-bin PIT *mean* rests
    on all ~675 points in the bin and is stable. Two different objects.

    It is needed because the claim it supports is otherwise confounded. The
    aggregate comparison, tail 0.348 against hole 0.531, is read as "the tail is
    biased and the hole is not". But the tail's held-out points reach 2.13 std
    out while the hole's stop at 0.45, so if bias grows with distance that
    contrast could be a distance effect and nothing more. This is the same
    confound the matched-distance analysis exists to remove, one diagnostic
    over. Binning PIT is what lets the comparison be made at matched distance,
    or retracted.
    """
    out = df[df['region'] == 'out']
    rows, masks = distance_bins(out['distance'].to_numpy(), N_BINS)

    for row, sel in zip(rows, masks, strict=True):
        chunk = out[sel]
        row['rmse'] = float(np.sqrt(np.mean((chunk['y_true'] - chunk['mean']) ** 2)))
        row['crps'] = float(chunk['crps'].mean())
        row['total'] = rms_uncertainty(chunk['total_variance'])
        # Below 0.5 the truth keeps landing under the prediction, so the mean
        # is drifting rather than the interval merely being too narrow. That
        # distinction decides whether widening intervals would help.
        row['pit_mean'] = float(chunk['pit'].mean())
        for level, value in zip(
            LEVELS,
            coverage_curve(chunk['y_true'], chunk['mean'], chunk['total_variance'], LEVELS),
            strict=True,
        ):
            row[f'coverage_{level:g}'] = float(value)

    return rows


def print_split(split, sets):
    print(f'\n{split}')
    header = (
        f'{"set":>14s} {"n":>7s} {"rmse":>9s} {"total":>9s} {"ratio":>7s} '
        f'{"crps":>9s} {"cov@0.9":>9s} {"pit_mean":>9s}'
    )
    print(header)
    print('-' * len(header))
    for name, row in sets.items():
        print(
            f'{name:>14s} {row["n"]:7,d} {row["rmse"]:9.5f} {row["total"]:9.5f} '
            f'{row["total_over_rmse"]:7.2f} {row["crps"]:9.5f} '
            f'{row[f"coverage_{HEADLINE_LEVEL:g}"]:9.3f} {row["pit_mean"]:9.3f}'
        )


def print_bins(split, rows):
    print(f'\n{split}, out of region, by distance')
    header = (
        f'{"d_lo":>7s} {"d_hi":>7s} {"n":>6s} {"rmse":>9s} {"crps":>9s} '
        f'{"cov@0.9":>9s} {"cov@0.5":>9s} {"pit":>7s}'
    )
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f'{row["d_lo"]:7.3f} {row["d_hi"]:7.3f} {row["n"]:6,d} '
            f'{row["rmse"]:9.5f} {row["crps"]:9.5f} '
            f'{row[f"coverage_{HEADLINE_LEVEL:g}"]:9.3f} {row["coverage_0.5"]:9.3f} '
            f'{row["pit_mean"]:7.3f}'
        )


def main():
    payload, bin_rows, point_rows = {}, [], []

    for split in SPLITS:
        df = load_points(split)

        sets = {
            'in-region': score(df[df['region'] == 'in']),
            'out-of-region': score(df[df['region'] == 'out']),
        }
        bins = bin_out_of_region(df)

        payload[split] = {'sets': sets, 'bins': bins}
        print_split(split, sets)

        # The random split has no meaningful distance axis, every point sits at
        # roughly zero, so its bins exist in the JSON for completeness and are
        # not worth printing or plotting.
        if split != 'random':
            print_bins(split, bins)

        for row in bins:
            bin_rows.append({'split': split, **row})

        # Per-point PIT and CRPS so the notebook draws the histograms without
        # recomputing. Rounded at write, as the step 5 tables are.
        for record in df.to_dict('records'):
            point_rows.append(
                {
                    'split': split,
                    'region': record['region'],
                    'distance': round(float(record['distance']), 6),
                    'axis': round(float(record['axis']), 6),
                    'y_true': round(float(record['y_true']), 8),
                    'mean': round(float(record['mean']), 8),
                    'total_variance': float(record['total_variance']),
                    'pit': round(float(record['pit']), 6),
                    'crps': round(float(record['crps']), 8),
                }
            )

    # The headline contrast, in the one form that is actionable. Nominal is 0.9
    # in every row, so the only thing moving is honesty.
    print(f'\ncoverage at nominal {HEADLINE_LEVEL:g}, in region versus out')
    for split in SPLITS:
        sets = payload[split]['sets']
        inside = sets['in-region'][f'coverage_{HEADLINE_LEVEL:g}']
        outside = sets['out-of-region'][f'coverage_{HEADLINE_LEVEL:g}']
        print(f'  {split:>7s}  {inside:.3f} in   {outside:.3f} out   {outside - inside:+.3f}')

    constants = {
        'N_BINS': N_BINS,
        'LEVELS': LEVELS,
        'HEADLINE_LEVEL': HEADLINE_LEVEL,
        'SPLITS': SPLITS,
        'scored_on': 'total_variance, per 7.1 and the 7.5 amendment',
        'source': 'results/mv_ensemble_{split}_points.csv',
    }
    save_results('calibration', payload, constants=constants)
    save_table('calibration_bins', bin_rows)
    save_table('calibration_points', point_rows)
    print('\nsaved results/calibration.json, _bins.csv and _points.csv')


if __name__ == '__main__':
    main()
