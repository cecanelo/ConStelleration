"""Replication: every headline number across seeds, with its spread.

Until 2026-08-31 the only replicated experiment in this project was the N-sweep.
Everything else, including all six calibration ratios, both deliverable curves
and the matched-distance comparison, came from seed 0 alone. The effects are far
larger than the seed noise the sweep measured, but "larger than the variability
of a different experiment" is an argument by proxy, not evidence, and it is the
first thing a reviewer is entitled to ask about.

This reads whatever per-seed results exist and reports mean and spread for each
headline quantity. It is additive: seed 0 keeps the unsuffixed filenames every
other script reads, and `--seed N` runs write `_s{N}` alongside. Nothing here
changes a headline number; it puts an error bar on one.

What varies with the seed, and what does not. Member initialisation, shuffle
order, the in-region slice and the validation set all move. **The hole and tail
held-out sets do not**, because they are quantile cutoffs on the axis with no
random component, so the out-of-region numbers are replicated on a fixed test
set. Only the random split's test set moves, and that split is the control.

Member seeds are spaced by N_MEMBERS in mv_ensemble.py. `train_mv_ensemble`
uses `seed = base_seed + k`, so consecutive replication seeds would otherwise
share nine of ten member initialisations and this whole exercise would measure
almost nothing.

    python3 scripts/seed_spread.py
"""

import itertools

import numpy as np
import pandas as pd

from constellaration_uq.metrics import (
    coverage,
    crps_gaussian,
    deferral_curve,
    pit_values,
    rms_uncertainty,
)
from constellaration_uq.results import RESULTS_DIR, save_results, save_table

SPLITS = ['random', 'hole', 'tail']
COVERAGE_LEVEL = 0.9
RATES = np.linspace(0.0, 1.0, 51)


def available_seeds(stem):
    """Seeds with results on disk, seed 0 being the unsuffixed file."""
    seeds = [0] if (RESULTS_DIR / f'{stem}_points.csv').exists() else []
    for path in sorted(RESULTS_DIR.glob(f'{stem}_s*_points.csv')):
        suffix = path.name[len(stem) + 2 :].split('_')[0]
        if suffix.isdigit():
            seeds.append(int(suffix))
    return sorted(set(seeds))


def points_for(split, seed):
    stem = f'mv_ensemble_{split}' + ('' if seed == 0 else f'_s{seed}')
    return pd.read_csv(RESULTS_DIR / f'{stem}_points.csv')


def score_region(df):
    """The headline quantities for one region of one seed."""
    error = df['y_true'] - df['mean']
    rmse = float(np.sqrt(np.mean(error**2)))
    total = rms_uncertainty(df['total_variance'])
    return {
        'rmse': rmse,
        'epistemic': rms_uncertainty(df['epistemic_variance']),
        'aleatoric': rms_uncertainty(df['aleatoric_variance']),
        'total': total,
        'ratio': total / rmse,
        'coverage': coverage(df['y_true'], df['mean'], df['total_variance'], COVERAGE_LEVEL),
        'pit_mean': float(pit_values(df['y_true'], df['mean'], df['total_variance']).mean()),
    }


def deferral_aucs(df):
    """The three ranking signals against each other, out of region.

    Margins here are a few percent, small enough that seed 0 alone cannot settle
    them. All three come from one ensemble, so the comparison is paired: what
    varies within a run is the signal and what varies between runs is the level.
    Count wins per run rather than comparing the marginals, which overlap.

    Aleatoric joined the set on 2026-09-03. It was cut in 8.1 on the assumption
    that it would sit at the numerical floor, which step 2 falsified.
    """
    crps = crps_gaussian(df['y_true'], df['mean'], df['total_variance'])
    return {
        f'auc_{name}': float(np.trapz(deferral_curve(crps, signal, RATES), RATES))
        for name, signal in (
            ('total', df['total_variance']),
            ('epistemic', df['epistemic_variance']),
            ('aleatoric', df['aleatoric_variance']),
        )
    }


# Lower AUC is better, so a "win" is the smaller number.
RANKING_SIGNALS = ('total', 'aleatoric', 'epistemic')


def summarise(values):
    """mean, spread and range over seeds, as a flat dict."""
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'n_seeds': len(arr),
    }


def collect():
    per_seed, rows = {}, []
    for split in SPLITS:
        seeds = available_seeds(f'mv_ensemble_{split}')
        if not seeds:
            continue

        measured = {}
        for seed in seeds:
            df = points_for(split, seed)
            entry = {region: score_region(df[df['region'] == region]) for region in ('in', 'out')}
            entry['out'].update(deferral_aucs(df[df['region'] == 'out']))
            # The gap the whole project is about, per seed rather than once.
            entry['gap'] = entry['out']['rmse'] / entry['in']['rmse']
            measured[seed] = entry

        keys = [
            (region, key)
            for region in ('in', 'out')
            for key in ('rmse', 'epistemic', 'aleatoric', 'total', 'ratio', 'coverage', 'pit_mean')
        ]
        summary = {
            f'{region}_{key}': summarise([measured[s][region][key] for s in seeds])
            for region, key in keys
        }
        summary['gap'] = summarise([measured[s]['gap'] for s in seeds])
        aucs = {
            name: [measured[s]['out'][f'auc_{name}'] for s in seeds] for name in RANKING_SIGNALS
        }
        for name, values in aucs.items():
            summary[f'auc_{name}'] = summarise(values)

        per_seed[split] = {'seeds': seeds, 'summary': summary, 'deferral_aucs': aucs}
        for key, stats in summary.items():
            rows.append({'split': split, 'quantity': key, **stats})

    return per_seed, rows


def matched_premium():
    """The matched-distance edge cost across seeds of distance_error.py.

    Carried separately because it comes from single MLPs rather than ensembles,
    and because at 12% it is the headline claim closest to seed noise.
    """
    values = []
    for path in [
        RESULTS_DIR / 'distance_error_matched.csv',
        *sorted(RESULTS_DIR.glob('distance_error_s*_matched.csv')),
    ]:
        if not path.exists():
            continue
        table = pd.read_csv(path)
        # The last row is the whole shared range, written that way by
        # distance_error.matched_windows.
        values.append(float(table.iloc[-1]['premium']))
    return summarise(values) if values else None


def paired_wins(per_seed):
    """How often each signal beats each other one, counted run by run.

    A run is one split at one seed, so three splits by three seeds is nine. The
    marginals overlap across seeds while the sign of the difference is stable,
    which is exactly the case where pairing is the whole point.
    """
    result = {}
    for a, b in itertools.combinations(RANKING_SIGNALS, 2):
        wins, margins = 0, []
        for block in per_seed.values():
            paired = zip(block['deferral_aucs'][a], block['deferral_aucs'][b], strict=True)
            for auc_a, auc_b in paired:
                wins += auc_a < auc_b
                margins.append(1.0 - auc_a / auc_b)
        result[f'{a}_over_{b}'] = {
            'wins': int(wins),
            'runs': len(margins),
            'margin_mean': float(np.mean(margins)),
            'margin_min': float(np.min(margins)),
            'margin_max': float(np.max(margins)),
        }
    return result


def print_paired(wins):
    print('paired deferral comparison, out of region, one run per split per seed')
    print(f'{"comparison":>24s} {"wins":>9s} {"margin":>9s} {"min":>8s} {"max":>8s}')
    for name, row in wins.items():
        print(
            f'{name.replace("_", " "):>24s} {row["wins"]:4d}/{row["runs"]:<4d} '
            f'{row["margin_mean"]:8.1%} {row["margin_min"]:7.1%} {row["margin_max"]:7.1%}'
        )
    print('Lower AUC is better, so a win is the smaller number and a positive margin.\n')


def print_table(per_seed):
    header = (
        f'{"split":>8s} {"quantity":>16s} {"seeds":>6s} {"mean":>10s} '
        f'{"std":>9s} {"min":>10s} {"max":>10s} {"spread":>8s}'
    )
    print(header)
    print('-' * len(header))
    for split, block in per_seed.items():
        for key, stats in block['summary'].items():
            relative = stats['std'] / abs(stats['mean']) if stats['mean'] else 0.0
            print(
                f'{split:>8s} {key:>16s} {stats["n_seeds"]:6d} {stats["mean"]:10.5f} '
                f'{stats["std"]:9.5f} {stats["min"]:10.5f} {stats["max"]:10.5f} '
                f'{relative:7.1%}'
            )
        print()


def main():
    per_seed, rows = collect()
    if not per_seed:
        raise SystemExit('no mv_ensemble points files found')

    print_table(per_seed)

    wins = paired_wins(per_seed)
    print_paired(wins)

    premium = matched_premium()
    if premium:
        print(
            f'matched-distance edge cost over the shared range: '
            f'{premium["mean"]:.3f} ± {premium["std"]:.3f} '
            f'over {premium["n_seeds"]} seed(s), range {premium["min"]:.3f} to {premium["max"]:.3f}'
        )

    seeds = per_seed[SPLITS[-1]]['seeds']
    if len(seeds) < 2:
        print('\nWarning: only one seed found. Run mv_ensemble.py --seed 1 and 2 first.')
    else:
        print(
            f'\nRead the spread column against the effect being claimed. A difference\n'
            f'many times the spread survives replication; one comparable to it does not.\n'
            f'Seeds: {seeds}.'
        )

    save_results(
        'seed_spread',
        {'splits': per_seed, 'matched_premium': premium, 'paired_deferral': wins},
        constants={
            'SPLITS': SPLITS,
            'COVERAGE_LEVEL': COVERAGE_LEVEL,
            'RANKING_SIGNALS': list(RANKING_SIGNALS),
            'note': 'seed varies member init, in-region slice and validation set; '
            'hole and tail held-out sets are deterministic',
        },
    )
    save_table('seed_spread', rows)
    print('saved results/seed_spread.json and .csv')


if __name__ == '__main__':
    main()
