"""Step 7: the deferral curve.

Deliverable two. Steps 2 to 6 established that the uncertainty signal is
miscalibrated off-distribution: at the compact edge a stated 90% interval holds
0.78 of the truth, and 0.59 in the furthest bin. This asks the question that
actually matters for using the surrogate anyway. **A signal can be useless as an
absolute interval and still be perfectly good at ranking.** Deferral only needs
the ranking: send the least trustworthy shapes to VMEC++ and keep the rest.

The picture is a trade. Defer nothing and you make zero solver calls and eat the
full surrogate error. Defer everything and you make every solver call and eat
none. The curve between those ends is the whole result, and its shape says
whether the uncertainty estimate is worth having.

No training. Reads results/mv_ensemble_tail_points.csv, which already carries
the per-point mean and all three variance terms. Seconds.

Five lines per region, from 8.1 and 8.4:

    epistemic-ranked   defer where the members disagree most
    aleatoric-ranked   defer where the members' own variances are largest
    total-ranked       defer where the full predicted spread is largest
    random             the floor, defer at the same rate but pick at random
    oracle             the ceiling, defer by the per-point score itself

The three "mine" curves cost one extra sort each, and which of them wins is a
result rather than a setup detail. Without the floor the curve is unreadable,
since any monotone-in-anything ranking slopes downward. The gap to random is the
finding; the gap to oracle is the headroom left on the table.

Aleatoric-ranked was cut in 8.1 and reinstated 2026-09-03. The cut assumed the
term would sit at the numerical floor, so ranking by it would be ranking by
noise and the curve would land on random by construction. Step 2 falsified that
premise: aleatoric is model misfit, larger than epistemic in region, and misfit
is informative about local difficulty. It is measured here rather than argued.

The oracle defers by whichever score is plotted (8.4, corrected). Ranked by
anything else it stops bounding the curve it is drawn against, and the "mine"
curves could cross their own ceiling, which reads as a bug.

Two scoring rules, selected with --score. **MAE is the deliverable and the
default** (8.3, reversed 2026-09-04): CRPS scales with sigma for a Gaussian and
this curve ranks by sigma, so CRPS strips high-scoring points partly by
construction, while MAE never touches the predicted variance. CRPS runs
alongside as the check that the finding also holds under a proper scoring rule.
They agree: 40.8% cut at 20% deferral and 63% of the headroom on MAE, 41.0% and
63% on CRPS. Filenames follow the figures: CRPS unsuffixed, MAE as _mae.

Deferred points are credited with the exact solver value, contributing zero
error. That assumes the fallback always succeeds, which 8.6 measured and closed
on 2026-08-31: about a third of VMEC++ calls fail in the region deferral sends
work to, so this curve is optimistic in level. The gap to random survives, since
random deferral draws from the same population and eats the same failure rate.

    python3 scripts/deferral.py
    python3 scripts/deferral.py --score crps
"""

import argparse

import numpy as np
import pandas as pd

from constellaration_uq.metrics import crps_gaussian, deferral_curve
from constellaration_uq.results import RESULTS_DIR, save_results, save_table

SEED = 0


def absolute_error(y_true, mean, variance):
    """MAE's per-point score. Takes the variance it does not use so the two
    scorers share a signature and selecting one swaps a single callable."""
    del variance
    return np.abs(np.asarray(y_true) - np.asarray(mean))


# 8.3, reversed 2026-09-04: MAE is the deliverable and CRPS the supporting
# check. CRPS scales with sigma for a Gaussian and the curve ranks by sigma, so
# it strips high-CRPS points partly by construction; MAE never touches the
# predicted variance. Each writes its own files, so neither can overwrite the
# other and the pair can be compared.
SCORERS = {
    'mae': (absolute_error, 'MAE'),
    'crps': (crps_gaussian, 'CRPS'),
}

# The tail split is the headline (8.5). It is the extrapolation condition the
# ConStellaration paper actually warned about, and the case where deferral has
# to earn its keep. In-region runs alongside as contrast, never as a rival.
SPLIT = 'tail'
REGIONS = {'out': 'out-of-region', 'in': 'in-region'}

# 2% steps. Fine enough to read a knee off the plot, coarse enough that the
# curve file stays small.
RATES = np.linspace(0.0, 1.0, 51)

# The random floor is an average over permutations rather than the straight line
# its expectation traces, so the floor is measured on the same footing as the
# other three rather than asserted analytically.
RANDOM_REPEATS = 50

REPORT_AT = [0.0, 0.1, 0.2, 0.3, 0.5]


def random_curve(crps, rates, seed):
    """The floor: defer at the same rate, choose at random."""
    rng = np.random.default_rng(seed)
    runs = [deferral_curve(crps, rng.permutation(len(crps)), rates) for _ in range(RANDOM_REPEATS)]
    return np.mean(runs, axis=0)


def rate_to_halve(rates, values):
    """Deferral rate at which CRPS first falls to half its no-deferral value.

    The one number that turns the figure into a sentence: "solve x% of designs
    and the error halves". Linearly interpolated between grid points, and None
    if the curve never gets there, which cannot happen here since every curve
    ends at zero but is guarded anyway.
    """
    target = values[0] / 2
    below = np.flatnonzero(values <= target)
    if not len(below):
        return None
    i = below[0]
    if i == 0:
        return 0.0
    span = values[i - 1] - values[i]
    frac = (values[i - 1] - target) / span if span else 0.0
    return float(rates[i - 1] + frac * (rates[i] - rates[i - 1]))


def run_region(df, score_fn):
    """All five curves for one evaluation set, under one scoring rule."""
    score = score_fn(df['y_true'], df['mean'], df['total_variance'])

    curves = {
        'epistemic': deferral_curve(score, df['epistemic_variance'], RATES),
        'aleatoric': deferral_curve(score, df['aleatoric_variance'], RATES),
        'total': deferral_curve(score, df['total_variance'], RATES),
        'random': random_curve(score, RATES, SEED),
        # Ranking by the score itself is the best any ranking can do, which is
        # exactly what makes it the ceiling. It has to be the score actually
        # plotted, or it stops bounding the curve it is drawn against.
        'oracle': deferral_curve(score, score, RATES),
    }

    summary = {
        name: {
            'score_at': {f'{r:g}': float(np.interp(r, RATES, values)) for r in REPORT_AT},
            'rate_to_halve': rate_to_halve(RATES, values),
            # Area under the curve. One number for the whole trade, lower is
            # better, and the only fair way to compare two rankings that cross.
            'auc': float(np.trapz(values, RATES)),
        }
        for name, values in curves.items()
    }
    return curves, summary


def print_region(label, summary, score_label):
    print(f'\n{label}, {score_label} of the hybrid system at a given deferral rate')
    header = f'{"ranking":>10s} ' + ' '.join(f'{r:>9.0%}' for r in REPORT_AT)
    header += f' {"auc":>9s} {"halve at":>9s}'
    print(header)
    print('-' * len(header))
    for name in ('epistemic', 'aleatoric', 'total', 'random', 'oracle'):
        row = summary[name]
        cells = ' '.join(f'{row["score_at"][f"{r:g}"]:9.5f}' for r in REPORT_AT)
        halve = row['rate_to_halve']
        halve_text = f'{halve:9.1%}' if halve is not None else f'{"never":>9s}'
        print(f'{name:>10s} {cells} {row["auc"]:9.5f}{halve_text}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', choices=sorted(SCORERS), default='mae')
    args = parser.parse_args()
    score_fn, score_label = SCORERS[args.score]
    # CRPS keeps the unsuffixed filenames it already had, and MAE is suffixed,
    # matching the figures: deferral_curve.png is CRPS, deferral_curve_mae.png
    # is MAE. Same rule in both places, so a reader learns it once.
    suffix = '' if args.score == 'crps' else f'_{args.score}'

    df = pd.read_csv(RESULTS_DIR / f'mv_ensemble_{SPLIT}_points.csv')

    payload, rows = {}, []
    for region, label in REGIONS.items():
        subset = df[df['region'] == region]
        curves, summary = run_region(subset, score_fn)

        payload[region] = {'n': len(subset), 'rankings': summary}
        print_region(f'{label}, n = {len(subset):,}', summary, score_label)

        for name, values in curves.items():
            for rate, value in zip(RATES, values, strict=True):
                rows.append(
                    {
                        'region': region,
                        'ranking': name,
                        'deferred': round(float(rate), 4),
                        # Solver calls SAVED, which is the x axis of 8.2 and the
                        # complement of the rate deferred. Stored explicitly so
                        # the plot cannot get the direction backwards.
                        'saved': round(1.0 - float(rate), 4),
                        'score': float(value),
                    }
                )

    out = payload['out']['rankings']
    print('\nthe headline, out of region at the compact edge')
    for name in ('epistemic', 'aleatoric', 'total'):
        no_deferral = out[name]['score_at']['0']
        at_20 = out[name]['score_at']['0.2']
        print(
            f'  {name:>10s}-ranked: solving the worst 20% cuts {score_label} '
            f'{no_deferral:.5f} to {at_20:.5f}, {1 - at_20 / no_deferral:.0%} lower'
        )
    random_20 = out['random']['score_at']['0.2']
    oracle_20 = out['oracle']['score_at']['0.2']
    print(f'  {"random":>10s} at 20%: {random_20:.5f}   {"oracle":>6s}: {oracle_20:.5f}')
    print('\nRead the gap to random as the result and the gap to oracle as the headroom.')

    constants = {
        'SPLIT': SPLIT,
        'SEED': SEED,
        'RATES': [float(r) for r in RATES],
        'RANDOM_REPEATS': RANDOM_REPEATS,
        # The file says which rule it was scored under, so score_at and the
        # csv's score column stay generic instead of naming one metric.
        'score': args.score,
        'y_axis': f'{score_label} of the hybrid system, physical units, '
        'deferred points credited zero',
        'x_axis': 'fraction of solver calls saved, per 8.2',
        'source': f'results/mv_ensemble_{SPLIT}_points.csv',
    }
    save_results(f'deferral{suffix}', payload, constants=constants)
    save_table(f'deferral_curve{suffix}', rows)
    print(f'\nsaved results/deferral{suffix}.json and deferral_curve{suffix}.csv')


if __name__ == '__main__':
    main()
