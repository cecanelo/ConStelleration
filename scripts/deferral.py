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

Four lines per region, frozen in 8.1 and 8.4:

    epistemic-ranked   defer where the members disagree most
    total-ranked       defer where the full predicted spread is largest
    random             the floor, defer at the same rate but pick at random
    oracle             the ceiling, defer by the per-point CRPS itself

The two "mine" curves cost one extra sort, and which of them wins is a result
rather than a setup detail. Without the floor the curve is unreadable, since any
monotone-in-anything ranking slopes downward. The gap to random is the finding;
the gap to oracle is the headroom left on the table.

⚠️ The oracle defers by per-point CRPS, not by absolute error (8.4, corrected).
An error-ranked oracle bounds MAE, not the metric plotted here, so the "mine"
curves could cross it, which reads as a bug and is tedious to explain.

⚠️ Deferred points are credited with the exact solver value, contributing zero
error. That assumes the fallback always succeeds, which 8.6 leaves open as a
stretch item: if VMEC++ fails at a meaningful rate in the compact region, this
curve is optimistic.

    python3 scripts/deferral.py
"""

import numpy as np
import pandas as pd

from constellaration_uq.metrics import crps_gaussian
from constellaration_uq.results import RESULTS_DIR, save_results, save_table

SEED = 0

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


def curve(crps, signal, rates):
    """Mean CRPS over all points when the top fraction by `signal` is deferred.

    Deferred points contribute zero, so this is the score of the hybrid system,
    surrogate plus solver, not of the surrogate on what is left. Dividing by the
    full count rather than the retained count is what makes the ends of the
    curve mean something: at rate 1 every point is solved and the score is 0.

    One cumulative sum serves every rate, so the whole curve costs one sort.
    """
    order = np.argsort(-np.asarray(signal), kind='stable')
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(crps)[order])])
    n = len(crps)
    total = cumulative[-1]
    return np.array([(total - cumulative[round(r * n)]) / n for r in rates])


def random_curve(crps, rates, seed):
    """The floor: defer at the same rate, choose at random."""
    rng = np.random.default_rng(seed)
    runs = [curve(crps, rng.permutation(len(crps)), rates) for _ in range(RANDOM_REPEATS)]
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


def run_region(df):
    """All four curves for one evaluation set."""
    crps = crps_gaussian(df['y_true'], df['mean'], df['total_variance'])

    curves = {
        'epistemic': curve(crps, df['epistemic_variance'], RATES),
        'total': curve(crps, df['total_variance'], RATES),
        'random': random_curve(crps, RATES, SEED),
        # Ranking by the score itself is the best any ranking can do, which is
        # exactly what makes it the ceiling.
        'oracle': curve(crps, crps, RATES),
    }

    summary = {
        name: {
            'crps_at': {f'{r:g}': float(np.interp(r, RATES, values)) for r in REPORT_AT},
            'rate_to_halve': rate_to_halve(RATES, values),
            # Area under the curve. One number for the whole trade, lower is
            # better, and the only fair way to compare two rankings that cross.
            'auc': float(np.trapz(values, RATES)),
        }
        for name, values in curves.items()
    }
    return curves, summary


def print_region(label, summary):
    print(f'\n{label}, CRPS of the hybrid system at a given deferral rate')
    header = f'{"ranking":>10s} ' + ' '.join(f'{r:>9.0%}' for r in REPORT_AT)
    header += f' {"auc":>9s} {"halve at":>9s}'
    print(header)
    print('-' * len(header))
    for name in ('epistemic', 'total', 'random', 'oracle'):
        row = summary[name]
        cells = ' '.join(f'{row["crps_at"][f"{r:g}"]:9.5f}' for r in REPORT_AT)
        halve = row['rate_to_halve']
        halve_text = f'{halve:9.1%}' if halve is not None else f'{"never":>9s}'
        print(f'{name:>10s} {cells} {row["auc"]:9.5f}{halve_text}')


def main():
    df = pd.read_csv(RESULTS_DIR / f'mv_ensemble_{SPLIT}_points.csv')

    payload, rows = {}, []
    for region, label in REGIONS.items():
        subset = df[df['region'] == region]
        curves, summary = run_region(subset)

        payload[region] = {'n': len(subset), 'rankings': summary}
        print_region(f'{label}, n = {len(subset):,}', summary)

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
                        'crps': float(value),
                    }
                )

    out = payload['out']['rankings']
    print('\nthe headline, out of region at the compact edge')
    for name in ('epistemic', 'total'):
        no_deferral = out[name]['crps_at']['0']
        at_20 = out[name]['crps_at']['0.2']
        print(
            f'  {name:>10s}-ranked: solving the worst 20% cuts CRPS '
            f'{no_deferral:.5f} to {at_20:.5f}, {1 - at_20 / no_deferral:.0%} lower'
        )
    random_20 = out['random']['crps_at']['0.2']
    oracle_20 = out['oracle']['crps_at']['0.2']
    print(f'  {"random":>10s} at 20%: {random_20:.5f}   {"oracle":>6s}: {oracle_20:.5f}')
    print('\nRead the gap to random as the result and the gap to oracle as the headroom.')

    constants = {
        'SPLIT': SPLIT,
        'SEED': SEED,
        'RATES': [float(r) for r in RATES],
        'RANDOM_REPEATS': RANDOM_REPEATS,
        'y_axis': 'CRPS of the hybrid system, physical units, deferred points credited zero',
        'x_axis': 'fraction of solver calls saved, per 8.2',
        'source': f'results/mv_ensemble_{SPLIT}_points.csv',
    }
    save_results('deferral', payload, constants=constants)
    save_table('deferral_curve', rows)
    print('\nsaved results/deferral.json and deferral_curve.csv')


if __name__ == '__main__':
    main()
