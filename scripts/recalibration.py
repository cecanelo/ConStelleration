"""Optional scope 9.4: does a standard in-region correction transfer out of region?

The obvious objection to "the surrogate is overconfident at the compact edge" is
"so recalibrate it". This answers with a measurement rather than an argument.

**Variance scaling**, one scalar per split, `sigma' = s * sigma`. Under Gaussian
NLL the minimiser is closed form, `s = sqrt(mean(z^2))` with `z = (y - mu) /
sigma`, so there is no optimiser and no new dependency. Two richer methods were
considered and rejected in 9.4: isotonic recalibration of the CDF, which makes
the predictive distribution non-Gaussian and retires the closed-form CRPS the
deferral curve is plotted in, and conformal prediction, whose coverage guarantee
assumes exchangeability between the calibration and test sets, which is exactly
what a tail split breaks. One parameter is also the right expressiveness for the
claim: the question is whether an in-region correction TRANSFERS, and a richer
corrector makes the answer about which corrector was chosen.

**Reporting only. This never feeds the deferral rule** (7.4), so the two
results stay independent.

**The calibration set is in-region and is not the set the numbers are
reported on.** Fitting `s` on the in-region slice and then reporting the
corrected ratio on those same rows is circular. The slice is split in half:
`s` is fitted on one half and every reported number comes from the other half
and from the untouched out-of-region set. The early-stopping validation set is
not a substitute, since training already saw it through the stopping rule.

**`s` is not the reciprocal of the reported calibration ratio.** The table's
ratio is sqrt(mean(sigma^2)) / sqrt(mean(e^2)); this scalar is
sqrt(mean(e^2 / sigma^2)). They coincide only when sigma is constant across
points. Near, not equal, and neither is a check on the other.

**A global scalar cannot change the deferral ranking**, since sorting by
s^2 * sigma^2 is the same order as sorting by sigma^2. Both "mine" curves are
identical after recalibration and only the CRPS level moves, so the deferral
curve is not rerun. The invariance is asserted here and tested in check_ranking.

Prediction committed before this ran (9.4): s below 1, near 0.85 to 0.90, so the
correction shrinks intervals and pushes the out-of-region ratio from 0.677
toward 0.60. The standard fix should make the off-distribution problem worse.

    python3 scripts/recalibration.py
"""

import numpy as np
import pandas as pd

from constellaration_uq.metrics import coverage, crps_gaussian, pit_values, rms_uncertainty
from constellaration_uq.results import RESULTS_DIR, save_results, save_table

SEED = 0
SPLITS = ['random', 'hole', 'tail']
COVERAGE_LEVEL = 0.9

# Half the in-region slice fits the scalar, half reports. An even split because
# neither job is more important than the other: a scalar fitted on too few points
# is noisy, and a corrected ratio reported on too few points is not a measurement.
FIT_FRACTION = 0.5


def fit_scale(y_true, mean, variance):
    """The Gaussian-NLL optimal variance scale, closed form.

    Minimising sum(log(s * sigma_i) + z_i^2 / (2 s^2)) over s gives
    s^2 = mean(z^2). A perfectly calibrated set returns 1; an over-dispersed one,
    where the intervals are wider than the errors need, returns less than 1 and
    the correction shrinks them.
    """
    z = (np.asarray(y_true) - np.asarray(mean)) / np.sqrt(np.asarray(variance))
    return float(np.sqrt(np.mean(z**2)))


def score(df, scale):
    """Every reported quantity, with the predictive variance scaled by `scale`.

    scale = 1.0 reproduces the uncorrected numbers exactly, which is how the
    before column in the output is produced: the same code path, not a second
    implementation that could drift from calibration.py.
    """
    variance = df['total_variance'].to_numpy() * scale**2
    error = df['y_true'].to_numpy() - df['mean'].to_numpy()
    rmse = float(np.sqrt(np.mean(error**2)))
    total = rms_uncertainty(variance)
    return {
        'n': len(df),
        'rmse': rmse,
        'total': total,
        'ratio': total / rmse,
        'coverage': coverage(df['y_true'], df['mean'], variance, COVERAGE_LEVEL),
        'pit_mean': float(pit_values(df['y_true'], df['mean'], variance).mean()),
        'crps': float(np.mean(crps_gaussian(df['y_true'], df['mean'], variance))),
    }


def check_ranking(df, scale):
    """The deferral ranking must be untouched by a global scale. Verify, do not assume.

    Cheap, and it is the one claim in 9.4 that is asserted rather than measured.
    A scale that somehow reordered points would silently invalidate the decision
    not to rerun the deferral curve.
    """
    base = np.argsort(-df['total_variance'].to_numpy(), kind='stable')
    scaled = np.argsort(-(df['total_variance'].to_numpy() * scale**2), kind='stable')
    return bool(np.array_equal(base, scaled))


def split_in_region(df, seed=SEED):
    """Halve the in-region slice into a calibration half and a reporting half."""
    in_region = df[df['region'] == 'in']
    shuffled = np.random.default_rng(seed).permutation(len(in_region))
    cut = round(FIT_FRACTION * len(in_region))
    return in_region.iloc[shuffled[:cut]], in_region.iloc[shuffled[cut:]]


def run_split(name):
    df = pd.read_csv(RESULTS_DIR / f'mv_ensemble_{name}_points.csv')
    calibration, in_report = split_in_region(df)
    out_report = df[df['region'] == 'out']

    scale = fit_scale(calibration['y_true'], calibration['mean'], calibration['total_variance'])

    return {
        'scale': scale,
        'n_calibration': len(calibration),
        'ranking_preserved': check_ranking(out_report, scale),
        'in': {'before': score(in_report, 1.0), 'after': score(in_report, scale)},
        'out': {'before': score(out_report, 1.0), 'after': score(out_report, scale)},
    }


def print_split(name, block):
    print(f'\n{name}, scale fitted on {block["n_calibration"]:,} in-region points')
    print(
        f'  s = {block["scale"]:.4f}   ({"shrinks" if block["scale"] < 1 else "widens"} intervals)'
    )

    header = (
        f'  {"region":>7s} {"n":>6s} {"ratio":>16s} {"cov@0.9":>16s} '
        f'{"pit mean":>16s} {"crps":>18s}'
    )
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for region in ('in', 'out'):
        before, after = block[region]['before'], block[region]['after']
        print(
            f'  {region:>7s} {before["n"]:6,d} '
            f'{before["ratio"]:7.3f} -> {after["ratio"]:5.3f} '
            f'{before["coverage"]:7.3f} -> {after["coverage"]:5.3f} '
            f'{before["pit_mean"]:7.3f} -> {after["pit_mean"]:5.3f} '
            f'{before["crps"]:8.5f} -> {after["crps"]:7.5f}'
        )


def main():
    payload = {name: run_split(name) for name in SPLITS}

    for name, block in payload.items():
        print_split(name, block)

    if not all(block['ranking_preserved'] for block in payload.values()):
        raise SystemExit(
            'a global variance scale reordered the deferral ranking, which is '
            'impossible for a positive scalar. Something is wrong with the sort.'
        )
    print('\nranking preserved under scaling on every split, so the deferral curve stands')

    tail = payload['tail']
    print(
        f'\nthe question 9.4 was written to answer, on the tail split\n'
        f'  in region  the correction moves the ratio '
        f'{tail["in"]["before"]["ratio"]:.3f} -> {tail["in"]["after"]["ratio"]:.3f}\n'
        f'  out region it moves it '
        f'{tail["out"]["before"]["ratio"]:.3f} -> {tail["out"]["after"]["ratio"]:.3f}\n'
        f'  1.0 is honest. Read whether the out-of-region number moved toward it or away.'
    )

    rows = [
        {
            'split': name,
            'scale': block['scale'],
            'region': region,
            'when': when,
            **block[region][when],
        }
        for name, block in payload.items()
        for region in ('in', 'out')
        for when in ('before', 'after')
    ]

    save_results(
        'recalibration',
        payload,
        constants={
            'SEED': SEED,
            'SPLITS': SPLITS,
            'COVERAGE_LEVEL': COVERAGE_LEVEL,
            'FIT_FRACTION': FIT_FRACTION,
            'method': 'variance scaling, sigma -> s * sigma, s = sqrt(mean(z^2)) '
            'under Gaussian NLL, fitted on a held-out half of the in-region slice',
            'scope': 'reporting only, never feeds the deferral rule (7.4)',
        },
    )
    save_table('recalibration', rows)
    print('\nsaved results/recalibration.json and recalibration.csv')


if __name__ == '__main__':
    main()
