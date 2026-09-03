"""Does VMEC++ fail more often at the compact end? (decision log 8.6)

The deferral curve credits every deferred shape with the exact solver value, so
it assumes the fallback always succeeds. If VMEC++ fails at a higher rate in the
compact region, that assumption is weakest exactly where deferral sends the most
work, and the curve is optimistic where it matters most.

This is a statement about DEPLOYMENT, not about the measured curve. Every
shape in our pool solved successfully, by construction: the filter chain keeps
only converged rows. So the deferral experiment is not wrong on its own terms.
The risk is that an optimizer proposing new compact shapes meets a failure rate
the experiment never saw.

The obstacle, and why this script is more than a groupby. `metrics.aspect_ratio`
is computed FROM the solved equilibrium, so it is null on all 23,536 forward
model failures. We know which shapes failed and not how compact they were. Two
cheap geometric proxies were tried and rejected: 1/|r10| reaches Spearman 0.72
and 1/sqrt(|r10 z10|) reaches 0.83, which would scatter shapes across
neighbouring bins badly enough to hide a real effect or manufacture a fake one.

So aspect ratio is predicted from the 80 boundary coefficients, which gate 3.5
established is possible at R2 0.988. That gate asked whether the split axis is
input-measurable; this reuses the answer for a different purpose.

Both groups must be measured with the SAME instrument. Using the true aspect
ratio for successes and a prediction for failures would compare two populations
through two different lenses, and any difference in failure rate could then be
prediction error rather than physics. Every row here gets an out-of-fold
prediction: successes from a model that did not see them, failures from the
average of the same fold models. Nobody is scored in sample.

The aspect-ratio model is itself extrapolating onto the failures. It was fit
only on shapes that solved, and there is no label on a failure to check it
against, so its accuracy there is unmeasurable. This is the project's own
subject appearing one level down. The held-out R2 below bounds the instrument
on successes and nothing bounds it on failures.

    python3 scripts/solver_failures.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from constellaration_uq.baseline import DATA_DIR
from constellaration_uq.data import extract_input_features, load_raw, trim_target_tails
from constellaration_uq.metrics import distance_bins
from constellaration_uq.results import save_results, save_table

SEED = 0
N_FOLDS = 5
N_BINS = 10
MAX_ITER = 1000

AXIS_COL = 'metrics.aspect_ratio'
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'

# The outcome. Deliberately this flag alone rather than the five-flag chain: the
# question is whether the SOLVER failed on a shape that reached it, and the other
# four mark shapes that failed at generation and never reached VMEC++ at all.
SOLVER_FLAG = 'misc.has_neurips_2025_forward_model_error'

# Removed from the population rather than counted as failures, for the same
# reason. A shape that failed at generation was never a solver call.
GENERATION_FLAGS = [
    'misc.has_optimize_boundary_omnigenity_vmec_error',
    'misc.has_optimize_boundary_omnigenity_desc_error',
    'misc.has_generate_qp_initialization_from_targets_error',
    'misc.has_generate_nae_initialization_from_targets_error',
]

# The tail split holds out the lowest 20% of aspect ratio (day 1-2 grid).
TEST_FRACTION = 0.2


def load_population(data_dir=None):
    """The filter chain with the solver failures KEPT.

    Every other script in this project calls filter_valid, which drops all five
    error flags on the first step. This one needs the forward-model failures, so
    it repeats the remaining filters and keeps that flag as an outcome column.
    Counts print at every step because the whole result is a rate, and a rate is
    only as trustworthy as its denominator.
    """
    df = load_raw(data_dir or DATA_DIR)
    print(f'{"raw":26s} {len(df):>8,}')

    df = df[~df[GENERATION_FLAGS].fillna(False).any(axis=1)]
    print(f'{"generated without error":26s} {len(df):>8,}')

    df = df[df['boundary.n_field_periods'] == 3]
    print(f'{"nfp_3":26s} {len(df):>8,}')

    has_pathway = (
        df['desc_omnigenous_field_optimization_settings.id'].notna()
        | df['vmec_omnigenous_field_optimization_settings.id'].notna()
    )
    df = df[has_pathway]
    print(f'{"pathway_filter":26s} {len(df):>8,}')

    df = df[df['boundary.r_cos'].notna() & df['boundary.z_sin'].notna()]
    print(f'{"null_boundary_guard":26s} {len(df):>8,}')

    df = df.copy()
    df['failed'] = df[SOLVER_FLAG].fillna(False).astype(bool)
    return df


def out_of_fold_aspect(X, aspect, failed, seed=SEED):
    """An out-of-sample predicted aspect ratio for every row, failed or not.

    Successes are predicted by the one fold model that did not train on them.
    Failures have no fold to be held out of, so they are predicted by all
    N_FOLDS models and averaged, which is a mild ensemble and therefore slightly
    MORE accurate than the successes' single-model predictions. That direction is
    the safe one: it cannot manufacture extra scatter in the failure group and so
    cannot fake a flat failure-rate curve.
    """
    predicted = np.full(len(X), np.nan)
    success_idx = np.flatnonzero(~failed)
    failure_idx = np.flatnonzero(failed)
    failure_preds = []

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fit_pos, held_pos in folds.split(success_idx):
        fit_rows = success_idx[fit_pos]
        model = HistGradientBoostingRegressor(max_iter=MAX_ITER, random_state=seed).fit(
            X[fit_rows], aspect[fit_rows]
        )
        held_rows = success_idx[held_pos]
        predicted[held_rows] = model.predict(X[held_rows])
        failure_preds.append(model.predict(X[failure_idx]))

    predicted[failure_idx] = np.mean(failure_preds, axis=0)
    return predicted


def instrument_quality(aspect, predicted, failed):
    """How good the aspect-ratio predictor is, on the only rows that can say."""
    truth = aspect[~failed]
    guess = predicted[~failed]
    residual = truth - guess
    return {
        'n': len(truth),
        'rmse': float(np.sqrt(np.mean(residual**2))),
        'r2': float(1 - np.var(residual) / np.var(truth)),
        'std_of_aspect': float(np.std(truth)),
        'spearman': float(pd.Series(truth).corr(pd.Series(guess), method='spearman')),
    }


def binomial_stderr(rate, n):
    return float(np.sqrt(rate * (1 - rate) / n)) if n else float('nan')


def rate_in(mask, failed):
    n = int(mask.sum())
    if not n:
        return {'n': 0, 'n_failed': 0, 'rate': float('nan'), 'stderr': float('nan')}
    n_failed = int(failed[mask].sum())
    rate = n_failed / n
    return {'n': n, 'n_failed': n_failed, 'rate': rate, 'stderr': binomial_stderr(rate, n)}


def binned_rates(predicted, failed, n_bins=N_BINS):
    """Failure rate in equal-count bins of predicted aspect ratio.

    The bin geometry comes from metrics.distance_bins, which is named for the
    distance figures but is plain quantile binning of a 1-D array and is already
    tested. Equal-count for the usual reason: a rate needs a denominator, and
    equal-width bins would put the compact end, which is the whole question, in
    the thinnest bin.
    """
    geometry, masks = distance_bins(predicted, n_bins)
    return [
        {
            'bin': b,
            'a_lo': row['d_lo'],
            'a_hi': row['d_hi'],
            'a_median': row['d_median'],
            **rate_in(mask, failed),
        }
        for b, (row, mask) in enumerate(zip(geometry, masks, strict=True))
    ]


def pathway_labels(df):
    """Which generator proposed each shape, as 'desc' or 'vmec'.

    Read off which settings id is non-null, per 1.6. The pathway filter already
    guarantees at least one, and the dataset has exactly one on all but a single
    row of 182,222, so a shape carrying both would be a data surprise rather than
    a case to handle quietly.
    """
    desc = df['desc_omnigenous_field_optimization_settings.id'].notna().to_numpy()
    vmec = df['vmec_omnigenous_field_optimization_settings.id'].notna().to_numpy()
    if (desc & vmec).any():
        raise ValueError(f'{int((desc & vmec).sum())} rows claim both generation pathways')
    return np.where(desc, 'desc', 'vmec')


def pathway_confound(pathway, predicted, failed, cutoff):
    """Is the compact end failing because it is compact, or because of who made it?

    The obvious alternative explanation for the headline: if one generator both
    dominates the compact end and fails more everywhere, the 2.68x is about the
    generator and not about the geometry. The test is whether the below-cutoff
    elevation survives INSIDE each pathway, where the generator is held fixed by
    construction. If both pathways show it, geometry is doing the work.
    """
    rows = []
    for name in sorted(set(pathway)):
        mine = pathway == name
        below = rate_in(mine & (predicted <= cutoff), failed)
        above = rate_in(mine & (predicted > cutoff), failed)
        rows.append(
            {
                'pathway': name,
                'n': int(mine.sum()),
                'share_below_cutoff': float((mine & (predicted <= cutoff)).sum() / mine.sum()),
                'rate_below': below['rate'],
                'n_below': below['n'],
                'rate_above': above['rate'],
                'n_above': above['n'],
                'relative_risk': below['rate'] / above['rate'] if above['rate'] else float('nan'),
            }
        )
    return rows


def main():
    df = load_population()

    X = extract_input_features(df)
    aspect = df[AXIS_COL].to_numpy(dtype=float)
    failed = df['failed'].to_numpy()
    print(f'\nsolver failures {failed.sum():,} of {len(df):,} ({failed.mean():.2%})')
    print(f'aspect ratio present on failed rows: {np.isfinite(aspect[failed]).sum():,}')

    # The experiment's own cutoff, not a fresh quantile of this population. The
    # tail ensemble held out the lowest 20% of aspect ratio in the trimmed,
    # error-free pool, and that pool is exactly the successes here.
    successes = df[~df['failed']]
    cutoff = float(trim_target_tails(successes, TARGET_COL)[AXIS_COL].quantile(TEST_FRACTION))
    print(f'tail split cutoff, aspect ratio p{TEST_FRACTION:.0%}: {cutoff:.4f}')

    predicted = out_of_fold_aspect(X, aspect, failed)
    quality = instrument_quality(aspect, predicted, failed)
    print(
        f'\naspect-ratio predictor, out of fold on {quality["n"]:,} successes\n'
        f'  R2 {quality["r2"]:.4f}   RMSE {quality["rmse"]:.4f}   '
        f'spearman {quality["spearman"]:.4f}   (std of aspect {quality["std_of_aspect"]:.4f})'
    )

    rows = binned_rates(predicted, failed)
    header = f'{"bin":>4s} {"aspect range":>18s} {"n":>8s} {"failed":>8s} {"rate":>8s} {"± se":>7s}'
    print('\nsolver failure rate by predicted aspect ratio, compact end first')
    print(header)
    print('-' * len(header))
    for row in rows:
        span = f'{row["a_lo"]:.2f} to {row["a_hi"]:.2f}'
        print(
            f'{row["bin"]:4d} {span:>18s} {row["n"]:8,d} {row["n_failed"]:8,d} '
            f'{row["rate"]:8.2%} {row["stderr"]:7.2%}'
        )

    below = rate_in(predicted <= cutoff, failed)
    above = rate_in(predicted > cutoff, failed)
    ratio = below['rate'] / above['rate'] if above['rate'] else float('nan')
    print(
        f'\nthe comparison that matters, split at the tail cutoff\n'
        f'  below (the deferral region) {below["rate"]:.2%} ± {below["stderr"]:.2%} '
        f'on {below["n"]:,}\n'
        f'  above (the training region) {above["rate"]:.2%} ± {above["stderr"]:.2%} '
        f'on {above["n"]:,}\n'
        f'  relative risk {ratio:.2f}x'
    )
    pathway = pathway_labels(df)
    confound = pathway_confound(pathway, predicted, failed, cutoff)
    header = (
        f'{"pathway":>8s} {"n":>8s} {"% compact":>10s} {"rate below":>11s} '
        f'{"rate above":>11s} {"rel risk":>9s}'
    )
    print('\nconfound check: does the elevation survive inside each generator?')
    print(header)
    print('-' * len(header))
    for row in confound:
        print(
            f'{row["pathway"]:>8s} {row["n"]:8,d} {row["share_below_cutoff"]:10.1%} '
            f'{row["rate_below"]:11.2%} {row["rate_above"]:11.2%} {row["relative_risk"]:8.2f}x'
        )
    print(
        'Both pathways elevated below the cutoff means the geometry is doing the\n'
        'work. One pathway carrying it would mean the headline is about who\n'
        'proposed the shape rather than about how compact it is.'
    )

    print(
        '\nA flat curve leaves the deferral result standing as published. A rate\n'
        'that climbs toward the compact end makes it optimistic in deployment, by\n'
        'roughly the fraction of deferred solves that would fail.'
    )

    payload = {
        'population': {
            'n': len(df),
            'n_failed': int(failed.sum()),
            'failure_rate': float(failed.mean()),
            'aspect_present_on_failures': int(np.isfinite(aspect[failed]).sum()),
        },
        'instrument': quality,
        'cutoff': cutoff,
        'bins': rows,
        'below_cutoff': below,
        'above_cutoff': above,
        'relative_risk': ratio,
        'pathway_confound': confound,
    }
    save_results(
        'solver_failures',
        payload,
        constants={
            'SEED': SEED,
            'N_FOLDS': N_FOLDS,
            'N_BINS': N_BINS,
            'MAX_ITER': MAX_ITER,
            'SOLVER_FLAG': SOLVER_FLAG,
            'GENERATION_FLAGS': GENERATION_FLAGS,
            'TEST_FRACTION': TEST_FRACTION,
            'note': 'aspect ratio is null on every solver failure and is predicted '
            'out of fold from the 80 boundary coefficients for both groups',
        },
    )
    save_table('solver_failures_bins', rows)
    print('\nsaved results/solver_failures.json and solver_failures_bins.csv')


if __name__ == '__main__':
    main()
