"""Step 8: the N-sweep. Does the epistemic/aleatoric decomposition hold as data grows?

The last training run in the project, and the only one that is validation rather
than a deliverable. Steps 6 and 7 measured whether the uncertainty signal is
calibrated and whether it ranks well. Both can come out fine while the split into
two terms is meaningless, because a signal that ranks points usefully has said
nothing about which half of it is ignorance and which is noise.

The claim under test is what earns the two terms their names:

    epistemic is ignorance, so it must SHRINK as training data grows
    aleatoric is noise, so it must NOT

⚠️ On clean targets the second half is vacuous, and that is why this runs in two
noise conditions. 5.1 and 5.4 established that on clean targets the aleatoric
term is the model's own misfit, which *does* shrink with N, so "aleatoric stays
flat" would have no content. Adding Gaussian noise of σ = 0.020 to the training
targets installs a floor that genuinely should not move, and the contrast between
the conditions is the result: epistemic decaying toward a flat aleatoric floor.
It also makes the claim falsifiable to a number, "aleatoric stays at 0.020 as N
grows seventeenfold", rather than to a trend.

σ = 0.020 because 0.005 is too close to the baseline misfit to separate from it
and 0.050 dominates everything and hides the epistemic decay. It is 25% of the
target's spread and was recovered at ratio 1.07 in step 3.

Scope is pinned and deliberately narrow (see the N-sweep entry in CLAUDE.md):
the TAIL split only, since the random split has no out-of-region set to measure
"epistemic rises off-distribution" against, and the PRIMARY target only, since
the sweep validates the uncertainty machinery rather than any one metric.

    python3 scripts/n_sweep.py            # 30 ensembles, the full grid
    python3 scripts/n_sweep.py --reduced  # 12 ensembles, the documented floor

⚠️ Report the SHAPE of the decay, not a fitted exponent (6.3). At ten members
with a fixed architecture the curve is dominated by how fast members stop
disagreeing, which is not the clean statistical rate a fitted number implies.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from constellaration_uq.baseline import load_pool, rmse
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.ensemble import N_MEMBERS, decompose, train_mv_ensemble
from constellaration_uq.metrics import coverage, rms_uncertainty
from constellaration_uq.nets import VAL_SIZE, resolve_device, split_validation
from constellaration_uq.results import load_results, save_results, save_table
from constellaration_uq.splits import (
    distance_from_training_region,
    nested_subsamples,
    tail_split,
)

TEST_FRACTION = 0.2
IN_REGION_TEST_FRACTION = 0.2
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
AXIS_COL = 'metrics.aspect_ratio'

# None means the whole fit pool, whatever it turns out to be, so the top rung is
# never quietly a subsample.
SIZES = [1000, 2000, 4000, 8000, None]
SEEDS = [0, 1, 2]

# Clean, and a floor of known size. Both conditions or the aleatoric half of the
# claim is untestable.
NOISE_LEVELS = [0.0, 0.020]

# The documented fallback if the schedule slips: 3 sizes x 2 seeds x 2 noise
# conditions. The reporting commitment is the shape of the decay, and 12 points
# show a shape. Never zero: nothing else in the project tests whether the
# epistemic term is reducible with data, which is the only thing that earns it
# the name.
REDUCED_SIZES = [1000, 4000, None]
REDUCED_SEEDS = [0, 1]

COVERAGE_LEVEL = 0.9


@dataclass(frozen=True)
class Config:
    sizes: list
    seeds: list
    noise_levels: list

    @property
    def n_ensembles(self):
        return len(self.sizes) * len(self.seeds) * len(self.noise_levels)


def build_sets(X, y, axis, seed=0):
    """The four frozen sets, carved once and reused at every N.

    Returns (fit_pool_X, fit_pool_y, X_val, y_val, in_idx, out_idx, distance).

    ⚠️ The test sets and the validation set are frozen ACROSS the whole sweep.
    Only the subsample drawn from the fit pool varies. If the validation set
    changed with N, models would stop training at different points for a reason
    unrelated to data volume, which is the confound the fixed 500-point
    validation set in 4.7 exists to prevent. If the test sets changed, every
    comparison between rungs would be against a different yardstick.

    Note this uses seed 0 regardless of the sweep seed. The sweep's seeds vary
    which rows are subsampled, not which rows are held out.
    """
    train_mask, out_mask = tail_split(axis, 'low', TEST_FRACTION)

    train_idx = np.flatnonzero(train_mask)
    fit_pool_idx, in_idx = train_test_split(
        train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=seed
    )

    X_fit_pool, y_fit_pool, X_val, y_val = split_validation(
        X[fit_pool_idx], y[fit_pool_idx], VAL_SIZE, seed
    )

    # Distance measured against the fit pool, not train_mask, for the same
    # reason as everywhere else: train_mask holds the in-region slice, which
    # would then read distance zero by construction rather than by measurement.
    # Measured against the FULL pool, not each subsample, so the x axis does not
    # move between rungs.
    fit_mask = np.zeros(len(axis), dtype=bool)
    fit_mask[fit_pool_idx] = True
    distance = distance_from_training_region(axis, fit_mask)

    return X_fit_pool, y_fit_pool, X_val, y_val, np.flatnonzero(out_mask), in_idx, distance


def evaluate(label, y_true, parts, selection):
    """The decomposition plus accuracy over one held-out set."""
    mean = parts['mean'][selection]
    total_variance = parts['total_variance'][selection]
    return {
        'set': label,
        'n': len(selection),
        'rmse': rmse(y_true, mean),
        'epistemic': rms_uncertainty(parts['epistemic_variance'][selection]),
        'aleatoric': rms_uncertainty(parts['aleatoric_variance'][selection]),
        'total': rms_uncertainty(total_variance),
        'coverage': coverage(y_true, mean, total_variance, COVERAGE_LEVEL),
    }


def run_cell(X, y, sets, size, seed, sigma):
    """One ensemble: one (N, seed, noise) combination."""
    X_fit_pool, y_fit_pool, X_val, y_val, out_idx, in_idx, _ = sets

    n = len(X_fit_pool) if size is None else size
    (subset,) = nested_subsamples(len(X_fit_pool), [n], seed)
    X_fit, y_fit = X_fit_pool[subset], y_fit_pool[subset]

    # Noise goes on the fit and validation targets, never on the test ones.
    # Early stopping has to see the distribution the loss is optimising, and the
    # model is scored against clean truth so the injected floor is what the
    # variance head must report rather than what it must predict.
    if sigma > 0:
        rng = np.random.default_rng(1000 + seed)
        y_fit = y_fit + rng.normal(scale=sigma, size=len(y_fit))
        y_val = y_val + rng.normal(scale=sigma, size=len(y_val))

    predict_all, histories = train_mv_ensemble(X_fit, y_fit, X_val, y_val, base_seed=seed)

    eval_idx = np.concatenate([in_idx, out_idx])
    member_means, member_variances = predict_all(X[eval_idx])
    parts = decompose(member_means, member_variances)

    n_in = len(in_idx)
    return {
        'n_train': n,
        'seed': seed,
        'sigma': sigma,
        'in_region': evaluate('in-region', y[in_idx], parts, np.arange(n_in)),
        'out_region': evaluate('out-of-region', y[out_idx], parts, np.arange(n_in, len(eval_idx))),
        'mean_epochs': float(np.mean([len(h) for h in histories])),
    }


def print_header():
    header = (
        f'{"sigma":>6s} {"N":>7s} {"seed":>5s} '
        f'{"rmse_in":>8s} {"epi_in":>8s} {"ale_in":>8s} '
        f'{"rmse_out":>9s} {"epi_out":>8s} {"ale_out":>8s} {"time":>7s}'
    )
    print(header)
    print('-' * len(header))
    return header


def print_cell(row, elapsed):
    inside, outside = row['in_region'], row['out_region']
    print(
        f'{row["sigma"]:6.3f} {row["n_train"]:7,d} {row["seed"]:5d} '
        f'{inside["rmse"]:8.5f} {inside["epistemic"]:8.5f} {inside["aleatoric"]:8.5f} '
        f'{outside["rmse"]:9.5f} {outside["epistemic"]:8.5f} {outside["aleatoric"]:8.5f} '
        f'{elapsed:6.1f}s',
        flush=True,
    )


def summarise(rows, sizes, noise_levels):
    """Average over seeds, which is the table the shape is read off."""
    print('\naveraged over seeds, in-region')
    header = (
        f'{"sigma":>6s} {"N":>7s} {"rmse":>9s} {"epistemic":>10s} '
        f'{"aleatoric":>10s} {"total":>9s} {"cov":>6s}'
    )
    print(header)
    print('-' * len(header))

    averaged = []
    for sigma in noise_levels:
        for size in sizes:
            matching = [r for r in rows if r['sigma'] == sigma and _matches(r, size, rows)]
            if not matching:
                continue
            entry = {'sigma': sigma, 'n_train': matching[0]['n_train'], 'n_seeds': len(matching)}
            for region in ('in_region', 'out_region'):
                for key in ('rmse', 'epistemic', 'aleatoric', 'total', 'coverage'):
                    entry[f'{region}_{key}'] = float(np.mean([r[region][key] for r in matching]))
            averaged.append(entry)
            print(
                f'{sigma:6.3f} {entry["n_train"]:7,d} {entry["in_region_rmse"]:9.5f} '
                f'{entry["in_region_epistemic"]:10.5f} {entry["in_region_aleatoric"]:10.5f} '
                f'{entry["in_region_total"]:9.5f} {entry["in_region_coverage"]:6.3f}'
            )
    return averaged


def _matches(row, size, rows):
    """Rows for one rung, with None resolved to the largest N actually run."""
    if size is not None:
        return row['n_train'] == size
    return row['n_train'] == max(r['n_train'] for r in rows)


def report_shape(averaged, noise_levels):
    """The claim, stated as the two ratios that carry it."""
    print('\nthe claim: epistemic shrinks with N, aleatoric does not')
    for sigma in noise_levels:
        cells = sorted((c for c in averaged if c['sigma'] == sigma), key=lambda c: c['n_train'])
        if len(cells) < 2:
            continue
        first, last = cells[0], cells[-1]
        growth = last['n_train'] / first['n_train']
        epi = last['in_region_epistemic'] / first['in_region_epistemic']
        ale = last['in_region_aleatoric'] / first['in_region_aleatoric']
        print(f'  sigma {sigma:.3f}: N x{growth:.0f}   epistemic x{epi:.2f}   aleatoric x{ale:.2f}')
    print(
        '\nOn clean targets both terms should fall, since aleatoric there is the\n'
        "model's own misfit (5.1). At sigma = 0.020 epistemic should still fall\n"
        'while aleatoric holds near 0.020, and THAT contrast is the result.\n'
        'Report the shape, not a fitted exponent (6.3).'
    )


def load_completed(constants):
    """Cells already on disk from an interrupted run, or [] if there are none.

    ⚠️ Refuses to resume across a configuration change. The saved constants must
    match the current ones exactly, because the alternative failure is silent: a
    results file half produced under one recipe and half under another looks
    complete and is not comparable rung to rung, which is the single assumption
    the whole sweep rests on. A mismatch starts over rather than asking.

    Also refuses to resume a run that already finished, so rerunning the script
    on purpose does what it looks like it does.
    """
    try:
        document = load_results('n_sweep')
    except FileNotFoundError:
        return []

    if document.get('constants') != constants:
        print('⚠️  existing results/n_sweep.json was produced under different')
        print('   constants. Starting over rather than mixing two recipes.\n')
        return []

    results = document.get('results', {})
    if results.get('complete'):
        print('existing results/n_sweep.json is already complete, rerunning it all.\n')
        return []

    return results.get('cells', [])


def cell_key(sigma, n_train, seed):
    """Identity of one ensemble, used to skip work already on disk."""
    return (float(sigma), int(n_train), int(seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--reduced',
        action='store_true',
        help='the 12-ensemble floor instead of the full 30',
    )
    parser.add_argument(
        '--restart',
        action='store_true',
        help='ignore any partial results on disk and run every cell again',
    )
    args = parser.parse_args()

    config = Config(
        sizes=REDUCED_SIZES if args.reduced else SIZES,
        seeds=REDUCED_SEEDS if args.reduced else SEEDS,
        noise_levels=NOISE_LEVELS,
    )

    started = time.time()
    print(f'device: {resolve_device()}   {config.n_ensembles} ensembles, tail split')

    df = load_pool()
    trimmed = trim_target_tails(df, TARGET_COL)
    X = extract_input_features(trimmed)
    y = trimmed[TARGET_COL].to_numpy()
    axis = trimmed[AXIS_COL].to_numpy()

    sets = build_sets(X, y, axis)
    X_fit_pool, _, X_val, _, out_idx, in_idx, _ = sets
    print(
        f'pool {len(trimmed):,} rows   fit pool {len(X_fit_pool):,}   '
        f'validation {len(X_val):,}   in-region {len(in_idx):,}   '
        f'out-of-region {len(out_idx):,}\n'
    )

    constants = {
        'SIZES': [s if s is not None else 'full' for s in config.sizes],
        'SEEDS': config.seeds,
        'NOISE_LEVELS': config.noise_levels,
        'N_MEMBERS': N_MEMBERS,
        'VAL_SIZE': VAL_SIZE,
        'COVERAGE_LEVEL': COVERAGE_LEVEL,
        'TARGET_COL': TARGET_COL,
        'AXIS_COL': AXIS_COL,
        'split': 'tail-low',
        'reduced': args.reduced,
    }

    rows = [] if args.restart else load_completed(constants)
    done = {cell_key(r['sigma'], r['n_train'], r['seed']) for r in rows}
    if done:
        print(f'resuming: {len(done)} of {config.n_ensembles} ensembles already on disk\n')

    def checkpoint(rows, averaged=None, total_seconds=None):
        """Write what exists so far.

        ⚠️ Called after every cell, not once at the end. This run is 30
        ensembles and roughly 40 minutes, and the summary and reporting
        functions do not execute until the last one finishes. Without this, a
        typo in either would discard every trained ensemble, which is the one
        failure mode where the cost is measured in hours rather than seconds.
        Rewriting two small files 30 times is free by comparison.
        """
        save_results(
            'n_sweep',
            {
                'cells': rows,
                'averaged': averaged or [],
                'complete': total_seconds is not None,
                'total_seconds': round(total_seconds, 1) if total_seconds else None,
            },
            constants=constants,
        )
        save_table(
            'n_sweep',
            [
                {
                    'sigma': r['sigma'],
                    'n_train': r['n_train'],
                    'seed': r['seed'],
                    'mean_epochs': r['mean_epochs'],
                    **{f'in_{k}': v for k, v in r['in_region'].items() if k != 'set'},
                    **{f'out_{k}': v for k, v in r['out_region'].items() if k != 'set'},
                }
                for r in rows
            ],
        )

    full_n = len(X_fit_pool)
    print_header()
    for sigma in config.noise_levels:
        for size in config.sizes:
            for seed in config.seeds:
                if cell_key(sigma, size or full_n, seed) in done:
                    continue
                cell_started = time.time()
                row = run_cell(X, y, sets, size, seed, sigma)
                rows.append(row)
                print_cell(row, time.time() - cell_started)
                checkpoint(rows)

    # Sorted so a resumed run writes the same order as an uninterrupted one,
    # which keeps the CSV diffable and the JSON free of a record of when the
    # interruption happened.
    rows.sort(key=lambda r: (r['sigma'], r['n_train'], r['seed']))

    averaged = summarise(rows, config.sizes, config.noise_levels)
    report_shape(averaged, config.noise_levels)

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')

    checkpoint(rows, averaged, total_seconds)
    print('saved results/n_sweep.json and n_sweep.csv')


if __name__ == '__main__':
    main()
