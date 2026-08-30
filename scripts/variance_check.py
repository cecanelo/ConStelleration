"""Step 3: does the variance head measure anything? (4.4, 6.2)

The problem. On this dataset the world has no label noise: Stage 2 found
near-twin shapes whose targets differ by exactly 0.000000. So a variance head
that always reported approximately zero would look correct and would be
untestable, and every uncertainty number downstream would rest on an assertion.
Step 2 then reported aleatoric 0.01048, not zero, which is the model's own
misfit rather than noise in the world. Either way there is no ground truth to
check it against, so one has to be manufactured.

⚠️ The check is NOT "aleatoric went up". Anything makes a number go up. It is
whether aleatoric lands on a magnitude that was chosen in advance.

  1. Add Gaussian noise of a known standard deviation to the training targets.
  2. Train the same ensemble, same split, same frozen recipe.
  3. Aleatoric should come out at about sqrt(baseline squared + sigma squared).
  4. Repeat at several sigma so the answer is a curve, not one coincidence.

Evaluation is against CLEAN targets. The noise exists to give the head something
real to measure, and the mean should still be roughly as accurate as the
baseline because independent noise averages out across a training set.

---

Why not hide input coefficients, which is what decision 6.2 originally
specified. It was tried first and the instrument was too weak. Four ways of
hiding were priced by finding near-twin shapes in the reduced input space and
measuring how far apart their targets are, and every one injected about 0.003,
a predicted rise in aleatoric of 3 to 6 percent, which is inside seed noise:

    candidate      dropped  inputs  injected  predicted    rise
    m4                  18      62   0.00299    0.01090   +4.0%
    n4                  18      62   0.00338    0.01101   +5.1%
    resolution_3        32      48   0.00359    0.01108   +5.7%
    m3plus              36      44   0.00277    0.01084   +3.4%

Dropping 36 of 80 coefficients injected less than dropping 18. Two readings,
not separable from that table alone. The estimator may be biased low, because
pairs that are near-identical in the kept coordinates are probably also
near-identical in the dropped ones on this pool, where every shape came from
the same optimizers and the coefficients are correlated across modes. Or the
hidden coefficients genuinely carry little independent information about the
rotational transform, which is dominated by low-order structure.

Added noise avoids the question entirely: the injected magnitude is exact
rather than estimated, with no pair assumption and no root-two correction. The
cost is realism, since missing inputs mimic an unobserved confounder and added
noise does not. This check is about whether the head can measure at all, so the
exact instrument wins. Run with --modes to reproduce the table above.
"""

import argparse
import time
from itertools import pairwise

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from constellaration_uq.baseline import load_pool, rmse
from constellaration_uq.data import (
    drop_columns,
    extract_input_features,
    mode_columns_at_resolution,
    poloidal_mode_columns,
    toroidal_mode_columns,
    trim_target_tails,
)
from constellaration_uq.ensemble import N_MEMBERS, decompose, train_mv_ensemble
from constellaration_uq.metrics import rms_uncertainty
from constellaration_uq.nets import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    VAL_SIZE,
    WARMUP_EPOCHS,
    resolve_device,
    split_validation,
)
from constellaration_uq.results import load_results, save_results, save_table
from constellaration_uq.splits import random_split

SEED = 0
TEST_FRACTION = 0.2
IN_REGION_TEST_FRACTION = 0.2
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
AXIS_COL = 'metrics.aspect_ratio'

# Chosen to bracket the baseline aleatoric of 0.01048 rather than to be round.
# The target's own standard deviation is 0.0786, so these are 6%, 25% and 64%
# of it: one below the baseline, one clearly above, one dominant. A head that
# merely rescales something would fail the low level; a head that saturates
# would fail the high one.
NOISE_LEVELS = [0.005, 0.020, 0.050]

# Tighter than the band the hidden-coefficient version used, because sigma here
# is exact rather than estimated. Still a band and not a point: ten members at
# one seed is a noisy estimator of a mean predicted variance.
RECOVERY_BAND = (0.75, 1.35)

# Read from step 2 rather than hardcoded, so a rerun of that cannot leave this
# comparing against a stale baseline.
BASELINE_RESULTS = 'mv_ensemble_random'

# Kept only to reproduce the negative result recorded in the module docstring.
CANDIDATES = {
    'm4': lambda: poloidal_mode_columns(4),
    'n4': lambda: np.unique(np.concatenate([toroidal_mode_columns(-4), toroidal_mode_columns(4)])),
    'resolution_3': lambda: mode_columns_at_resolution(4),
    'm3plus': lambda: np.unique(
        np.concatenate([poloidal_mode_columns(3), poloidal_mode_columns(4)])
    ),
}
TWIN_PERCENTILE = 1.0


def measure_injected_noise(X_reduced, y, percentile=TWIN_PERCENTILE):
    """Estimate the scatter reduced inputs cannot explain, from near-twin pairs.

    For a near-twin pair the reduced model sees one input and two answers, so
    the difference between those answers is noise from its point of view. If
    both carry independent noise of standard deviation sigma, the expected
    squared difference is 2 sigma squared, hence the division by root two.

    ⚠️ Probably biased low on this pool, see the module docstring. Kept for the
    record, not used by the main check.
    """
    scaled = StandardScaler().fit_transform(X_reduced)

    # k=2 because the first neighbour of a point is itself.
    index = NearestNeighbors(n_neighbors=2, algorithm='brute').fit(scaled)
    distance, neighbour = index.kneighbors(scaled)
    distance, neighbour = distance[:, 1], neighbour[:, 1]

    cut = np.percentile(distance, percentile)
    twins = distance <= cut
    delta = np.abs(y[twins] - y[neighbour[twins]])

    return {
        'n_pairs': int(twins.sum()),
        'distance_cut': float(cut),
        'median_distance_all': float(np.median(distance)),
        'median_abs_delta': float(np.median(delta)),
        'injected': float(np.sqrt(np.mean(delta**2) / 2.0)),
    }


def load_everything():
    df = load_pool()
    trimmed = trim_target_tails(df, TARGET_COL)
    return (
        extract_input_features(trimmed),
        trimmed[TARGET_COL].to_numpy(),
        trimmed[AXIS_COL].to_numpy(),
        len(trimmed),
    )


def fit_indices(axis):
    """The same split and carving order as step 2, so the only thing that
    differs between the runs is the injected noise."""
    train_mask, _ = random_split(axis, SEED, TEST_FRACTION)
    train_idx = np.flatnonzero(train_mask)
    return train_test_split(train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=SEED)


def run_level(sigma, X, y, fit_pool, in_idx, baseline_aleatoric):
    """One ensemble trained on targets corrupted by noise of size sigma."""
    X_fit, y_fit, X_val, y_val = split_validation(X[fit_pool], y[fit_pool], VAL_SIZE, SEED)

    # Noise goes on both the fit and validation targets. Early stopping has to
    # see the same distribution the model is trained on, or it selects against
    # a cleaner signal than the loss is optimising.
    rng = np.random.default_rng(SEED)
    y_fit_noisy = y_fit + rng.normal(scale=sigma, size=len(y_fit))
    y_val_noisy = y_val + rng.normal(scale=sigma, size=len(y_val))

    expected = float(np.sqrt(baseline_aleatoric**2 + sigma**2))
    print(f'\nsigma {sigma:.4f}   aleatoric should come out near {expected:.5f}')
    header = f'{"member":>7s} {"epochs":>7s} {"best_nll":>9s} {"time":>7s}'
    print(header)
    print('-' * len(header))
    previous = [time.time()]

    def report(k, epochs, best):
        now = time.time()
        print(f'{k:7d} {epochs:7d} {best:9.4f} {now - previous[0]:6.1f}s', flush=True)
        previous[0] = now

    predict_all, histories = train_mv_ensemble(
        X_fit, y_fit_noisy, X_val, y_val_noisy, base_seed=SEED, progress=report
    )

    # Evaluated against CLEAN targets. Independent noise averages out of the
    # fitted mean, so RMSE should stay near the baseline while aleatoric rises.
    member_means, member_variances = predict_all(X[in_idx])
    parts = decompose(member_means, member_variances)

    # ⚠️ Aggregated by metrics.rms_uncertainty, which was mean(sqrt(variance))
    # inline here until 2026-08-30. The aggregation is not cosmetic in this
    # script: the entire check is that injected noise adds in VARIANCE, so
    # expected = sqrt(baseline^2 + sigma^2) is an identity about mean variances
    # and only the mean of variances can be compared to it. Averaging standard
    # deviations breaks the identity by an amount that changes with sigma, so
    # the bias does not even cancel between rows of the recovery table.
    achieved = rms_uncertainty(parts['aleatoric_variance'])
    ratio = achieved / expected
    lo, hi = RECOVERY_BAND

    return {
        'sigma': sigma,
        'expected_aleatoric': expected,
        'aleatoric': achieved,
        'ratio': ratio,
        'recovered': bool(lo <= ratio <= hi),
        'rmse_vs_clean': rmse(y[in_idx], parts['mean']),
        'epistemic': rms_uncertainty(parts['epistemic_variance']),
        'total': rms_uncertainty(parts['total_variance']),
        'member_epochs': [len(h) for h in histories],
    }


def measure_modes(baseline_aleatoric):
    """Reproduce the hidden-coefficient pricing table. No training."""
    X_full, y, axis, n_rows = load_everything()
    fit_pool, _ = fit_indices(axis)
    print(f'pool: {n_rows:,} rows after target trim, fit set {len(fit_pool):,}\n')

    header = (
        f'{"candidate":>14s} {"dropped":>8s} {"inputs":>7s} '
        f'{"injected":>9s} {"predicted":>10s} {"rise":>7s}'
    )
    print(header)
    print('-' * len(header))

    rows = []
    for name, columns_for in CANDIDATES.items():
        dropped = columns_for()
        X = drop_columns(X_full[fit_pool], dropped)
        noise = measure_injected_noise(X, y[fit_pool])

        predicted = float(np.sqrt(baseline_aleatoric**2 + noise['injected'] ** 2))
        rise = predicted / baseline_aleatoric - 1
        print(
            f'{name:>14s} {len(dropped):8d} {X.shape[1]:7d} {noise["injected"]:9.5f} '
            f'{predicted:10.5f} {rise:+7.1%}'
        )
        rows.append(
            {
                'candidate': name,
                'n_dropped': len(dropped),
                'n_inputs': int(X.shape[1]),
                'predicted_aleatoric': predicted,
                'rise': rise,
                **noise,
            }
        )

    print('\nEvery candidate injects far too little to clear seed noise, which is')
    print('why the main check uses added target noise instead. See the docstring.')
    save_results('variance_check_modes', {'baseline_aleatoric': baseline_aleatoric, 'rows': rows})
    print('saved results/variance_check_modes.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--modes',
        action='store_true',
        help='reproduce the hidden-coefficient pricing table and stop, no training',
    )
    args = parser.parse_args()

    started = time.time()
    print(f'device: {resolve_device()}')

    baseline = load_results(BASELINE_RESULTS)['results']
    baseline_in = next(row for row in baseline['sets'] if row['set'] == 'in-region')
    baseline_aleatoric = baseline_in['aleatoric']
    print(f'baseline from results/{BASELINE_RESULTS}.json, clean targets:')
    print(
        f'  rmse {baseline_in["rmse"]:.5f}  epistemic {baseline_in["epistemic"]:.5f}  '
        f'aleatoric {baseline_aleatoric:.5f}\n'
    )

    if args.modes:
        measure_modes(baseline_aleatoric)
        return

    X, y, axis, n_rows = load_everything()
    fit_pool, in_idx = fit_indices(axis)
    print(f'pool: {n_rows:,} rows after target trim')
    print(f'fit {len(fit_pool) - VAL_SIZE:,}  validation {VAL_SIZE}  in-region {len(in_idx):,}')
    print(f'noise levels: {NOISE_LEVELS}   target std {y.std():.5f}')

    rows = [run_level(s, X, y, fit_pool, in_idx, baseline_aleatoric) for s in NOISE_LEVELS]

    print(
        f'\n{"sigma":>8s} {"expected":>9s} {"measured":>9s} {"ratio":>7s} '
        f'{"rmse":>9s} {"epistemic":>10s} {"":>5s}'
    )
    print(
        f'{0.0:8.4f} {baseline_aleatoric:9.5f} {baseline_aleatoric:9.5f} {1.0:7.2f} '
        f'{baseline_in["rmse"]:9.5f} {baseline_in["epistemic"]:10.5f}  baseline'
    )
    for row in rows:
        print(
            f'{row["sigma"]:8.4f} {row["expected_aleatoric"]:9.5f} {row["aleatoric"]:9.5f} '
            f'{row["ratio"]:7.2f} {row["rmse_vs_clean"]:9.5f} {row["epistemic"]:10.5f}  '
            f'{"ok" if row["recovered"] else "MISS"}'
        )

    recovered = sum(row['recovered'] for row in rows)
    verdict = 'RECOVERS' if recovered == len(rows) else f'PARTIAL ({recovered}/{len(rows)})'
    print(f'\nverdict: {verdict}')
    print(f'a ratio inside {RECOVERY_BAND} counts as recovered')

    if all(a['aleatoric'] < b['aleatoric'] for a, b in pairwise(rows)):
        print('aleatoric is monotone in sigma, which rules out a head that reports a constant')

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')

    constants = {
        'SEED': SEED,
        'TEST_FRACTION': TEST_FRACTION,
        'IN_REGION_TEST_FRACTION': IN_REGION_TEST_FRACTION,
        'NOISE_LEVELS': NOISE_LEVELS,
        'RECOVERY_BAND': list(RECOVERY_BAND),
        'TARGET_COL': TARGET_COL,
        'AXIS_COL': AXIS_COL,
        'N_MEMBERS': N_MEMBERS,
        'VAL_SIZE': VAL_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'BATCH_SIZE': BATCH_SIZE,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'MAX_EPOCHS': MAX_EPOCHS,
        'PATIENCE': PATIENCE,
        'BASELINE_RESULTS': BASELINE_RESULTS,
    }
    payload = {
        'verdict': verdict,
        'baseline': baseline_in,
        'target_std': float(y.std()),
        'levels': rows,
        'total_seconds': round(total_seconds, 1),
    }

    save_results('variance_check', payload, constants=constants)
    save_table(
        'variance_check',
        [{k: v for k, v in row.items() if k != 'member_epochs'} for row in rows],
    )
    print('saved results/variance_check.json and .csv')


if __name__ == '__main__':
    main()
