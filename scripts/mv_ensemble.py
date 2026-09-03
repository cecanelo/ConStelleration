"""Steps 2, 4 and 5: a mean-variance ensemble on one split.

One script, split name as an argument, because "same model and recipe for every
split" is the premise that makes the three results comparable at all (3.8).
Three scripts would be three places for that premise to quietly stop holding.

    python3 scripts/mv_ensemble.py random
    python3 scripts/mv_ensemble.py hole
    python3 scripts/mv_ensemble.py tail

Four disjoint sets come out of the pool, and keeping them straight is most of
what this script does:

    fit          trains the ten members
    validation   500 fixed points, early stopping only, never reported
    in-region    held out from the training region, the "normal accuracy" number
    out-region   the split's held-out set, near-identical to in-region for the
                 random split and the actual subject for hole and tail

Per-point predictions are written to CSV so the calibration figures and the
deferral curve can be built without refitting.
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from constellaration_uq.baseline import load_pool, rmse
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.ensemble import N_MEMBERS, decompose, train_mv_ensemble
from constellaration_uq.metrics import rms_uncertainty
from constellaration_uq.nets import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    VAL_SIZE,
    VARIANCE_FLOOR,
    WARMUP_EPOCHS,
    resolve_device,
    split_validation,
)
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
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
AXIS_COL = 'metrics.aspect_ratio'

# The plain MSE ensemble's held-out RMSE on the random split (4.6). The
# mean-variance model should land near it; a large gap means the variance head
# or the NLL loss cost accuracy, which is what step 1 exists to let us say.
MSE_ENSEMBLE_RMSE = 0.01052

# A member variance within this factor of the floor counts as pinned. The
# fraction of pinned predictions is the observable beta-NLL trigger from 4.4.
FLOOR_TOLERANCE = 1.01


@dataclass(frozen=True)
class Config:
    """One typed object threaded through the entry point (4.8). Architecture and
    training hyperparameters are deliberately not fields: they are frozen
    constants in nets.py, and making them configurable would reopen 4.7."""

    split: str
    seed: int = SEED
    target_col: str = TARGET_COL
    axis_col: str = AXIS_COL


def build_split(config, axis):
    if config.split == 'random':
        return random_split(axis, config.seed, TEST_FRACTION)
    if config.split == 'hole':
        return hole_split(axis, TEST_FRACTION, HOLE_CENTRE)
    if config.split == 'tail':
        return tail_split(axis, 'low', TEST_FRACTION)
    raise ValueError(f"split must be 'random', 'hole' or 'tail', got {config.split!r}")


def summarise(label, y_true, parts, selection):
    """Point accuracy and the uncertainty decomposition over one set of rows.

    Reported as standard deviations rather than variances, because that is the
    scale the target lives on and the only one a reader can compare against
    RMSE. The addition happened in decompose, in variance space, where it is
    valid.

    This used mean(sqrt(variance)) until 2026-08-30, biasing every reported
    calibration ratio low. metrics.rms_uncertainty carries the reasoning. Caught
    by scripts/calibration.py computing the same quantity a second way and
    disagreeing, and independently confirmed by in-region coverage reading 0.95
    to 0.97 against a nominal 0.9 on all three splits.
    """
    mean = parts['mean'][selection]

    return {
        'set': label,
        'n': int(selection.sum()) if selection.dtype == bool else len(selection),
        'rmse': rmse(y_true, mean),
        'epistemic': rms_uncertainty(parts['epistemic_variance'][selection]),
        'aleatoric': rms_uncertainty(parts['aleatoric_variance'][selection]),
        'total': rms_uncertainty(parts['total_variance'][selection]),
    }


def restored_rows(df, trimmed, config, axis, out_mask):
    """The rows the 0.05% target trim deleted from this split's held-out region.

    The trim reads held-out labels. `trim_target_tails` computes its quantiles
    over the whole pool and drops rows by their target value, before any split
    exists. Target and split axis are correlated, which is this project's own
    premise, so the deleted rows do not land evenly: 22 of 28 fall inside the
    tail's held-out region. The headline out-of-region RMSE is therefore
    computed on a test set whose hardest members were removed using the answers
    the experiment is supposed to be predicting.

    This function recovers them so the cost can be measured rather than
    estimated. It changes nothing about training: the ensemble is already fitted
    when this is called, and the trimmed evaluation is reported unchanged
    alongside.

    Returns (X_extra, y_extra, axis_extra, is_sign_flipped), split out because
    the restored rows are two different populations and averaging them into one
    number would be misleading. Roughly a third carry NEGATIVE edge rotational
    transform, a sign class of 133 rows in 27,022. Those fail because the model
    has barely seen one anywhere in training, not because they sit at low aspect
    ratio, so folding them into the headline would conflate "extrapolating along
    the split axis is hard" with "a 0.5% physical class is unlearnable".
    """
    # The split was built on the trimmed axis, so recover its numeric boundary
    # and apply that same value to the untrimmed pool. Re-running the split on
    # untrimmed data would move the quantile and change which rows are held out.
    if not out_mask.any():
        raise ValueError('cannot recover a cutoff from an empty held-out set')
    lo, hi = axis[out_mask].min(), axis[out_mask].max()

    # A tail split is one-sided, so its lower bound is the pool minimum
    # rather than a real boundary. Keeping it would exclude any restored row
    # lying past that minimum, which is precisely the kind of row worth
    # recovering. A hole is genuinely two-sided and keeps both bounds.
    if config.split == 'tail':
        lo = -np.inf

    dropped = df.loc[df.index.difference(trimmed.index)]
    dropped_axis = dropped[config.axis_col].to_numpy()
    inside = (dropped_axis >= lo) & (dropped_axis <= hi)
    dropped = dropped[inside]

    y_extra = dropped[config.target_col].to_numpy()
    return (
        extract_input_features(dropped),
        y_extra,
        dropped_axis[inside],
        y_extra < 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('split', choices=['random', 'hole', 'tail'])
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help='replication seed. Varies member init, the in-region slice and the '
        'validation set. For hole and tail the out-of-region set is a quantile '
        'cutoff on the axis and does not move, so the headline test set is fixed.',
    )
    parser.add_argument(
        '--sensitivity',
        action='store_true',
        help='also score the rows the target trim deleted from the held-out region',
    )
    args = parser.parse_args()
    config = Config(split=args.split, seed=args.seed)

    started = time.time()
    print(f'device: {resolve_device()}   split: {config.split}')

    # `df` is kept untrimmed so --sensitivity can recover the rows the trim
    # deleted. Training uses `trimmed` only, exactly as before.
    df = load_pool()
    trimmed = trim_target_tails(df, config.target_col)
    X = extract_input_features(trimmed)
    y = trimmed[config.target_col].to_numpy()
    axis = trimmed[config.axis_col].to_numpy()
    print(f'pool: {len(trimmed):,} rows after target trim')

    train_mask, out_mask = build_split(config, axis)

    # The in-region slice comes out first, then the validation set out of what
    # remains, so the three never overlap. Early stopping must not see the rows
    # the in-region number is computed on, and neither may see out-of-region
    # data (3.9), or the extrapolation condition leaks into training.
    train_idx = np.flatnonzero(train_mask)
    fit_pool, in_idx = train_test_split(
        train_idx, test_size=IN_REGION_TEST_FRACTION, random_state=config.seed
    )
    X_fit, y_fit, X_val, y_val = split_validation(X[fit_pool], y[fit_pool], VAL_SIZE, config.seed)
    out_idx = np.flatnonzero(out_mask)

    print(
        f'fit {len(X_fit):,}  validation {len(X_val):,}  '
        f'in-region {len(in_idx):,}  out-of-region {len(out_idx):,}\n'
        f'recipe: lr {LEARNING_RATE:g}, batch {BATCH_SIZE}, warmup {WARMUP_EPOCHS}, '
        f'cap {MAX_EPOCHS}, patience {PATIENCE}\n'
    )

    header = f'{"member":>7s} {"epochs":>7s} {"best_nll":>9s} {"time":>7s}'
    print(header)
    print('-' * len(header))
    previous = [time.time()]

    def report(k, epochs, best):
        now = time.time()
        print(f'{k:7d} {epochs:7d} {best:9.4f} {now - previous[0]:6.1f}s', flush=True)
        previous[0] = now

    # base_seed * N_MEMBERS, not base_seed. train_mv_ensemble uses
    # `seed = base_seed + k` for its ten members, so consecutive replication
    # seeds would share nine of ten member initialisations and the measured
    # spread would read far smaller than the real one. Blocks of N_MEMBERS keep
    # them disjoint: 0-9, 10-19, 20-29. Seed 0 is unchanged, since 0 * 10 = 0,
    # which is what lets the existing results stay byte-identical.
    predict_all, histories = train_mv_ensemble(
        X_fit, y_fit, X_val, y_val, base_seed=config.seed * N_MEMBERS, progress=report
    )

    eval_idx = np.concatenate([in_idx, out_idx])
    member_means, member_variances = predict_all(X[eval_idx])
    parts = decompose(member_means, member_variances)

    n_in = len(in_idx)
    rows = [
        summarise('in-region', y[in_idx], parts, np.arange(n_in)),
        summarise('out-of-region', y[out_idx], parts, np.arange(n_in, len(eval_idx))),
    ]

    print(
        f'\n{"set":>14s} {"n":>7s} {"rmse":>9s} {"epistemic":>10s} {"aleatoric":>10s} {"total":>9s}'
    )
    for row in rows:
        print(
            f'{row["set"]:>14s} {row["n"]:7,d} {row["rmse"]:9.5f} '
            f'{row["epistemic"]:10.5f} {row["aleatoric"]:10.5f} {row["total"]:9.5f}'
        )

    # The beta-NLL trigger from 4.4, made observable. A head pinned on its floor
    # is not measuring anything, it is saturated, and that is the symptom that
    # says warm-up plus floor was not enough.
    floor_physical = VARIANCE_FLOOR * y_fit.std() ** 2
    pinned = float((member_variances <= floor_physical * FLOOR_TOLERANCE).mean())
    print(f'\nvariance floor in physical units: {floor_physical:.3e}')
    print(f'member predictions pinned on it:  {pinned * 100:.2f}%')
    if pinned > 0.5:
        print('  Warning: over half pinned. This is the 4.4 trigger for beta-NLL.')

    if config.split == 'random':
        print(
            f'\nplain MSE ensemble was {MSE_ENSEMBLE_RMSE:.5f} on this split; '
            f'variance head costs {rows[0]["rmse"] / MSE_ENSEMBLE_RMSE - 1:+.1%} on in-region RMSE'
        )

    sensitivity = None
    if args.sensitivity:
        X_extra, y_extra, _, flipped = restored_rows(df, trimmed, config, axis, out_mask)
        extra_means, extra_variances = predict_all(X_extra)
        extra = decompose(extra_means, extra_variances)

        y_out = y[out_idx]
        mean_out = parts['mean'][n_in:]

        def combined(keep):
            """RMSE over the held-out set plus a chosen subset of restored rows.

            Pooled in squared error, not averaged over two RMSEs, since the two
            groups differ in size by two orders of magnitude.
            """
            errors = np.concatenate([y_out - mean_out, y_extra[keep] - extra['mean'][keep]])
            return float(np.sqrt(np.mean(errors**2))), len(errors)

        ordinary = ~flipped
        levels = [
            ('trimmed (headline)', float(rmse(y_out, mean_out)), len(y_out)),
            ('+ ordinary tail rows', *combined(ordinary)),
            ('+ sign-flipped rows', *combined(np.ones_like(flipped))),
        ]

        print(f'\n{"trim sensitivity":>22s} {"n":>7s} {"rmse_out":>9s} {"ratio":>7s}')
        for label, value, count in levels:
            print(f'{label:>22s} {count:7,d} {value:9.5f} {value / rows[0]["rmse"]:7.2f}')

        print(
            f'\n{len(y_extra)} rows restored, {int(flipped.sum())} of them sign-flipped '
            f'(negative target).\nThe sign class is 0.5% of the pool, so those rows measure '
            'unfamiliarity with a\nrare regime rather than distance along the split axis. '
            'The headline stays on the\ntrimmed set for A.4 comparability, and is '
            'conservative as a result.'
        )

        sensitivity = {
            'n_restored': len(y_extra),
            'n_sign_flipped': int(flipped.sum()),
            'levels': [
                {'set': label, 'n': count, 'rmse_out': value, 'ratio': value / rows[0]['rmse']}
                for label, value, count in levels
            ],
            'restored_targets': [round(float(v), 8) for v in y_extra],
        }

    # Distance measured against the rows the model saw, not against train_mask.
    # The mask holds the in-region slice, which would then read distance 0 by
    # construction rather than by measurement.
    #
    # Precisely, the reference is fit UNION validation, since fit_pool is the
    # index array before split_validation carves the 500 early-stopping rows out
    # of it. That is deliberate, the model did see those rows for stopping, and
    # they are a random draw from the training region so they barely move any
    # nearest-neighbour distance. The comment said "the fit set" until
    # 2026-08-31, which was imprecise rather than wrong.
    fit_mask = np.zeros(len(axis), dtype=bool)
    fit_mask[fit_pool] = True
    distance = distance_from_training_region(axis, fit_mask)[eval_idx]

    total_seconds = time.time() - started
    print(f'\ntotal {total_seconds:.1f}s')

    constants = {
        'SEED': config.seed,
        'SPLIT': config.split,
        'TEST_FRACTION': TEST_FRACTION,
        'IN_REGION_TEST_FRACTION': IN_REGION_TEST_FRACTION,
        'HOLE_CENTRE': HOLE_CENTRE,
        'TARGET_COL': config.target_col,
        'AXIS_COL': config.axis_col,
        'N_MEMBERS': N_MEMBERS,
        'VAL_SIZE': VAL_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'BATCH_SIZE': BATCH_SIZE,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'MAX_EPOCHS': MAX_EPOCHS,
        'PATIENCE': PATIENCE,
        'VARIANCE_FLOOR': VARIANCE_FLOOR,
        'architecture': '(256, 256, 256) tanh, Adam, MSE warmup then Gaussian NLL',
    }
    payload = {
        'sets': rows,
        'pinned_fraction': pinned,
        'floor_physical': floor_physical,
        'member_epochs': [len(h) for h in histories],
        'target_std': float(y.std()),
        'total_seconds': round(total_seconds, 1),
    }
    if sensitivity is not None:
        payload['trim_sensitivity'] = sensitivity

    # A separate name, so a sensitivity run can never overwrite the headline
    # files the calibration and deferral steps read. Seed 0 also keeps the
    # unsuffixed name, so replication runs are additive and nothing downstream
    # has to learn about seeds.
    name = f'mv_ensemble_{config.split}'
    if config.seed != SEED:
        name += f'_s{config.seed}'
    save_results(name + ('_sensitivity' if args.sensitivity else ''), payload, constants=constants)

    # Per-point rows so the calibration figures and the deferral curve need no
    # refit. Everything a Gaussian predictive distribution needs is here.
    save_table(
        name + ('_sensitivity' if args.sensitivity else '') + '_points',
        [
            {
                'region': 'in' if i < n_in else 'out',
                'distance': round(float(distance[i]), 6),
                'axis': round(float(axis[eval_idx[i]]), 6),
                'y_true': round(float(y[eval_idx[i]]), 8),
                'mean': round(float(parts['mean'][i]), 8),
                'epistemic_variance': float(parts['epistemic_variance'][i]),
                'aleatoric_variance': float(parts['aleatoric_variance'][i]),
                'total_variance': float(parts['total_variance'][i]),
            }
            for i in range(len(eval_idx))
        ],
    )
    written = name + ('_sensitivity' if args.sensitivity else '')
    print(f'saved results/{written}.json and _points.csv')


if __name__ == '__main__':
    main()
