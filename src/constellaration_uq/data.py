# Load the `default` subset, filter to NFP=3 / DESC-VMEC pathways,
# flatten boundary.r_cos and boundary.z_sin to the 80 input columns,
# apply the 0.05% target-only tail trim.

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

ERROR_FLAG_COLUMNS = [
    'misc.has_optimize_boundary_omnigenity_vmec_error',
    'misc.has_optimize_boundary_omnigenity_desc_error',
    'misc.has_generate_qp_initialization_from_targets_error',
    'misc.has_generate_nae_initialization_from_targets_error',
    'misc.has_neurips_2025_forward_model_error',
]


def load_raw(data_dir: Path) -> pd.DataFrame:
    file_paths = sorted(Path(data_dir).glob('train-*.parquet'))
    frames = [pd.read_parquet(path) for path in file_paths]
    return pd.concat(frames, ignore_index=True)


def filter_valid(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    yield 'raw', df

    has_error = df[ERROR_FLAG_COLUMNS].fillna(False).any(axis=1)
    df = df[~has_error]
    yield 'error_filter', df

    df = df[df['boundary.n_field_periods'] == 3]
    yield 'nfp_3', df

    has_pathway = (
        df['desc_omnigenous_field_optimization_settings.id'].notna()
        | df['vmec_omnigenous_field_optimization_settings.id'].notna()
    )
    df = df[has_pathway]
    yield 'pathway_filter', df

    has_boundary = df['boundary.r_cos'].notna() & df['boundary.z_sin'].notna()
    df = df[has_boundary]
    yield 'null_boundary_guard', df


def trim_target_tails(
    df: pd.DataFrame, target_column: str, tail_fraction: float = 0.0005
) -> pd.DataFrame:
    """Drop the extreme tails of one target column.

    ⚠️ Reads labels, and does so before any split exists. On a random split that
    is harmless. On this project's splits it is not: target and split axis are
    correlated, so the dropped rows concentrate in the held-out region. 22 of
    the 28 dropped rows land in the tail split's held-out set. The cost is
    measured rather than assumed, see `mv_ensemble.py --sensitivity` and
    decision log 1.4; it is about 2% on the tail headline, in the flattering
    direction.

    ⚠️ NaN targets would vanish here silently. `between` is False for NaN, so a
    NaN row is dropped by this function and counted against the tail fraction,
    which makes the trim quietly remove more than it claims to. There are none
    in the current pool, so this is a guard against a future target column
    rather than a fix for an observed bug.
    """
    missing = int(df[target_column].isna().sum())
    if missing:
        raise ValueError(
            f'{target_column} has {missing} NaN values. `between` silently drops them, '
            'so they would be folded into the tail trim rather than counted. '
            'Filter them explicitly before trimming.'
        )

    lower, upper = df[target_column].quantile([tail_fraction, 1 - tail_fraction])
    return df[df[target_column].between(lower, upper)]


def extract_input_features(df: pd.DataFrame) -> np.ndarray:
    r_cos = np.array([row.tolist() for row in df['boundary.r_cos']]).reshape(len(df), -1)[:, 5:]
    z_sin = np.array([row.tolist() for row in df['boundary.z_sin']]).reshape(len(df), -1)[:, 5:]
    return np.concatenate([r_cos, z_sin], axis=1)


# The 80 columns are two surfaces of 40. Each 40 is the (5, 9) grid of poloidal
# mode m = 0..4 by toroidal mode n = -4..4, raveled row-major, with the first
# five entries dropped by the slice above: four are exactly zero under
# stellarator symmetry and the fifth is R(0,0), fixed by convention.
N_TOROIDAL_MODES = 9
N_DROPPED_FROM_M0 = 5
N_PER_SURFACE = 40


def poloidal_mode_columns(m: int) -> np.ndarray:
    """Column indices in the 80-vector belonging to one poloidal mode number.

    ⚠️ The dropped five shift every later block, so an m block does NOT sit at
    m * 9 in the flattened vector. m = 4 is columns 31 to 39 and 71 to 79, not
    the last 18 of the 80. Taking the last 18 would grab one surface's m = 3
    and m = 4 and none of the other's, which runs fine and produces plausible
    numbers, so this is worth a function and a test rather than a slice written
    inline at the call site.
    """
    if not 0 <= m < 5:
        raise ValueError(f'poloidal mode must be 0 to 4, got {m}')

    if m == 0:
        within = np.arange(N_TOROIDAL_MODES - N_DROPPED_FROM_M0)
    else:
        start = m * N_TOROIDAL_MODES - N_DROPPED_FROM_M0
        within = np.arange(start, start + N_TOROIDAL_MODES)

    return np.concatenate([within, within + N_PER_SURFACE])


def toroidal_mode_columns(n: int) -> np.ndarray:
    """Column indices in the 80-vector belonging to one toroidal mode number.

    n runs -4 to 4, so this is a column of the (5, 9) grid rather than a row,
    and it is scattered through the flattened vector instead of contiguous.
    The m = 0 row contributes only for n >= 1, since its n <= 0 entries are the
    five that were dropped.
    """
    if not -4 <= n <= 4:
        raise ValueError(f'toroidal mode must be -4 to 4, got {n}')

    offset = n + 4  # position of this n within a row of nine
    within = [
        m * N_TOROIDAL_MODES + offset - N_DROPPED_FROM_M0
        for m in range(5)
        if m > 0 or offset >= N_DROPPED_FROM_M0
    ]
    within = np.array(within)

    return np.concatenate([within, within + N_PER_SURFACE])


def mode_columns_at_resolution(max_mode: int) -> np.ndarray:
    """Columns at the outer edge of the grid: m == max_mode or |n| == max_mode.

    Dropping these hands the model a lower-resolution description of the same
    shape, resolution max_mode minus one instead of max_mode, rather than a
    hand-picked subset of coefficients. That is both easier to defend and
    closer to what "hide the high mode numbers" actually means.
    """
    blocks = [poloidal_mode_columns(max_mode)]
    blocks += [toroidal_mode_columns(n) for n in (-max_mode, max_mode)]
    return np.unique(np.concatenate(blocks))


def drop_columns(X: np.ndarray, dropped: np.ndarray) -> np.ndarray:
    """Return X without the given columns, others in their original order."""
    kept = np.setdiff1d(np.arange(X.shape[1]), dropped)
    return X[:, kept]


def drop_poloidal_modes(X: np.ndarray, modes) -> tuple[np.ndarray, np.ndarray]:
    """Hide whole poloidal mode blocks. Returns (reduced X, dropped columns).

    Used by the hidden-coefficient check, which manufactures a known amount of
    irreducible noise by withholding shape detail the target genuinely depends
    on, then asks whether the variance head recovers that magnitude.
    """
    dropped = np.unique(np.concatenate([poloidal_mode_columns(m) for m in modes]))
    return drop_columns(X, dropped), dropped


def load_dataset(data_dir: Path, target_column: str, verbose: bool = False) -> pd.DataFrame:
    raw = load_raw(data_dir)

    for step_name, filtered in filter_valid(raw):
        if verbose:
            print(f'{step_name:20s} {len(filtered):>7,}')

    return trim_target_tails(filtered, target_column)
