# Load the `default` subset, filter to NFP=3 / DESC-VMEC pathways,
# flatten boundary.r_cos and boundary.z_sin to the 80 input columns,
# apply the 0.05% target-only tail trim.

import pandas as pd 
import numpy as np 
from pathlib import Path 
from collections.abc import Iterator


ERROR_FLAG_COLUMNS = [
    "misc.has_optimize_boundary_omnigenity_vmec_error",
    "misc.has_optimize_boundary_omnigenity_desc_error",
    "misc.has_generate_qp_initialization_from_targets_error",
    "misc.has_generate_nae_initialization_from_targets_error",
    "misc.has_neurips_2025_forward_model_error",
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
        df["desc_omnigenous_field_optimization_settings.id"].notna()
        | df["vmec_omnigenous_field_optimization_settings.id"].notna()
    )
    df = df[has_pathway]
    yield 'pathway_filter', df

    has_boundary = df['boundary.r_cos'].notna() & df['boundary.z_sin'].notna()
    df = df[has_boundary]
    yield 'null_boundary_guard', df


def trim_target_tails(
    df: pd.DataFrame, target_column: str, tail_fraction: float = 0.0005
) -> pd.DataFrame:
    lower, upper = df[target_column].quantile([tail_fraction, 1 - tail_fraction])
    return df[df[target_column].between(lower, upper)]


def extract_input_features(df: pd.DataFrame) -> np.ndarray:
    r_cos = np.array([row.tolist() for row in df["boundary.r_cos"]]).reshape(len(df), -1)[:, 5:]
    z_sin = np.array([row.tolist() for row in df["boundary.z_sin"]]).reshape(len(df), -1)[:, 5:]
    return np.concatenate([r_cos, z_sin], axis=1)


def load_dataset(data_dir: Path, target_column: str, verbose: bool = False) -> pd.DataFrame:
    raw = load_raw(data_dir)

    for step_name, filtered in filter_valid(raw):
        if verbose:
            print(f"{step_name:20s} {len(filtered):>7,}")

    return trim_target_tails(filtered, target_column)
