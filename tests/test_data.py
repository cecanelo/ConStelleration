"""Regression tests for the 1.8 filter chain and the 1.5 boundary flatten.

Scope is deliberately narrow: the three functions in data.py that do real logic
and that can fail *silently*. load_raw and load_dataset are thin wiring and are
covered end to end by running the pipeline against the real parquet files.
"""

import numpy as np
import pandas as pd
import pytest

from constellaration_uq.data import (
    ERROR_FLAG_COLUMNS,
    extract_input_features,
    filter_valid,
    trim_target_tails,
)

TARGET = 'metrics.edge_rotational_transform_over_n_field_periods'
DESC_ID = 'desc_omnigenous_field_optimization_settings.id'
VMEC_ID = 'vmec_omnigenous_field_optimization_settings.id'

STEP_ORDER = ['raw', 'error_filter', 'nfp_3', 'pathway_filter', 'null_boundary_guard']


def nested_boundary(flat_values):
    """Build a boundary array shaped the way parquet actually returns it.

    Not a clean (5, 9) float block. The real column holds, per row, an
    object array of 5 entries, each entry its own 9-element float array.
    Building the fixture any other way would let the pre-fix version of
    extract_input_features pass these tests.
    """
    rows = np.asarray(flat_values, dtype=float).reshape(5, 9)
    nested = np.empty(5, dtype=object)
    for m in range(5):
        nested[m] = rows[m]
    return nested


def clean_row():
    """One row that survives every step of the 1.8 chain."""
    row = {flag: False for flag in ERROR_FLAG_COLUMNS}
    row['boundary.n_field_periods'] = 3.0
    row[DESC_ID] = 'desc-001'
    row[VMEC_ID] = None
    row['boundary.r_cos'] = nested_boundary(np.arange(45))
    row['boundary.z_sin'] = nested_boundary(np.arange(45) + 100)
    row[TARGET] = 0.25
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


def step_counts(df):
    return {name: len(step_df) for name, step_df in filter_valid(df)}


def test_clean_row_survives_every_step():
    counts = step_counts(frame(clean_row()))
    assert [counts[name] for name in STEP_ORDER] == [1, 1, 1, 1, 1]


@pytest.mark.parametrize(
    'mutation, rejected_at',
    [
        ({'misc.has_neurips_2025_forward_model_error': True}, 'error_filter'),
        ({'misc.has_optimize_boundary_omnigenity_desc_error': True}, 'error_filter'),
        ({'misc.has_generate_nae_initialization_from_targets_error': True}, 'error_filter'),
        ({'boundary.n_field_periods': 4.0}, 'nfp_3'),
        ({DESC_ID: None}, 'pathway_filter'),
        ({'boundary.r_cos': None}, 'null_boundary_guard'),
        ({'boundary.z_sin': None}, 'null_boundary_guard'),
    ],
)
def test_bad_row_is_rejected_at_the_expected_step(mutation, rejected_at):
    bad = clean_row() | mutation
    counts = step_counts(frame(clean_row(), bad))

    cut = STEP_ORDER.index(rejected_at)
    assert all(counts[name] == 2 for name in STEP_ORDER[:cut])
    assert all(counts[name] == 1 for name in STEP_ORDER[cut:])


def test_null_error_flags_count_as_no_error():
    """1.6: their loader does fillna(False), so a null flag means 'no error'."""
    row = clean_row()
    for flag in ERROR_FLAG_COLUMNS:
        row[flag] = None

    counts = step_counts(frame(row))
    assert counts['error_filter'] == 1


def test_vmec_pathway_is_accepted():
    """Step 3 keeps a row if *either* settings id is populated, not just desc."""
    row = clean_row() | {DESC_ID: None, VMEC_ID: 'vmec-001'}
    counts = step_counts(frame(row))
    assert counts['pathway_filter'] == 1


def test_null_boundary_guard_is_standalone():
    """1.6's trap row: fully null boundary while every error flag reads clean.

    Guards against the guard ever being folded into the error-flag filter.
    """
    trap = clean_row() | {'boundary.r_cos': None, 'boundary.z_sin': None}
    counts = step_counts(frame(clean_row(), trap))

    assert counts['error_filter'] == 2
    assert counts['null_boundary_guard'] == 1


def test_extract_input_features_shape_and_layout():
    """1.5: ravel each (5, 9) array, drop the first 5, r_cos then z_sin."""
    r_flat = np.arange(45, dtype=float)
    z_flat = np.arange(45, dtype=float) + 100.0

    row = clean_row()
    row['boundary.r_cos'] = nested_boundary(r_flat)
    row['boundary.z_sin'] = nested_boundary(z_flat)

    X = extract_input_features(frame(row))

    assert X.shape == (1, 80)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X[0, :40], r_flat[5:])
    np.testing.assert_array_equal(X[0, 40:], z_flat[5:])


def test_extract_input_features_drops_the_first_five():
    """The 4 symmetry zeros plus R(0,0) must not reach the model."""
    sentinel = 999.0
    r_flat = np.arange(45, dtype=float)
    r_flat[:5] = sentinel
    z_flat = np.arange(45, dtype=float) + 100.0
    z_flat[:5] = sentinel

    row = clean_row()
    row['boundary.r_cos'] = nested_boundary(r_flat)
    row['boundary.z_sin'] = nested_boundary(z_flat)

    X = extract_input_features(frame(row))

    # Shape first. Without it "no sentinel" is vacuously true on an empty array.
    assert X.shape == (1, 80)
    assert not (X == sentinel).any()


def test_extract_input_features_preserves_row_order():
    rows = []
    for i in range(3):
        row = clean_row()
        row['boundary.r_cos'] = nested_boundary(np.full(45, float(i)))
        row['boundary.z_sin'] = nested_boundary(np.full(45, float(i) + 0.5))
        rows.append(row)

    X = extract_input_features(frame(*rows))

    assert X.shape == (3, 80)
    for i in range(3):
        assert (X[i, :40] == float(i)).all()
        assert (X[i, 40:] == float(i) + 0.5).all()


def test_trim_target_tails_drops_both_extremes():
    df = pd.DataFrame({TARGET: range(100)})
    trimmed = trim_target_tails(df, TARGET, tail_fraction=0.1)

    assert len(trimmed) == 80
    assert trimmed[TARGET].min() == 10
    assert trimmed[TARGET].max() == 89


def test_trim_target_tails_ignores_other_columns():
    """1.4: the trim must never touch the split axis, only the target."""
    df = pd.DataFrame({TARGET: range(100), 'metrics.aspect_ratio': 5.0})
    df.loc[50, 'metrics.aspect_ratio'] = 1e6

    trimmed = trim_target_tails(df, TARGET, tail_fraction=0.1)

    assert 1e6 in trimmed['metrics.aspect_ratio'].values
