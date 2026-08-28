"""Tests for the results saving helper.

Low logic, so the coverage is deliberately thin. The one thing worth testing
properly is numpy serialization, because payloads come straight out of numpy
and a plain json.dump raises TypeError on the first np.float64 it meets.
"""

import numpy as np
import pytest

from constellaration_uq.results import load_results, save_results, save_table


def test_round_trip(tmp_path):
    payload = {'ratio': 3.02, 'verdict': 'PASS'}
    save_results('demo', payload, results_dir=tmp_path)

    document = load_results('demo', results_dir=tmp_path)
    assert document['results'] == payload
    assert document['name'] == 'demo'


def test_numpy_types_survive(tmp_path):
    payload = {
        'float': np.float64(0.001834),
        'int': np.int64(276),
        'bool': np.bool_(True),
        'array': np.arange(3, dtype=float),
    }
    save_results('numpy', payload, results_dir=tmp_path)

    got = load_results('numpy', results_dir=tmp_path)['results']
    assert got['float'] == pytest.approx(0.001834)
    assert got['int'] == 276
    assert got['bool'] is True
    assert got['array'] == [0.0, 1.0, 2.0]


def test_unserializable_type_still_raises(tmp_path):
    """The converter must not silently swallow genuinely bad payloads."""
    with pytest.raises(TypeError):
        save_results('bad', {'obj': object()}, results_dir=tmp_path)


def test_metadata_is_recorded(tmp_path):
    save_results('meta', {'x': 1}, constants={'SEED': 0}, results_dir=tmp_path)
    document = load_results('meta', results_dir=tmp_path)

    assert document['constants'] == {'SEED': 0}
    assert document['saved_at'].endswith('+00:00')
    assert 'git_commit' in document
    assert isinstance(document['git_dirty'], bool)


def test_rerun_overwrites_rather_than_accumulates(tmp_path):
    save_results('once', {'value': 1}, results_dir=tmp_path)
    save_results('once', {'value': 2}, results_dir=tmp_path)

    assert len(list(tmp_path.glob('once*.json'))) == 1
    assert load_results('once', results_dir=tmp_path)['results']['value'] == 2


def test_save_table_writes_csv(tmp_path):
    rows = [
        {'axis': 'aspect_ratio', 'cut': 'tail_low', 'ratio': 3.02},
        {'axis': 'aspect_ratio', 'cut': 'hole', 'ratio': 0.95},
    ]
    path = save_table('grid', rows, results_dir=tmp_path)

    lines = path.read_text().strip().splitlines()
    assert lines[0] == 'axis,cut,ratio'
    assert len(lines) == 3


def test_save_table_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        save_table('empty', [], results_dir=tmp_path)
