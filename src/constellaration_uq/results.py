"""Persist analysis results as JSON, with enough metadata to trust them later.

Every script that produces numbers writes one file here. Two reasons. The day
1-2 grid takes 746s, so re-plotting from stdout means re-running it. And the
decision log currently holds hand-transcribed tables, which is one typo away
from a wrong claim in the write-up.

Fixed filenames, overwritten on rerun. Git history is the version history.

The git commit and dirty flag are the point of the metadata: a result produced
from uncommitted code cannot be reproduced, and you will not remember which
run a stale file came from.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results'


def _git(*args):
    """Run a git command, returning None if git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ['git', *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def _json_default(obj):
    """numpy scalars and arrays are not JSON serializable on their own.

    Payloads come straight out of numpy computations, so without this every
    save would raise TypeError on the first np.float64 it met.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'not JSON serializable: {type(obj).__name__}')


def save_results(name, payload, constants=None, results_dir=None):
    """Write results/{name}.json with provenance metadata. Returns the path.

    `constants` is for the script's configuration (seed, bin counts, thresholds).
    Recording it separately from the payload makes it obvious what settings
    produced these numbers.
    """
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    document = {
        'name': name,
        'saved_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain')),
        'constants': constants or {},
        'results': payload,
    }

    path = results_dir / f'{name}.json'
    with path.open('w') as f:
        json.dump(document, f, indent=2, default=_json_default)
        f.write('\n')
    return path


def load_results(name, results_dir=None):
    """Read back a saved document. Plotting scripts use this instead of recomputing."""
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    with (results_dir / f'{name}.json').open() as f:
        return json.load(f)


def save_table(name, rows, results_dir=None):
    """Write results/{name}.csv from a list of dicts. Returns the path.

    For results that are naturally tabular, like the day 1-2 grid, so they can
    go straight into a plot or a spreadsheet without unpacking JSON.
    """
    import csv

    if not rows:
        raise ValueError('save_table needs at least one row')

    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    path = results_dir / f'{name}.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
