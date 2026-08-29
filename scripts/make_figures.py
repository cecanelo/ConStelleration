"""Regenerate every figure into figures/, as PNG and PDF.

Two kinds of figure, and they differ in what they read.

Data-derived figures recompute from the parquet pool, which is cheap because
they only need one metric column, never the 80-coefficient flatten.

Results-derived figures read results/*.json and draw in under a second. This is
why plotting is separate from computation: the day 1-2 grid takes 746s, and
tweaking a figure must not cost a rerun.

Missing results are skipped with a message rather than crashing, so this is
runnable before every analysis script has been run.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from constellaration_uq.data import filter_valid, load_raw
from constellaration_uq.results import REPO_ROOT, load_results
from constellaration_uq.splits import (
    distance_from_training_region,
    hole_split,
    tail_split,
)

DATA_DIR = REPO_ROOT / 'data_raw' / 'data'
FIGURES_DIR = REPO_ROOT / 'figures'

AXIS = 'metrics.aspect_ratio'
TEST_FRACTION = 0.2

# One palette across every figure. Consistency matters more than the specific
# choice: the same colour must mean the same thing in all of them.
COLOR_TRAIN = '#9ecae1'
COLOR_TAIL = '#d94801'
COLOR_HOLE = '#2171b5'
COLOR_NEUTRAL = '#525252'

plt.rcParams.update(
    {
        'figure.figsize': (8, 4.5),
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.25,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 10,
    }
)


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(FIGURES_DIR / f'{name}.{suffix}')
    plt.close(fig)
    print(f'  wrote figures/{name}.png and .pdf')


def load_axis_values():
    """Only the split axis column is needed, never the 80-coefficient flatten."""
    for _, df in filter_valid(load_raw(DATA_DIR)):
        pass
    return df[AXIS].to_numpy()


def figure_split_design(axis):
    """Where the cuts fall on the sampling density. The explanatory figure."""
    _, tail_test = tail_split(axis, 'low', TEST_FRACTION)
    _, hole_test = hole_split(axis, TEST_FRACTION)

    fig, ax = plt.subplots()
    bins = np.linspace(axis.min(), axis.max(), 80)
    ax.hist(axis, bins=bins, color=COLOR_TRAIN, label='training pool')
    ax.hist(axis[tail_test], bins=bins, color=COLOR_TAIL, label='tail-low held out')
    ax.hist(axis[hole_test], bins=bins, color=COLOR_HOLE, label='interior hole held out')

    ax.set_xlabel('aspect ratio')
    ax.set_ylabel('configurations')
    ax.set_title('Two ways of holding out 20% of the data')
    ax.legend()
    save_figure(fig, 'split_design')


def figure_distance_reach(axis):
    """Why the matched-distance claim is hard: the hole barely reaches out."""
    tail_train, tail_test = tail_split(axis, 'low', TEST_FRACTION)
    hole_train, hole_test = hole_split(axis, TEST_FRACTION)

    tail_d = distance_from_training_region(axis, tail_train)[tail_test]
    hole_d = distance_from_training_region(axis, hole_train)[hole_test]

    fig, ax = plt.subplots()
    bins = np.linspace(0, max(tail_d.max(), hole_d.max()), 60)
    ax.hist(tail_d, bins=bins, color=COLOR_TAIL, alpha=0.8, label=f'tail (max {tail_d.max():.2f})')
    ax.hist(hole_d, bins=bins, color=COLOR_HOLE, alpha=0.8, label=f'hole (max {hole_d.max():.2f})')

    ax.set_yscale('log')
    ax.set_xlabel('distance from training region (std of aspect ratio)')
    ax.set_ylabel('held-out configurations')
    ax.set_title('How far each split actually reaches')
    ax.legend()
    save_figure(fig, 'distance_reach')


def figure_grid_ratios():
    rows = load_results('day_1_2_grid')['results']['rows']

    targets = sorted({r['target'] for r in rows})
    combos = []
    for r in rows:
        key = (r['axis'], r['cut'])
        if key not in combos:
            combos.append(key)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(combos))
    width = 0.38

    for i, target in enumerate(targets):
        values = [
            next(r['ratio'] for r in rows if (r['axis'], r['cut']) == c and r['target'] == target)
            for c in combos
        ]
        ax.bar(x + (i - 0.5) * width, values, width, label=target)

    ax.axhline(1.0, color=COLOR_NEUTRAL, linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}\n{c}' for a, c in combos], fontsize=8)
    ax.set_ylabel('out-of-region RMSE / in-region RMSE')
    ax.set_title('Generalization gap by axis and cut (1.0 = no penalty)')
    ax.legend()
    save_figure(fig, 'grid_ratios')


def figure_noise_floor():
    spread = load_results('stage_2_noise_floor')['results']['residual_spread']

    fig, axes = plt.subplots(1, len(spread), figsize=(10, 4.5))
    axes = np.atleast_1d(axes)

    for ax, (name, result) in zip(axes, spread.items(), strict=False):
        d = [b['median_distance'] for b in result['bins']]
        gap = [b['median_delta'] for b in result['bins']]
        ax.plot(d, gap, 'o-', color=COLOR_HOLE, label='measured')

        # The fitted line extended back to zero distance. Its intercept is the
        # floor estimate, which is the whole point of the figure.
        xs = np.linspace(0, max(d), 50)
        ax.plot(
            xs,
            result['intercept'] + result['slope'] * xs,
            '--',
            color=COLOR_TAIL,
            label=f'fit, intercept {result["intercept"]:.5f}',
        )
        ax.axhline(result['bar'], color=COLOR_NEUTRAL, linestyle=':', label='floor bar')

        ax.set_xlabel('neighbour distance (z-scored, 80D)')
        ax.set_ylabel('median |target difference|')
        ax.set_title(f'{name}\n{result["verdict"]}')
        ax.legend(fontsize=8)

    fig.suptitle('Near-identical shapes carry near-identical answers')
    save_figure(fig, 'noise_floor')


def figure_gate_3_5_deciles():
    result = load_results('gate_3_5_split_axis')['results']
    deciles = result['deciles']

    centres = [(d['lo'] + d['hi']) / 2 for d in deciles]
    errors = [d['mean_abs_error'] for d in deciles]
    widths = [(d['hi'] - d['lo']) * 0.9 for d in deciles]

    fig, ax = plt.subplots()
    ax.bar(centres, errors, width=widths, color=COLOR_HOLE)
    ax.set_xlabel('aspect ratio')
    ax.set_ylabel('mean |error|')
    ax.set_title('Prediction error tracks data density, before any surrogate exists')
    save_figure(fig, 'gate_3_5_deciles')


def main():
    print('data-derived figures')
    axis = load_axis_values()
    figure_split_design(axis)
    figure_distance_reach(axis)

    print('results-derived figures')
    for name, fn in [
        ('day_1_2_grid', figure_grid_ratios),
        ('stage_2_noise_floor', figure_noise_floor),
        ('gate_3_5_split_axis', figure_gate_3_5_deciles),
    ]:
        try:
            fn()
        except FileNotFoundError:
            print(f'  skipped, results/{name}.json not found. Run that script first.')


if __name__ == '__main__':
    # Headless only when run as a script. Setting this at import time would
    # force a non-display backend on any notebook that imports these functions,
    # and nothing would render inline.
    matplotlib.use('Agg')
    main()
