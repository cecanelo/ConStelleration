"""Stage 2: is there a noise floor in this data, and where is it?

One kNN index on z-scored coefficients answers three questions:

  2.3  Residual spread. Do near-identical shapes carry near-identical targets?
       Spread among a point's k neighbours, read against neighbour distance,
       so "dense region" is defined by the data rather than assumed.

  2.2  Tolerance duplicates. The exact-match pass is done (8 in 27,050). This
       is the near-miss version: pairs below a small distance whose targets
       disagree. Tolerance comes from the nearest-neighbour distance
       distribution, not picked in advance.

  1.7  Mirror pairs. Negating every z_sin coefficient gives the mirror shape.
       Does the pool contain both handedness versions of the same design, and
       if so does the target flip sign? Query the same index with mirrored
       copies. The result is a comparison of two distance distributions, real
       neighbours versus mirrored neighbours, not a pair count.

Pre-registered prediction (2.1), written before looking: aleatoric sits at the
numerical floor. The inputs are fully observed, the solver is deterministic, and
a fixed convergence tolerance gives a deterministic high-frequency function
rather than noise. Confirming this is the result. Finding real noise instead
would mean something about the data is not what we think.

Gate 2.5 stops the project if dense-region spread is far above the floor, or if
near-duplicate inputs carry substantially disagreeing targets.
"""

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from constellaration_uq.data import extract_input_features, filter_valid, load_raw

DATA_DIR = Path(__file__).resolve().parents[1] / 'data_raw' / 'data'
K = 10
NEAREST_BINS_FOR_FIT = 3
BIN_QUANTILES = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
TOLERANCE = 0.2

# Pre-registered bar: 20% of Table 7's RMSE (0.006 edge iota, 0.051 log10 qi).
# Errors combine in quadrature, so a floor at 20% of model error inflates total
# error by 2%, which is invisible. Below the bar, aleatoric is negligible (2.1).

FLOOR_BAR = {
    'edge_rot_transform': 0.2 * 0.006,
    'log10_qi': 0.2 * 0.051,
}


def load_pool():
    for _, df in filter_valid(load_raw(DATA_DIR)):
        pass
    return df


def main():
    df = load_pool()
    print(f'pool: {len(df):,} rows')

    X = extract_input_features(df)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    targets = {
        'edge_rot_transform': df[
            'metrics.edge_rotational_transform_over_n_field_periods'
        ].to_numpy(),
        'log10_qi': np.log10(df['metrics.qi']).to_numpy(),
    }

    for name, y in targets.items():
        print(f'    {name:20s}  std={np.nanstd(y):.6f}   non_finite={(~np.isfinite(y)).sum()}')

    nn = NearestNeighbors(n_neighbors=K + 1, algorithm='brute').fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)

    self_matched = indices[:, 0] == np.arange(len(X_scaled))
    print(f'  self-match in column 0: {self_matched.sum():,} of {len(X_scaled):,}')

    distances, indices = distances[:, 1:], indices[:, 1:]
    print(
        f'  nearest-neighbour distance: median {np.median(distances[:, 0]):.4f}, '
        f'min {distances[:, 0].min():.6f}'
    )

    for name, y in targets.items():
        residual_spread(name, y, distances[:, 0], indices[:, 0])

    distance_percentiles(distances[:, 0])

    if TOLERANCE is None:
        print('\nTOLERANCE not set, skipping duplicate check. Pick it from the percentiles above.')
    else:
        for name, y in targets.items():
            tolerance_duplicates(name, y, distances[:, 0], indices[:, 0], TOLERANCE)


def residual_spread(name, y, nn_distance, nn_index):
    delta = np.abs(y - y[nn_index])
    finite = np.isfinite(delta)
    delta, dist = delta[finite], nn_distance[finite]

    edges = np.quantile(dist, BIN_QUANTILES)
    med_dist = []
    med_delta = []

    print(f'\n{name}: median |delta y| by nearest-neighbour distance bin')
    for i in range(len(edges) - 1):
        in_bin = (dist >= edges[i]) & (dist <= edges[i + 1])
        med_dist.append(np.median(dist[in_bin]))
        med_delta.append(np.median(delta[in_bin]))
        print(
            f'  d {edges[i]:7.4f} to {edges[i + 1]:7.4f}: {med_delta[-1]:.6f}  (n={in_bin.sum():,})'
        )

    window = NEAREST_BINS_FOR_FIT
    slope, intercept = np.polyfit(med_dist[:window], med_delta[:window], 1)
    bar = FLOOR_BAR[name]
    verdict = 'AT FLOOR' if intercept < bar else 'ABOVE FLOOR'
    print(f'  slope {slope:.6f}, intercept {intercept:.6f}, bar {bar:.6f}  -> {verdict}')
    return intercept


def distance_percentiles(nn_distance):
    print('\nnearest-neighbour distance percentiles')
    for q in (0.01, 0.05, 0.1, 0.5, 1, 2, 5, 25, 50):
        print(f'  p{q:<5g}: {np.percentile(nn_distance, q):.6f}')


def tolerance_duplicates(name, y, nn_distance, nn_index, tolerance):
    delta = np.abs(y - y[nn_index])
    close = (nn_distance < tolerance) & np.isfinite(delta)

    print(f'\n{name}: pairs closer than {tolerance:.6f}')
    print(f'  count: {close.sum():,}')
    if not close.any():
        return

    d = delta[close]
    bar = FLOOR_BAR[name]
    print(f'  |dy| median {np.median(d):.6f}, p90 {np.percentile(d, 90):.6f}, max {d.max():.6f}')
    print(f'  bar {bar:.6f}, pairs over bar: {(d > bar).sum():,}')


if __name__ == '__main__':
    main()
