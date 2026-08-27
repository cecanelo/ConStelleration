"""Gate 3.5: is the split axis a function of the inputs?

Predict metrics.aspect_ratio from the 80 boundary coefficients. If it comes out
near-perfectly, splitting on aspect ratio is a shift in the questions asked, not
in the answers, and the leakage objection is settled.

Pre-registered thresholds, decided before running:
    R2 >= 0.99      pass
    0.90 to 0.99    inconclusive, escalate before judging
    R2 < 0.90       real failure, aspect ratio is not what we think

Correction, made after the first run. The bands above were written as if they
scored ridge. They should score the best cheap model instead. Aspect ratio is
roughly R(0,0) over the minor radius, a ratio, and a linear model cannot
represent division, so ridge could never reach the ceiling regardless of whether
the mapping exists. The numbers are unchanged; only what they score changed.

Ridge is kept as a diagnostic, not a verdict. A weak linear fit combined with a
near-zero residual correlation against R(0,0) is the evidence that the shortfall
is model class rather than missing information.

First run, for the record: ridge R2 0.784, corr(residual, R(0,0)) +0.075.

Expected ceiling is around 0.999, not 1.0. R(0,0) is dropped from the inputs per
1.5 and aspect ratio depends on it, so there is a known information floor.

"""

from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from constellaration_uq.data import extract_input_features, filter_valid, load_raw

DATA_DIR = Path(__file__).resolve().parents[1] / 'data_raw' / 'data'
AXIS = 'metrics.aspect_ratio'
SEED = 0


def main():
    for _, df in filter_valid(load_raw(DATA_DIR)):
        pass
    print(f'pool: {len(df):,} rows')

    X = extract_input_features(df)
    y = df[AXIS].to_numpy()
    r00 = np.array([row.tolist() for row in df['boundary.r_cos']])[:, 0, 4]

    X_train, X_test, y_train, y_test, _, r00_test = train_test_split(
        X, y, r00, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)
    model = RidgeCV(alphas=np.logspace(-6, 2, 9)).fit(scaler.transform(X_train), y_train)

    pred = model.predict(scaler.transform(X_test))
    residual = y_test - pred
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(np.mean(residual**2))

    print('ridge (diagnostic, linear, not the verdict)')
    print(f'  alpha : {model.alpha_:g}')
    print(f'  R2    : {r2:.6f}')
    print(f'  RMSE  : {rmse:.6f}   (std of y: {y_test.std():.4f})')

    corr = np.corrcoef(residual, r00_test)[0, 1]
    print(f'corr(residual, R(0,0)): {corr:+.4f}')
    print(f'R(0,0) range: {r00_test.min():.4f} to {r00_test.max():.4f}')

    gbm = HistGradientBoostingRegressor(max_iter=1000, random_state=SEED).fit(X_train, y_train)
    gbm_pred = gbm.predict(X_test)
    gbm_r2 = r2_score(y_test, gbm_pred)
    gbm_rmse = np.sqrt(np.mean((y_test - gbm_pred) ** 2))

    print('\ngradient boosting (nonlinear, trees)')
    print(f'  R2    : {gbm_r2:.6f}')
    print(f'  RMSE  : {gbm_rmse:.6f}')

    y_mean, y_std = y_train.mean(), y_train.std()

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 256, 256),
        activation='tanh',
        random_state=SEED,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=50,
    ).fit(scaler.transform(X_train), (y_train - y_mean) / y_std)

    mlp_pred = mlp.predict(scaler.transform(X_test)) * y_std + y_mean
    mlp_r2 = r2_score(y_test, mlp_pred)
    mlp_rmse = np.sqrt(np.mean((y_test - mlp_pred) ** 2))

    print('\nMLP (nonlinear, smooth, A.4 architecture)')
    print(f'  iters : {mlp.n_iter_}')
    print(f'  R2    : {mlp_r2:.6f}')
    print(f'  RMSE  : {mlp_rmse:.6f}')

    best_r2 = max(gbm_r2, mlp_r2)
    best_name = 'gradient boosting' if gbm_r2 >= mlp_r2 else 'MLP'
    verdict = 'PASS' if best_r2 >= 0.99 else 'INCONCLUSIVE' if best_r2 >= 0.90 else 'FAIL'
    print(f'\nbest model: {best_name}, R2 {best_r2:.6f}')
    print(f'verdict: {verdict}')

    best_pred = gbm_pred if gbm_r2 >= mlp_r2 else mlp_pred
    abs_err = np.abs(y_test - best_pred)

    print(f'\nerror distribution ({best_name})')
    for label, value in [
        ('median', np.median(abs_err)),
        ('p90', np.percentile(abs_err, 90)),
        ('p99', np.percentile(abs_err, 99)),
        ('max', abs_err.max()),
    ]:
        print(f'  {label:6s}: {value:.4f}')

    print('\nmean |error| by aspect ratio decile')
    edges = np.quantile(y_test, np.linspace(0, 1, 11))
    for i in range(10):
        in_bin = (y_test >= edges[i]) & (y_test <= edges[i + 1])
        print(
            f'  A {edges[i]:5.2f} to {edges[i + 1]:5.2f}: '
            f'{abs_err[in_bin].mean():.4f}  (n={in_bin.sum()})'
        )

    lo, hi = np.quantile(y, [0.0005, 0.9995])
    keep = (y_test >= lo) & (y_test <= hi)
    trimmed_rmse = np.sqrt(np.mean((y_test[keep] - best_pred[keep]) ** 2))
    print(f'\nafter 0.05% trim ({(~keep).sum()} test points dropped)')
    print(f'  R2    : {r2_score(y_test[keep], best_pred[keep]):.6f}')
    print(f'  RMSE  : {trimmed_rmse:.6f}')


if __name__ == '__main__':
    main()
