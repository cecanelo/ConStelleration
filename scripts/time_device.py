"""Is the N-sweep faster on GPU? Time one member network on each device.

The sweep is 300 member networks and the only remaining training, so it is worth
five minutes to find out whether switching the studio to GPU pays. Assuming it
does is the trap: these are three layers of 256 on 80 inputs at batch 128, which
is small enough that a GPU can spend most of its time on kernel launch overhead
rather than arithmetic.

Timed at two sizes, because the answer can differ between them and the sweep
spends most of its runs at the small end. N = 1000 is the sweep's smallest rung
and its worst case for GPU utilisation; full N is the best case.

Run it on a GPU machine: it times both devices in one process, so the comparison
is on identical data with no switching in between.

    python3 scripts/time_device.py

⚠️ Times ONE member with the frozen recipe. Do not change the recipe to make a
number look better, the point is to measure the sweep as it will actually run.
"""

import time

import numpy as np
import torch

from constellaration_uq.baseline import load_pool
from constellaration_uq.data import extract_input_features, trim_target_tails
from constellaration_uq.nets import VAL_SIZE, split_validation, train_one_mv
from constellaration_uq.splits import tail_split

SEED = 0
TEST_FRACTION = 0.2
TARGET_COL = 'metrics.edge_rotational_transform_over_n_field_periods'
AXIS_COL = 'metrics.aspect_ratio'

# The sweep's smallest and largest rungs. The middle ones interpolate.
SIZES = [1000, None]

# One member per (size, device) is enough to separate a 3x speedup from a 1.1x,
# which is the only distinction that changes the decision.
N_MEMBERS_PER_CELL = 1


def devices():
    """CPU always, CUDA when present. Named so the printout is unambiguous."""
    found = [('cpu', torch.device('cpu'))]
    if torch.cuda.is_available():
        found.append((torch.cuda.get_device_name(0), torch.device('cuda')))
    else:
        print('⚠️  no CUDA device visible, timing CPU only. Switch the studio to')
        print('   a GPU machine and rerun to get the comparison.\n')
    return found


def warm_up(X_fit, y_fit, X_val, y_val, device):
    """One tiny fit before the timed one.

    CUDA context creation and kernel autotuning happen on first use and cost
    seconds, which would land entirely on whichever run went first and make the
    comparison meaningless.
    """
    train_one_mv(X_fit[:256], y_fit[:256], X_val, y_val, seed=99, max_epochs=2, device=device)


def main():
    df = load_pool()
    trimmed = trim_target_tails(df, TARGET_COL)
    X = extract_input_features(trimmed)
    y = trimmed[TARGET_COL].to_numpy()
    axis = trimmed[AXIS_COL].to_numpy()

    # The tail split, because that is the split the sweep runs on. Its fit set
    # is what the sweep subsamples from.
    train_mask, _ = tail_split(axis, 'low', TEST_FRACTION)
    fit_pool = np.flatnonzero(train_mask)
    X_fit_all, y_fit_all, X_val, y_val = split_validation(
        X[fit_pool], y[fit_pool], VAL_SIZE, SEED
    )
    print(f'pool {len(trimmed):,} rows, fit set {len(X_fit_all):,}, validation {len(X_val):,}\n')

    header = f'{"device":>22s} {"N":>7s} {"epochs":>7s} {"seconds":>9s} {"s/epoch":>9s}'
    print(header)
    print('-' * len(header))

    timings = {}
    for label, device in devices():
        for size in SIZES:
            n = len(X_fit_all) if size is None else size
            X_fit, y_fit = X_fit_all[:n], y_fit_all[:n]

            warm_up(X_fit, y_fit, X_val, y_val, device)

            started = time.time()
            for k in range(N_MEMBERS_PER_CELL):
                _, history = train_one_mv(X_fit, y_fit, X_val, y_val, seed=k, device=device)
            # CUDA is asynchronous, so without this the timer can stop before
            # the work does and the GPU looks arbitrarily fast.
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.time() - started) / N_MEMBERS_PER_CELL

            epochs = len(history)
            timings[(label, n)] = elapsed
            print(
                f'{label[:22]:>22s} {n:7,d} {epochs:7d} '
                f'{elapsed:8.1f}s {elapsed / epochs:8.2f}s'
            )

    gpu_labels = [label for label, _ in devices() if label != 'cpu']
    if not gpu_labels:
        return

    gpu = gpu_labels[0]
    print('\nspeedup, GPU over CPU')
    for size in SIZES:
        n = len(X_fit_all) if size is None else size
        ratio = timings[('cpu', n)] / timings[(gpu, n)]
        print(f'  N = {n:6,d}   {ratio:.2f}x')

    # The sweep is 300 member networks weighted toward the small rungs, so the
    # small-N ratio matters more than the large-N one.
    print(
        '\nThe sweep runs 300 networks across N = 1k to 18k, weighted toward the\n'
        'small end, so read the N = 1,000 row as the one that decides this.\n'
        'Under about 2x, stay on CPU: the switch costs money, and an attached\n'
        'VS Code session stops a GPU studio from auto-sleeping.'
    )


if __name__ == '__main__':
    main()
