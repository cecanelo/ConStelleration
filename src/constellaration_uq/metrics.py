"""Scoring functions and the shared distance binning.

Everything here is array in, array out. No file IO, no plotting, no model. This
is the part that can be silently wrong, so it is the part that gets tested.

The binning lives here rather than in a script because two figures depend on the
bins agreeing. The distance-error curve says error grows with distance and the
coverage curve says honesty falls with distance, and a reader is meant to lay
them side by side and see one story told twice. If each script derived its own
quantile edges, the two x axes would drift apart on any change to either
evaluation set, both plots would still render, and the comparison would quietly
stop being a comparison. One definition means they agree by construction.
"""

import numpy as np


def distance_bins(distance, n_bins):
    """Split points into equal-count quantile bins of distance.

    Returns (rows, masks). Each row carries only the bin geometry, count and
    edges and median distance, with no measured quantity in it. The caller adds
    its own fields using the matching mask, which is what lets the error curve
    and the calibration curves sit in identical bins while reporting different
    things.

    Equal-count rather than equal-width, so every point on a curve rests on the
    same amount of evidence and the thin far tail cannot produce a bin of two
    configurations masquerading as a measurement. Distance is heavily skewed and
    the hole and tail splits differ tenfold in range, so shared uniform edges
    would leave the hole with two usable bins and the tail with one crowded one.
    This is the same trap that flipped the verdict in the stage 2 noise floor
    check before it was rebinned.
    """
    edges = np.quantile(distance, np.linspace(0, 1, n_bins + 1))

    # searchsorted with side='right' puts the maximum past the last edge, so the
    # furthest point would be clipped back into the top bin rather than binned
    # into it. Nudging the top edge up by one float keeps that a measurement.
    edges[-1] = np.nextafter(edges[-1], np.inf)
    which = np.clip(np.searchsorted(edges, distance, side='right') - 1, 0, n_bins - 1)

    rows, masks = [], []
    for b in range(n_bins):
        sel = which == b
        if not sel.any():
            # Duplicate quantile edges collapse a bin. Happens on the random
            # split, where most distances are identical and tiny.
            continue
        rows.append(
            {
                'n': int(sel.sum()),
                'd_lo': float(edges[b]),
                'd_hi': float(edges[b + 1]),
                'd_median': float(np.median(distance[sel])),
            }
        )
        masks.append(sel)

    return rows, masks
