"""
Time-series assembly layer for rw_sa output files.

Stitches a single trial's sequence of per-timestep files
(<fileBase>-T<trial>-S<step>.txt, written every outputRate steps) into
one ordered time series, and provides the two things built on top of
that series: the spread-vs-time curve, and a diffusion coefficient D
estimated from its slope.

Why "spread" rather than raw position for the D fit: spread (RMS
distance from the ensemble centroid, from qoi.compute_centroid_and_spread)
already removes any uniform drift (flowVelocity/flowVelocityX/Y/Z),
since drift shifts every particle's position by the same amount and
therefore shifts the centroid by that same amount, cancelling out of
(r_i - centroid). So spread(t)^2 isolates the diffusive part of the
motion even when a non-zero flow velocity is present, with no need to
subtract the drift explicitly first.

For pure diffusion in d dimensions, MSD(t) = spread(t)^2 grows as
2*d*D*t, so D = slope(spread^2 vs t) / (2*d).
"""
import glob
import os
import re

import numpy as np

from qoi import compute_centroid_and_spread, positions_from_xyz, positions_velocities_from_lammps


def list_trial_files(directory, file_base, trial):
    """Return [(step:int, filepath)] for one trial, sorted by step."""
    pattern = os.path.join(directory, f"{file_base}-T{trial}-S*.txt")
    files = glob.glob(pattern)

    def step_of(path):
        m = re.search(r"-S(\d+)\.txt$", os.path.basename(path))
        return int(m.group(1))

    return sorted(((step_of(f), f) for f in files), key=lambda x: x[0])


def assemble_trial_timeseries(directory, file_base, trial, file_format):
    """
    file_format: 'xyz' (RW1D/RW2D) or 'lammps' (Brown3D).
    Returns (steps: list[int], positions_list: list[(N,3) array],
             velocities_list: list[(N,3) array] or None for 'xyz').
    """
    entries = list_trial_files(directory, file_base, trial)
    if not entries:
        raise FileNotFoundError(
            f"No files matching {file_base}-T{trial}-S*.txt in {directory}"
        )

    steps, positions_list, velocities_list = [], [], []
    for step, path in entries:
        if file_format == "xyz":
            positions_list.append(positions_from_xyz(path))
        elif file_format == "lammps":
            ts, pos, vel = positions_velocities_from_lammps(path)
            if ts != step:
                raise ValueError(
                    f"{path}: filename step {step} != in-file TIMESTEP {ts}"
                )
            positions_list.append(pos)
            velocities_list.append(vel)
        else:
            raise ValueError(f"unknown file_format: {file_format!r}")
        steps.append(step)

    return steps, positions_list, (velocities_list if file_format == "lammps" else None)


def spread_timeseries(positions_list):
    """Per-frame (centroid, spread) -> returns spreads as np.array."""
    return np.array([compute_centroid_and_spread(p)[1] for p in positions_list])


def estimate_diffusion_coefficient(steps, spreads, dim):
    """
    Linear fit of spread(t)^2 vs t; D = slope / (2*dim).
    Returns (D, slope, intercept) so the fit quality can be inspected
    (e.g. a non-trivial intercept or visibly curved residuals would
    mean the series isn't in the linear/diffusive regime yet).
    """
    t = np.asarray(steps, dtype=float)
    msd = np.asarray(spreads, dtype=float) ** 2
    slope, intercept = np.polyfit(t, msd, 1)
    return slope / (2 * dim), slope, intercept
