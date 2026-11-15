"""
Multi-trial pooling for the diffusion coefficient D fit.

Rather than averaging three separate per-trial D estimates (which
inherits each trial's own noisy spread estimate), this pools the raw
particle positions across all trials at each matching timestep before
computing centroid/spread once. That's equivalent to having
numParticles * numTrials independent particles at each timestep, which
is the larger, cleaner sample for estimating the true ensemble spread.
"""
import numpy as np

from qoi import compute_centroid_and_spread
from ts_assembly import assemble_trial_timeseries, estimate_diffusion_coefficient


def assemble_multi_trial_positions(directory, file_base, trials, file_format):
    """
    trials: list of trial numbers, e.g. [1, 2, 3].
    Returns (steps: list[int], pooled_positions: list[(N_total,3) array])
    where pooled_positions[i] concatenates every trial's particles at
    steps[i]. Raises if the trials don't share the same output steps.
    """
    per_trial = []
    reference_steps = None
    for trial in trials:
        steps, positions_list, _ = assemble_trial_timeseries(directory, file_base, trial, file_format)
        if reference_steps is None:
            reference_steps = steps
        elif steps != reference_steps:
            raise ValueError(
                f"trial {trial} has steps {steps}, expected {reference_steps} "
                f"(all trials must share the same output steps to pool)"
            )
        per_trial.append(positions_list)

    pooled_positions = []
    for frame_idx in range(len(reference_steps)):
        pooled_positions.append(np.vstack([per_trial[t][frame_idx] for t in range(len(trials))]))

    return reference_steps, pooled_positions


def estimate_diffusion_coefficient_multi_trial(directory, file_base, trials, file_format, dim):
    """Convenience wrapper: pool trials, compute spreads, fit D."""
    steps, pooled_positions = assemble_multi_trial_positions(directory, file_base, trials, file_format)
    spreads = np.array([compute_centroid_and_spread(p)[1] for p in pooled_positions])
    D, slope, intercept = estimate_diffusion_coefficient(steps, spreads, dim)
    return D, slope, intercept, steps, spreads
