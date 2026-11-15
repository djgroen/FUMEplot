import numpy as np

from qoi import compute_centroid_and_spread, compute_centroid_displacement, compute_total_kinetic_energy
from ts_assembly import estimate_diffusion_coefficient
from multi_trial import assemble_multi_trial_positions


def _spread_scaling_exponent(steps, spreads):
    logt = np.log(steps)
    logs = np.log(spreads)
    p, _ = np.polyfit(logt, logs, 1)
    return float(p)


class RandomWalkDecoder:

    def __init__(self, file_base, dim, num_trials=3):
        self.file_base = file_base
        self.dim = dim
        self.num_trials = num_trials
        self.output_columns = [
            "spread_final",
            "centroid_displacement_final",
            "diffusion_coefficient_D",
            "spread_profile",
            "centroid_displacement_profile",
            "steps",
        ]

    def parse_sim_output(self, run_info=None):
        run_info = run_info or {}
        run_dir = run_info.get("run_dir", ".")
        trials = list(range(1, self.num_trials + 1))
        steps, pooled_positions = assemble_multi_trial_positions(run_dir, self.file_base, trials, "xyz")

        spreads = np.array([compute_centroid_and_spread(p)[1] for p in pooled_positions])
        centroid_disps = np.array([compute_centroid_displacement(p) for p in pooled_positions])
        D, slope, intercept = estimate_diffusion_coefficient(steps, spreads, self.dim)

        return {
            "spread_final": float(spreads[-1]),
            "centroid_displacement_final": float(centroid_disps[-1]),
            "diffusion_coefficient_D": float(D),
            "spread_profile": spreads.tolist(),
            "centroid_displacement_profile": centroid_disps.tolist(),
            "steps": list(steps),
        }


class BrownianDecoder:

    def __init__(self, file_base="Brown3D", num_trials=3, masses=1.0):
        self.file_base = file_base
        self.num_trials = num_trials
        self.masses = masses
        self.output_columns = [
            "spread_final",
            "centroid_displacement_final",
            "kinetic_energy_final",
            "spread_scaling_exponent",
            "diffusion_coefficient_D",
            "spread_profile",
            "centroid_displacement_profile",
            "kinetic_energy_profile",
            "steps",
        ]

    def parse_sim_output(self, run_info=None):
        run_info = run_info or {}
        run_dir = run_info.get("run_dir", ".")
        trials = list(range(1, self.num_trials + 1))
        steps, pooled_positions = assemble_multi_trial_positions(run_dir, self.file_base, trials, "lammps")

        from ts_assembly import assemble_trial_timeseries
        per_trial_velocities = []
        for trial in trials:
            _, _, velocities_list = assemble_trial_timeseries(run_dir, self.file_base, trial, "lammps")
            per_trial_velocities.append(velocities_list)
        pooled_velocities = [
            np.vstack([per_trial_velocities[t][i] for t in range(len(trials))])
            for i in range(len(steps))
        ]

        spreads = np.array([compute_centroid_and_spread(p)[1] for p in pooled_positions])
        centroid_disps = np.array([compute_centroid_displacement(p) for p in pooled_positions])
        kinetic_energies = np.array(
            [compute_total_kinetic_energy(v, self.masses) for v in pooled_velocities]
        )
        exponent = _spread_scaling_exponent(steps, spreads)

        return {
            "spread_final": float(spreads[-1]),
            "centroid_displacement_final": float(centroid_disps[-1]),
            "kinetic_energy_final": float(kinetic_energies[-1]),
            "spread_scaling_exponent": exponent,
            "diffusion_coefficient_D": None,
            "spread_profile": spreads.tolist(),
            "centroid_displacement_profile": centroid_disps.tolist(),
            "kinetic_energy_profile": kinetic_energies.tolist(),
            "steps": list(steps),
        }
