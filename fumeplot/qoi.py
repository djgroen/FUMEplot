"""
QoI (Quantity of Interest) extraction functions for the rw_sa random walk /
Brownian motion output files (RW1D, RW2D, Brown3D - see rw_sa/ scripts).

Frame parsing itself is NOT done here - it reuses the real decoders already
in this plugin (xyz_decoder.read_xyz_file, brown3d_decoder.read_brown3d_file)
rather than duplicating that logic. The two functions below are thin
adapters converting those decoders' dict-of-tuples output into the numpy
(N, 3) arrays the QoI/time-series code operates on.

NOTE: the Brown3D script does NOT write per-particle mass to the output
file, even though mass varies per particle internally (massRange).
Kinetic energy below takes masses as an explicit argument for this reason
- see the flag in the chat reply.
"""
import numpy as np

from xyz_decoder import read_xyz_file
from brown3d_decoder import read_brown3d_file


def positions_from_xyz(filepath):
    """
    Adapter over the real xyz_decoder.read_xyz_file: returns just the
    (N, 3) position array the QoI/time-series code needs, sorted by
    particle ID so frame-to-frame ordering is consistent regardless of
    file-write order (the actual writer always writes ID-ascending, so
    this is a no-op in practice, but it's a one-line safety net).
    """
    particles = sorted(read_xyz_file(filepath)["particles"], key=lambda p: p[0])
    return np.array([[p[1], p[2], p[3]] for p in particles])


def positions_velocities_from_lammps(filepath):
    """
    Adapter over the real brown3d_decoder.read_brown3d_file: returns
    (timestep, positions (N,3), velocities (N,3)), sorted by particle ID
    for the same reason as positions_from_xyz.
    """
    data = read_brown3d_file(filepath)
    particles = sorted(data["particles"], key=lambda p: p[0])
    positions = np.array([[p[1], p[2], p[3]] for p in particles])
    velocities = np.array([[p[4], p[5], p[6]] for p in particles])
    return data["timestep"], positions, velocities


def compute_centroid_and_spread(positions):
    """
    positions: (N, 3) array for one frame.
    Returns (centroid: (3,) array, spread: float).
    spread = RMS distance of particles from the centroid
           = sqrt(mean(||r_i - centroid||^2))
    (equivalent to the radius of gyration of the particle cloud)
    """
    centroid = positions.mean(axis=0)
    diffs = positions - centroid
    spread = np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))
    return centroid, spread


def compute_centroid_displacement(positions, initial_centroid=None):
    """
    Displacement of the swarm's centroid from its starting point.
    All three scripts (RW1D/RW2D/Brown3D) initialise every particle's
    position at the origin, so initial_centroid defaults to (0,0,0)
    unless you pass the actual frame-0 centroid explicitly.
    """
    if initial_centroid is None:
        initial_centroid = np.zeros(3)
    centroid = positions.mean(axis=0)
    return float(np.linalg.norm(centroid - initial_centroid))


def compute_total_kinetic_energy(velocities, masses):
    """
    velocities: (N, 3) array. masses: (N,) array or scalar (uniform mass).
    KE = sum_i 0.5 * m_i * |v_i|^2
    Only meaningful for Brown3D (RW1D/RW2D have no velocity field).
    """
    speeds_sq = np.sum(velocities ** 2, axis=1)
    return float(np.sum(0.5 * masses * speeds_sq))
