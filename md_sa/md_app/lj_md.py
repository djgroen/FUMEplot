#!/usr/bin/env python3
"""
Minimal Lennard-Jones molecular dynamics engine (reduced LJ units, numpy only).

This is a stand-in for a production MD code (LAMMPS, GROMACS, ...) so that the
sensitivity campaign can run end-to-end on a laptop. It reads a JSON input
written by the EasyVVUQ encoder and writes a JSON output read by the decoder:

    python3 lj_md.py md_input.json md_output.json

Physics: N = 4*n_cells^3 atoms start on an FCC lattice at number density
`density`, interact through a truncated-and-shifted LJ potential with cutoff
`r_cut` (no tail corrections, so the cutoff is a genuine model parameter), and
are coupled to a Langevin thermostat at `temperature` with friction `gamma`
(BAOAB splitting). The random stream is seeded with `seed`, so a fixed seed
across a campaign gives common random numbers: the remaining run-to-run
variation is due to the sampled inputs only.

Quantities of interest (QoIs) are produced at three dimensionalities:

    0D  epot        mean potential energy per atom   (production phase)
        pressure    mean virial pressure             (production phase)
        diff_coeff  self-diffusion coefficient from the MSD slope
    1D  epot_t      potential energy per atom vs time t   (whole run, incl. melting)
        msd         mean squared displacement vs lag time (production phase)
        rdf         radial distribution function g(r)     (production phase)
    2D  rdf_t       time-resolved g(t, r) over consecutive windows (whole run)

Pulled cloud (cf. the GROMACS pull code): the `n_pull` atoms nearest to the box
centre at t=0 form a cloud whose centroid is pulled along +x with a constant
total force `pull_force`, shared equally by its atoms. Its QoIs, measured from
t=0 on unwrapped coordinates:

    0D  pull_disp       centroid displacement along the pull axis at the end of the run
        cloud_spread    RMS radius of the cloud about its centroid at the end of the run
    1D  pull_disp_t     centroid displacement vs time t
        cloud_spread_t  RMS radius vs time t

`diff_coeff` and `msd` use only the unpulled atoms, with the system centroid
removed per frame, so the imposed drift does not enter the diffusion estimate.

Coordinates of 1D/2D QoIs are written under `axes`, in array-axis order.

Optional LAMMPS dump output (off unless dump_every > 0): every `dump_every`
steps a single-frame LAMMPS text dump `<dump_base>-T<dump_trial>-S<step>.txt`
is written to the working directory,

    ITEM: TIMESTEP / NUMBER OF ATOMS / BOX BOUNDS pp pp pp / ATOMS id type x y z vx vy vz

i.e. the layout read by fumeplot/brown3d_decoder.py, so fumeplot.decoders.
BrownianDecoder(file_base=dump_base, num_trials=1) applies unchanged.
`dump_group` is 'cloud' (the pulled atoms; the decoder's spread then equals
cloud_spread_t) or 'all'. Two conventions follow that decoder rather than
plain LAMMPS: x y z are UNWRAPPED (LAMMPS would call them xu yu zu), and the
origin is the dumped group's centroid at t = 0 (box bounds shifted to match),
because the decoder measures centroid displacement from the origin. Step 0 is
not dumped (the decoder fits log(spread) vs log(step)).
"""

import sys
import json
import time

import numpy as np


def fccLattice(n_cells, box):
    a = box / n_cells
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)])
    return ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) + 0.25) * a


def pairTerms(x, box, r_cut):
    """Forces, potential energy, virial sum_{i<j} r_ij.F_ij and squared pair distances."""
    d = x[:, None, :] - x[None, :, :]
    d -= box * np.round(d / box)            # minimum image; works on unwrapped coordinates
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    inside = r2 < r_cut**2
    ir2 = np.where(inside, 1.0 / r2, 0.0)
    ir6 = ir2**3
    e_shift = 4.0 * (r_cut**-12 - r_cut**-6)
    epot = 0.5 * np.sum(np.where(inside, 4.0 * ir6 * (ir6 - 1.0) - e_shift, 0.0))
    fmag = 24.0 * ir6 * (2.0 * ir6 - 1.0) * ir2      # F_ij = fmag_ij * d_ij
    forces = np.einsum('ij,ijk->ik', fmag, d)
    virial = 0.5 * np.sum(fmag[inside] * r2[inside])
    return forces, epot, virial, r2


def meanSquaredDisplacement(frames):
    """MSD averaged over all time origins, for lags up to half the trajectory."""
    n_lags = len(frames) // 2
    msd = np.zeros(n_lags)
    for lag in range(1, n_lags):
        disp = frames[lag:] - frames[:-lag]
        msd[lag] = np.mean(np.sum(disp**2, axis=-1))
    return msd


def writeLammpsDump(path, step, ids, x, v, lo, hi):
    """Single-frame LAMMPS text dump (atom style, one atom type, unit masses)."""
    with open(path, 'w') as f:
        f.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(ids)}\nITEM: BOX BOUNDS pp pp pp\n")
        for a, b in zip(lo, hi):
            f.write(f"{a:.10g} {b:.10g}\n")
        f.write("ITEM: ATOMS id type x y z vx vy vz\n")
        np.savetxt(f, np.column_stack([ids, np.ones(len(ids)), x, v]), fmt=['%d', '%d'] + ['%.10g'] * 6)


def runMD(p):
    rng = np.random.default_rng(int(p['seed']))
    T, rho, r_cut, gamma, dt = (float(p[k]) for k in ('temperature', 'density', 'r_cut', 'gamma', 'dt'))
    n_cells, n_steps, n_equil = int(p['n_cells']), int(p['n_steps']), int(p['n_equil'])
    every, window, n_bins, r_max = int(p['sample_every']), int(p['rdf_window']), int(p['rdf_bins']), float(p['r_max'])

    n_pull, pull_force = int(p.get('n_pull', 0)), float(p.get('pull_force', 0.0))

    N = 4 * n_cells**3
    box = (N / rho) ** (1.0 / 3.0)
    if max(r_cut, r_max) > box / 2:
        raise ValueError(f"r_cut={r_cut} / r_max={r_max} exceed half the box ({box/2:.3f}); increase n_cells")

    x = fccLattice(n_cells, box)
    v = rng.normal(0.0, np.sqrt(T), (N, 3))
    v -= v.mean(axis=0)
    v *= np.sqrt(T / (np.sum(v**2) / (3 * (N - 1))))

    cloud = np.argsort(np.sum((x - box / 2)**2, axis=1))[:n_pull]
    free = np.setdiff1d(np.arange(N), cloud)
    f_pull = np.zeros((N, 3))
    if n_pull:
        f_pull[cloud, 0] = pull_force / n_pull
    centroid0 = x[cloud].mean(axis=0) if n_pull else np.zeros(3)

    dump_every = int(p.get('dump_every', 0))
    dump_group = str(p.get('dump_group', 'cloud'))
    dump_base = str(p.get('dump_base', 'LJ'))
    dump_trial = int(p.get('dump_trial', 1))
    if dump_every:
        if dump_group == 'cloud' and not n_pull:
            raise ValueError("dump_group='cloud' needs n_pull > 0; use dump_group='all'")
        if dump_group not in ('cloud', 'all'):
            raise ValueError(f"dump_group={dump_group!r}: expected 'cloud' or 'all'")
        dump_idx = np.sort(cloud) if dump_group == 'cloud' else np.arange(N)
        dump_origin = x[dump_idx].mean(axis=0)
        dump_lo, dump_hi = -dump_origin, box - dump_origin

    forces, epot, virial, r2 = pairTerms(x, box, r_cut)
    forces += f_pull
    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1**2) * T)

    edges = np.linspace(0.0, r_max, n_bins + 1)
    shell = 4.0 / 3.0 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    iu = np.triu_indices(N, 1)
    n_windows = n_steps // window
    rdf_t_hist = np.zeros((n_windows, n_bins))
    rdf_t_count = np.zeros(n_windows)
    rdf_hist = np.zeros(n_bins)

    t_series, epot_series, prod_epot, prod_press, prod_temp, frames = [], [], [], [], [], []
    pull_disp_t, cloud_spread_t = [], []

    for step in range(1, n_steps + 1):
        # BAOAB Langevin integrator
        v += 0.5 * dt * forces
        x += 0.5 * dt * v
        v = c1 * v + c2 * rng.standard_normal((N, 3))
        x += 0.5 * dt * v
        forces, epot, virial, r2 = pairTerms(x, box, r_cut)
        forces += f_pull
        v += 0.5 * dt * forces

        if dump_every and step % dump_every == 0:
            writeLammpsDump(f"{dump_base}-T{dump_trial}-S{step}.txt", step, dump_idx + 1,
                            x[dump_idx] - dump_origin, v[dump_idx], dump_lo, dump_hi)

        if step % every:
            continue
        t_kin = np.sum(v**2) / (3 * (N - 1))
        hist = np.histogram(np.sqrt(r2[iu]), bins=edges)[0]
        t_series.append(step * dt)
        epot_series.append(epot / N)
        if n_pull:
            centroid = x[cloud].mean(axis=0)
            pull_disp_t.append(centroid[0] - centroid0[0])
            cloud_spread_t.append(np.sqrt(np.mean(np.sum((x[cloud] - centroid)**2, axis=1))))

        w = (step - 1) // window
        if w < n_windows:
            rdf_t_hist[w] += hist
            rdf_t_count[w] += 1

        if step > n_equil:
            prod_epot.append(epot / N)
            prod_press.append(rho * t_kin + virial / (3.0 * box**3))
            prod_temp.append(t_kin)
            rdf_hist += hist
            frames.append(x[free] - x.mean(axis=0))

    norm = 0.5 * N * rho * shell             # pairs counted once (i<j)
    rdf = rdf_hist / (len(frames) * norm)
    rdf_t = rdf_t_hist / (rdf_t_count[:, None] * norm[None, :])

    frames = np.array(frames)
    msd = meanSquaredDisplacement(frames)
    lag = np.arange(len(msd)) * every * dt
    half = len(msd) // 2
    diff_coeff = np.polyfit(lag[half:], msd[half:], 1)[0] / 6.0

    r_mid = 0.5 * (edges[1:] + edges[:-1])
    t_mid = (np.arange(n_windows) + 0.5) * window * dt
    return {
        'epot': float(np.mean(prod_epot)),
        'pressure': float(np.mean(prod_press)),
        'diff_coeff': float(diff_coeff),
        'temp_kin': float(np.mean(prod_temp)),
        'pull_disp': float(pull_disp_t[-1]) if n_pull else 0.0,
        'cloud_spread': float(cloud_spread_t[-1]) if n_pull else 0.0,
        'pull_disp_t': pull_disp_t,
        'cloud_spread_t': cloud_spread_t,
        'epot_t': epot_series,
        'msd': msd.tolist(),
        'rdf': rdf.tolist(),
        'rdf_t': rdf_t.tolist(),
        'axes': {
            'epot_t': {'t': t_series},
            'pull_disp_t': {'t': t_series},
            'cloud_spread_t': {'t': t_series},
            'msd': {'lag': lag.tolist()},
            'rdf': {'r': r_mid.tolist()},
            'rdf_t': {'t': t_mid.tolist(), 'r': r_mid.tolist()},
        },
        'meta': {'n_atoms': N, 'box': box, 'n_pull': n_pull,
                 'lammps_dump': ({'pattern': f"{dump_base}-T{dump_trial}-S*.txt", 'every': dump_every,
                                  'group': dump_group, 'coords': 'unwrapped',
                                  'origin': dump_origin.tolist()} if dump_every else None)},
    }


if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else 'md_input.json'
    out_file = sys.argv[2] if len(sys.argv) > 2 else 'md_output.json'
    with open(in_file) as f:
        params = json.load(f)

    t0 = time.time()
    out = runMD(params)
    out['meta']['wall_time'] = time.time() - t0

    with open(out_file, 'w') as f:
        json.dump(out, f)
    print(f"[INFO] {out['meta']['n_atoms']} atoms, {params['n_steps']} steps in {out['meta']['wall_time']:.1f} s")
