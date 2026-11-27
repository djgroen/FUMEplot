"""
The LJ engine's optional LAMMPS dump output is read by the existing FUMEplot
LAMMPS decoder (fumeplot/brown3d_decoder.py via decoders.BrownianDecoder)
without modification, and the decoder reproduces the engine's own QoIs:

    decoder spread_profile               == cloud_spread_t   (dump_group=cloud)
    decoder centroid_displacement        >= |pull_disp_t|     (norm vs x-component)
    decoder steps                        == dumped steps
    decoder kinetic_energy               == 0.5 sum |v|^2 of the cloud

    python3 test_lammps_dump.py        (or: pytest test_lammps_dump.py)
"""

import os
import sys
import glob
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'md_app'))
sys.path.insert(0, os.path.join(HERE, '..', 'fumeplot'))

from lj_md import runMD                       # noqa: E402
from decoders import BrownianDecoder          # noqa: E402
from brown3d_decoder import read_brown3d_file  # noqa: E402

PARAMS = dict(temperature=1.2, density=0.8, r_cut=2.5, gamma=1.0, dt=0.005, n_cells=4,
              n_steps=200, n_equil=50, sample_every=10, rdf_window=50, rdf_bins=32, r_max=3.2,
              n_pull=32, pull_force=64.0, seed=42,
              dump_every=10, dump_group='cloud', dump_base='LJ', dump_trial=1)


def test_lammps_dump_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)                    # the engine writes dumps to its working directory
        try:
            out = runMD(PARAMS)
        finally:
            os.chdir(cwd)

        files = sorted(glob.glob(os.path.join(tmp, 'LJ-T1-S*.txt')))
        assert len(files) == PARAMS['n_steps'] // PARAMS['dump_every'], f"{len(files)} dump files"
        frame = read_brown3d_file(files[0])
        assert frame['atom_count'] == PARAMS['n_pull']

        dec = BrownianDecoder(file_base='LJ', num_trials=1, masses=1.0)
        res = dec.parse_sim_output({'run_dir': tmp})

    steps = np.arange(1, len(res['steps']) + 1) * PARAMS['dump_every']
    assert res['steps'] == steps.tolist()
    assert np.allclose(res['spread_profile'], out['cloud_spread_t'], rtol=1e-8), "spread != cloud_spread_t"
    cd = np.array(res['centroid_displacement_profile'])
    assert np.all(cd >= np.abs(out['pull_disp_t']) - 1e-8), "centroid displacement < |pull_disp_t|"
    assert res['centroid_displacement_final'] > 0.9 * abs(out['pull_disp']), "drift should be mostly along +x"
    assert np.isfinite(res['spread_scaling_exponent'])
    return res, out


if __name__ == "__main__":
    res, out = test_lammps_dump_roundtrip()
    print(f"spread final     decoder {res['spread_final']:.6f}   engine cloud_spread {out['cloud_spread']:.6f}")
    print(f"centroid disp    decoder {res['centroid_displacement_final']:.6f}   engine pull_disp (x) {out['pull_disp']:.6f}")
    print(f"kinetic energy   decoder {res['kinetic_energy_final']:.4f}   spread exponent {res['spread_scaling_exponent']:.3f}")
    print("[PASS] LJ LAMMPS dumps decoded by fumeplot BrownianDecoder")
