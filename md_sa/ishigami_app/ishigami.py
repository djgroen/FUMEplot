#!/usr/bin/env python3
"""
Ishigami test function as an EasyVVUQ application (same JSON-in / JSON-out
contract as md_app/lj_md.py), used to verify that the Sobol pipeline converges
to the analytic indices.

    Y = sin(x1) + a sin^2(x2) + b x3^4 sin(x1),    x_i ~ U(-pi, pi)

QoIs, all with closed-form Sobol indices (ishigami_analytic.py):

    0D  y      at the nominal (a, b)
    1D  y_a    over a grid of a, at the nominal b         axes: a
    2D  y_ab   over a grid of (b, a)                      axes: b (rows), a (columns)

Pure python (no numpy import) to keep the per-run start-up cost small.

    python3 ishigami.py input.json output.json
"""

import sys
import json
import math
import time


def ishigami(x1, x2, x3, a, b):
    return math.sin(x1) + a * math.sin(x2)**2 + b * x3**4 * math.sin(x1)


def grid(lo, hi, n):
    return [lo + (hi - lo) * k / (n - 1) for k in range(n)] if n > 1 else [lo]


if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else 'input.json'
    out_file = sys.argv[2] if len(sys.argv) > 2 else 'output.json'
    with open(in_file) as f:
        p = json.load(f)

    t0 = time.time()
    x = (float(p['x1']), float(p['x2']), float(p['x3']))
    a_grid = grid(float(p['a_min']), float(p['a_max']), int(p['n_a']))
    b_grid = grid(float(p['b_min']), float(p['b_max']), int(p['n_b']))

    out = {
        'y': ishigami(*x, float(p['a']), float(p['b'])),
        'y_a': [ishigami(*x, a, float(p['b'])) for a in a_grid],
        'y_ab': [[ishigami(*x, a, b) for a in a_grid] for b in b_grid],
        'axes': {'y_a': {'a': a_grid}, 'y_ab': {'b': b_grid, 'a': a_grid}},
    }
    out['meta'] = {'wall_time': time.time() - t0}

    with open(out_file, 'w') as f:
        json.dump(out, f)
