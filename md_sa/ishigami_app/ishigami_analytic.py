"""
Closed-form statistics and Sobol indices of the Ishigami function
Y = sin(x1) + a sin^2(x2) + b x3^4 sin(x1), x_i ~ U(-pi, pi):

    E[Y]  = a/2
    V     = a^2/8 + b pi^4/5 + b^2 pi^8/18 + 1/2
    V_1   = (1 + b pi^4/5)^2 / 2,   V_2 = a^2/8,   V_3 = 0
    V_13  = 8 b^2 pi^8 / 225,       V_12 = V_23 = V_123 = 0

All functions broadcast over arrays of a and b.
"""

import numpy as np
import yaml

PARAMS = ['x1', 'x2', 'x3']
PAIRS = [('x1', 'x2'), ('x1', 'x3'), ('x2', 'x3')]


def AnalyticIndices(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    pi4, pi8 = np.pi**4, np.pi**8
    v1 = 0.5 * (1 + b * pi4 / 5)**2
    v2 = a**2 / 8
    v13 = 8 * b**2 * pi8 / 225
    var = v1 + v2 + v13
    zero = np.zeros_like(var)
    return {
        'mean': a / 2,
        'var': var,
        'std': np.sqrt(var),
        'first': {'x1': v1 / var, 'x2': v2 / var, 'x3': zero},
        'second': {('x1', 'x2'): zero, ('x1', 'x3'): v13 / var, ('x2', 'x3'): zero},
        'total': {'x1': (v1 + v13) / var, 'x2': v2 / var, 'x3': v13 / var},
    }


def AnalyticForLayout(layout, a, b):
    """Analytic indices for every QoI of an exported qoi_layout (y, y_a, y_ab)."""
    out = {}
    for qoi, lay in layout.items():
        axes = lay.get('axes') or {}
        if qoi == 'y':
            out[qoi] = AnalyticIndices(a, b)
        elif qoi == 'y_a':
            out[qoi] = AnalyticIndices(np.asarray(axes['a']), b)
        elif qoi == 'y_ab':
            bb, aa = np.meshgrid(axes['b'], axes['a'], indexing='ij')
            out[qoi] = AnalyticIndices(aa, bb)
    return out


def _toList(v):
    return np.asarray(v, dtype=float).ravel().tolist()


def WriteAnalyticSobols(path, campaign_info, a, b):
    """Analytic indices in the sobols.yml layout, so FUMEplot renders the reference too."""
    info = dict(campaign_info)
    info.update({'name': 'ishigami_analytic', 'analysis': 'analytic',
                 'sampler': {'type': 'analytic', 'n_samples': 0}})
    exact = AnalyticForLayout(info['qoi_layout'], a, b)
    report = {'campaign_info': info, 'qoi_statistics': {}}
    report.update({p: {} for p in PARAMS})
    for qoi, ex in exact.items():
        report['qoi_statistics'][qoi] = {'mean': _toList(ex['mean']), 'std': _toList(ex['std'])}
        for p in PARAMS:
            report[p][f"{qoi}_sobols_first"] = _toList(ex['first'][p])
            report[p][f"{qoi}_sobols_total"] = _toList(ex['total'][p])
            report[p][f"{qoi}_sobols_second"] = {q: _toList(ex['second'][tuple(sorted((p, q)))])
                                                 for q in PARAMS if q != p}
    with open(path, 'w') as f:
        yaml.safe_dump(report, f, sort_keys=False, default_flow_style=None, width=120)
