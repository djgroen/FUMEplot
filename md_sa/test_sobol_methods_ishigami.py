"""
Verification test: convergence of four Sobol-index methods on the Ishigami
function, against its analytic indices.

Runs run_convergence.py on ishigami.yml (PCE-Gauss, SC-Gauss, SC-sparse-CC,
QMC-Saltelli; a few minutes locally), then checks that

  * every method delivers first- and total-order indices for the 0D, 1D and 2D QoIs,
  * every method's worst Sobol error decreases from its first to its finest level,
  * methods with a `verify.max_error` in ishigami.yml reach it for QoI y,
  * the summary, convergence and error plots, and the Sobol plots per method
    are produced.

Outputs stay in campaigns_ishigami/convergence_test/ for inspection
(summary.md, plots/).

    python3 test_sobol_methods_ishigami.py        (or: pytest test_sobol_methods_ishigami.py)
"""

import os
import sys
import shutil

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_convergence import RunConvergence   # noqa: E402

CONFIG = os.path.join(HERE, 'ishigami.yml')
OUT_DIR = os.path.join(HERE, 'campaigns_ishigami', 'convergence_test')
PLOTS = ['convergence_error_y.png', 'convergence_error_y_a.png', 'convergence_error_y_ab.png',
         'convergence_estimates_y.png', 'convergence_increment.png', 'convergence_cost.png',
         'error_finest_y.png', 'error_finest_y_a.png', 'error_finest_y_ab.png']


def test_sobol_methods_ishigami():
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    out_dir, df, summary = RunConvergence(CONFIG, out_dir=OUT_DIR)
    with open(CONFIG) as f:
        methods = yaml.safe_load(f)['methods']

    failures = []
    for m in methods:
        name = m['name']
        if name not in summary['methods']:
            failures.append(f"{name}: did not run")
            continue
        s = summary['methods'][name]
        d = df[df.method == name]

        for qoi in ('y', 'y_a', 'y_ab'):
            got = set(d[d.qoi == qoi].quantity)
            if not {'S1', 'ST'} <= got:
                failures.append(f"{name}/{qoi}: missing first/total order indices ({sorted(got)})")

        errors = s['max_sobol_error']
        if not errors[-1] < errors[0]:
            failures.append(f"{name}: worst Sobol error did not decrease ({errors[0]:.3g} -> {errors[-1]:.3g})")

        limit = (m.get('verify') or {}).get('max_error')
        if limit is not None:
            finest = d[(d.qoi == 'y') & (d.level == s['levels'][-1]) & d.quantity.isin(['S1', 'ST', 'S2'])]
            worst = finest.err_max.max()
            if not worst <= limit:
                failures.append(f"{name}: worst Sobol error on y at finest level {worst:.3g} > {limit}")

        sobol_dir = os.path.join(out_dir, 'plots', 'EnsemblePlots', f"{name}Plots", 'Sobol')
        if not os.path.isfile(os.path.join(sobol_dir, 'Sobol_scalar_partition.png')):
            failures.append(f"{name}: Sobol plots missing in {sobol_dir}")

    for f in PLOTS + ['combined_convergence.pdf']:
        if not os.path.isfile(os.path.join(out_dir, 'plots', f)):
            failures.append(f"missing plot {f}")
    for f in ('convergence.csv', 'summary.yml', 'summary.md', 'sobols/reference.yml'):
        if not os.path.isfile(os.path.join(out_dir, f)):
            failures.append(f"missing output {f}")

    with open(os.path.join(out_dir, 'summary.md')) as f:
        print('\n' + f.read())
    assert not failures, '\n'.join(failures)


if __name__ == "__main__":
    test_sobol_methods_ishigami()
    print(f"[PASS] four Sobol methods verified against the analytic Ishigami indices; outputs in {OUT_DIR}")
