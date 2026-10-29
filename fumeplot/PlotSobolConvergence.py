"""
Plots for the convergence comparison of Sobol-index methods
(output directory of md_sa/run_convergence.py).

  convergence_error_<qoi>      error vs number of runs, one panel per statistic
                               (S_1, S_T, S_ij: worst index and grid point;
                               mean, std: relative RMS), one line per method
  convergence_estimates_<qoi>  (0D QoIs) estimate vs runs per index, with the
                               reference value as a dashed line
  convergence_increment        reference-free indicator |S_k - S_{k-1}| next to
                               the error against the reference, per method
  convergence_cost             worst Sobol error vs runs, wall time per level
                               and projected MD compute
  error_finest_<qoi>           errors of each method's finest level: bars (0D),
                               profiles (1D), maps (2D)
  <method>Plots/Sobol/...      partition-of-unity plots (PlotSobolScalar /
                               Profile / Field) of the reference and of every
                               method's finest level

    python3 PlotSobolConvergence.py <convergence_dir>
"""

import os
import argparse

import numpy as np
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

from PlotSobolByTimestep import PARAM_COLOURS
from SobolPartition import LoadSobolSet, VarianceMask, SaveFigure, SEQUENTIAL_BLUE, INK, INK_SECONDARY, MASK_COLOUR
from PlotSobolScalar import plotSobolScalar
from PlotSobolProfile import plotSobolProfile
from PlotSobolField import plotSobolField

# methods take categorical slots 5-8 so they never share a colour with parameters (slots 1-4)
METHOD_COLOURS = PARAM_COLOURS[4:] + PARAM_COLOURS[:4]
METHOD_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
FLOOR = 1e-16
KIND_LABELS = {'S1': 'First order $S_i$', 'ST': 'Total order $S_{T,i}$', 'S2': 'Second order $S_{ij}$',
               'mean': 'Mean (relative)', 'std': 'Std (relative)'}


def _style(i):
    return dict(color=METHOD_COLOURS[i % len(METHOD_COLOURS)], marker=METHOD_MARKERS[i % len(METHOD_MARKERS)],
                markersize=5, linewidth=2)


def _methodHandles(methods):
    return [mlines.Line2D([], [], label=m, **_style(i)) for i, m in enumerate(methods)]


def _logAxes(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.33)
    ax.set_axisbelow(True)


def _guides(ax, x, y):
    """N^-1/2 and N^-1 reference slopes anchored at the top-left of the data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not len(x) or not np.isfinite(y).any():
        return
    xs = np.array([x.min(), x.max()])
    y0 = np.nanmax(y)
    for rate, style in ((0.5, ':'), (1.0, '--')):
        ax.plot(xs, y0 * (xs / xs[0])**-rate, color=INK_SECONDARY, linestyle=style, linewidth=1)
        ax.annotate(f"$N^{{-{rate:g}}}$", (xs[1], y0 * (xs[1] / xs[0])**-rate), fontsize='x-small',
                    color=INK_SECONDARY, ha='left', va='center')


def _worst(df, kinds, column):
    """Worst value over the selected quantities: median over replicates, with lo/hi range."""
    d = df[df.quantity.isin(kinds)]
    if 'replicate' not in d:
        d = d.assign(replicate=0)
    per_rep = d.groupby(['method', 'level', 'n_runs', 'replicate'], sort=False)[column].max().reset_index()
    g = per_rep.groupby(['method', 'level', 'n_runs'], sort=False)[column]
    return g.median().reset_index().assign(lo=g.min().values, hi=g.max().values)


def _line(ax, r, x, y, i, **kw):
    """Median line with the replicate range as a band (no band for deterministic designs)."""
    st = {**_style(i), **kw}
    ax.plot(r[x], np.maximum(r[y], FLOOR), **st)
    if 'lo' in r and (r.hi > r.lo).any():
        ax.fill_between(r[x], np.maximum(r.lo, FLOOR), np.maximum(r.hi, FLOOR), color=st['color'],
                        alpha=0.15, linewidth=0)


def plotErrorConvergence(df, methods, qoi, folder, pdf):
    d = df[df.qoi == qoi]
    kinds = [k for k in KIND_LABELS if k in set(d.quantity)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.0 * len(kinds), 3.1), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        column = 'err_rel' if kind in ('mean', 'std') else 'err_max'
        w = _worst(d, [kind], column)
        for i, m in enumerate(methods):
            r = w[w.method == m]
            if len(r):
                _line(ax, r, 'n_runs', column, i)
        _logAxes(ax)
        _guides(ax, w.n_runs, np.maximum(w[column], FLOOR))
        ax.set_title(KIND_LABELS[kind], fontsize='small')
        ax.set_xlabel('Runs $N$', fontsize='small')
    axes[0, 0].set_ylabel('Error vs reference', fontsize='small')
    fig.legend(handles=_methodHandles(methods), loc='upper center', ncol=len(methods), frameon=False,
               fontsize='small', bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"QoI {qoi}: worst index / grid point (Sobol), relative RMS (moments); "
                 f"band = range over replicates",
                 fontsize='small', y=-0.02, color=INK_SECONDARY)
    fig.tight_layout()
    SaveFigure(fig, f"convergence_error_{qoi}", folder, pdf)
    plt.close(fig)


def plotEstimates(df, methods, qoi, folder, pdf, ncols=4):
    d = df[(df.qoi == qoi) & df.quantity.isin(['S1', 'ST', 'S2'])]
    keys = list(dict.fromkeys(zip(d.quantity, d['index'])))
    # skip indices that are zero in the reference and in every estimate (round-off panels)
    keys = [(k, i) for k, i in keys
            if np.nanmax(np.abs(d[(d.quantity == k) & (d['index'] == i)][['estimate', 'reference']].values)) > 1e-8]
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), sharex=True, squeeze=False)
    for ax, (kind, index) in zip(axes.flat, keys):
        e = d[(d.quantity == kind) & (d['index'] == index)]
        for i, m in enumerate(methods):
            g = e[e.method == m].groupby('n_runs').estimate
            if len(e[e.method == m]):
                ax.plot(g.median().index, g.median().values, **_style(i))
                if (g.max() > g.min()).any():
                    ax.fill_between(g.median().index, g.min().values, g.max().values,
                                    color=_style(i)['color'], alpha=0.15, linewidth=0)
        ax.axhline(e.reference.iloc[0], color=INK, linestyle='--', linewidth=1.2)
        ax.set_xscale('log')
        ax.grid(visible=True, linestyle='-', linewidth=0.33)
        ax.set_title(f"{KIND_LABELS[kind].split(' $')[0]}  {index}", fontsize='small')
    for k in range(len(keys), nrows * ncols):
        axes.flat[k].set_visible(False)
    fig.supxlabel('Runs $N$', fontsize='small')
    fig.supylabel('Sobol index estimate', fontsize='small')
    handles = _methodHandles(methods) + [mlines.Line2D([], [], color=INK, linestyle='--', label='Reference')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, fontsize='small',
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    SaveFigure(fig, f"convergence_estimates_{qoi}", folder, pdf)
    plt.close(fig)


def plotIncrement(df, methods, folder, pdf):
    err = _worst(df, ['S1', 'ST', 'S2'], 'err_max')
    inc = _worst(df, ['S1', 'ST', 'S2'], 'increment')
    fig, axes = plt.subplots(1, len(methods), figsize=(3.0 * len(methods), 3.0), sharey=True, squeeze=False)
    for i, (ax, m) in enumerate(zip(axes[0], methods)):
        _line(ax, err[err.method == m], 'n_runs', 'err_max', i)
        _line(ax, inc[(inc.method == m) & inc.increment.notna()], 'n_runs', 'increment', i,
              linestyle='--', markerfacecolor='white')
        _logAxes(ax)
        ax.set_title(m, fontsize='small')
        ax.set_xlabel('Runs $N$', fontsize='small')
    axes[0, 0].set_ylabel('Worst Sobol-index difference', fontsize='small')
    handles = [mlines.Line2D([], [], color=INK, label='vs reference'),
               mlines.Line2D([], [], color=INK, linestyle='--', marker='o', markerfacecolor='white',
                             label='vs previous level (no reference needed)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize='small', bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    SaveFigure(fig, "convergence_increment", folder, pdf)
    plt.close(fig)


def plotCost(df, methods, summary, folder, pdf):
    w = _worst(df, ['S1', 'ST', 'S2'], 'err_max')
    t = df.groupby(['method', 'level'], sort=False)[['t_execute', 't_analyse']].first().reset_index()
    w = w.merge(t, on=['method', 'level'])
    run_cost = None
    for m in summary['methods'].values():
        if m.get('projected_md_cpu_h_finest') and m['runs_finest']:
            run_cost = m['projected_md_cpu_h_finest'] / m['runs_finest']
    panels = [('n_runs', 'Runs $N$'), ('wall', 'Wall time per level [s] (run + analysis)')]
    if run_cost:
        panels.append(('md', 'Projected MD compute [CPU-h]'))
    w['wall'] = w.t_execute + w.t_analyse
    w['md'] = w.n_runs * (run_cost or 0)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.1), sharey=True, squeeze=False)
    for ax, (col, label) in zip(axes[0], panels):
        for i, m in enumerate(methods):
            _line(ax, w[w.method == m], col, 'err_max', i)
        _logAxes(ax)
        ax.set_xlabel(label, fontsize='small')
    axes[0, 0].set_ylabel('Worst Sobol-index error', fontsize='small')
    fig.legend(handles=_methodHandles(methods), loc='upper center', ncol=len(methods), frameon=False,
               fontsize='small', bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    SaveFigure(fig, "convergence_cost", folder, pdf)
    plt.close(fig)


def plotFinestErrors(sets, ref, methods, folder, pdf):
    params = ref.params
    for name, qref in ref.qois.items():
        valid = ~VarianceMask(qref)
        if qref.ndim == 0:
            keys = [('S1', p) for p in params] + [('ST', p) for p in params] + [('S2', pq) for pq in qref.second]
            fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(keys) * len(methods) / 2, 3.2))
            width = 0.8 / len(methods)
            for i, m in enumerate(methods):
                q = sets[m].qois[name]
                errs = []
                for kind, k in keys:
                    est = {'S1': q.first, 'ST': q.total, 'S2': q.second}[kind].get(k)
                    true = {'S1': qref.first, 'ST': qref.total, 'S2': qref.second}[kind][k]
                    errs.append(abs(float(est - true)) if est is not None else np.nan)
                ax.bar(np.arange(len(keys)) + (i - (len(methods) - 1) / 2) * width, np.maximum(errs, FLOOR),
                       width=width, color=_style(i)['color'], edgecolor='white', linewidth=1, label=m)
            ax.set_yscale('log')
            ax.set_xticks(range(len(keys)), [f"{k}\n{' x '.join(i) if isinstance(i, tuple) else i}" for k, i in keys],
                          fontsize='x-small')
            ax.set_ylabel('|error| at finest level', fontsize='small')
            ax.grid(visible=True, axis='y', linestyle='-', linewidth=0.33)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize='x-small', ncol=len(methods), loc='upper center', bbox_to_anchor=(0.5, 1.15))
            ax.set_title('missing bar = index not provided by the method', fontsize='x-small',
                         color=INK_SECONDARY, loc='right', pad=24)

        elif qref.ndim == 1:
            axis_name, x = next(iter(qref.axes.items()))
            fig, axes = plt.subplots(2, len(params), figsize=(2.9 * len(params), 4.6), sharex=True, sharey=True,
                                     squeeze=False)
            for row, (kind, attr) in enumerate((('S1', 'first'), ('ST', 'total'))):
                for col, p in enumerate(params):
                    ax = axes[row, col]
                    for i, m in enumerate(methods):
                        est = getattr(sets[m].qois[name], attr)[p]
                        err = np.where(valid, np.abs(est - getattr(qref, attr)[p]), np.nan)
                        ax.plot(x, np.maximum(err, FLOOR), **{**_style(i), 'markersize': 3})
                    ax.set_yscale('log')
                    ax.grid(visible=True, linestyle='-', linewidth=0.33)
                    ax.set_title(f"{KIND_LABELS[kind].split(' $')[0]}  {ref.label(p)}", fontsize='small')
                    if row == 1:
                        ax.set_xlabel(ref.label(axis_name), fontsize='small')
            fig.supylabel('|error| at finest level', fontsize='small')
            fig.legend(handles=_methodHandles(methods), loc='upper center', ncol=len(methods), frameon=False,
                       fontsize='small', bbox_to_anchor=(0.5, 1.05))
            fig.tight_layout()

        else:
            (y_name, y), (x_name, x) = list(qref.axes.items())[:2]
            fig, axes = plt.subplots(len(methods), len(params), figsize=(2.8 * len(params) + 1.2, 2.3 * len(methods)),
                                     sharex=True, sharey=True, squeeze=False)
            norm = matplotlib.colors.LogNorm(vmin=1e-6, vmax=1)
            cmap = SEQUENTIAL_BLUE.copy()
            cmap.set_bad(MASK_COLOUR)
            for i, m in enumerate(methods):
                for j, p in enumerate(params):
                    err = np.abs(sets[m].qois[name].total[p] - qref.total[p])
                    err = np.ma.masked_where(~valid, np.clip(err, 1e-6, 1))
                    im = axes[i, j].pcolormesh(x, y, err, cmap=cmap, norm=norm, shading='nearest')
                    if i == 0:
                        axes[i, j].set_title(f"Total order  {ref.label(p)}", fontsize='small')
                    if j == 0:
                        axes[i, j].set_ylabel(f"{m}\n{ref.label(y_name)}", fontsize='small')
                    if i == len(methods) - 1:
                        axes[i, j].set_xlabel(ref.label(x_name), fontsize='small')
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='|error| at finest level')

        fig.suptitle(f"{ref.label(name)}: error of the finest level per method", x=0.02, ha='left', fontsize='small')
        SaveFigure(fig, f"error_finest_{name}", folder, pdf)
        plt.close(fig)


def plotSobolConvergence(conv_dir):
    df = pd.read_csv(os.path.join(conv_dir, 'convergence.csv'))
    with open(os.path.join(conv_dir, 'summary.yml')) as f:
        summary = yaml.safe_load(f)
    methods = list(summary['methods'])
    folder = os.path.join(conv_dir, 'plots')
    os.makedirs(folder, exist_ok=True)

    ref = LoadSobolSet(os.path.join(conv_dir, 'sobols', 'reference.yml'))
    finest = {m: os.path.join(conv_dir, 'sobols', m, f"{summary['methods'][m]['levels'][-1]}.yml") for m in methods}
    sets = {m: LoadSobolSet(p, ref.params) for m, p in finest.items()}

    with PdfPages(os.path.join(folder, 'combined_convergence.pdf')) as pdf:
        for qoi in df.qoi.unique():
            plotErrorConvergence(df, methods, qoi, folder, pdf)
            if df[df.qoi == qoi]['ndim'].iloc[0] == 0:
                plotEstimates(df, methods, qoi, folder, pdf)
        plotIncrement(df, methods, folder, pdf)
        plotCost(df, methods, summary, folder, pdf)
        plotFinestErrors(sets, ref, methods, folder, pdf)

    for label, path in [('reference', os.path.join(conv_dir, 'sobols', 'reference.yml'))] + list(finest.items()):
        sset = LoadSobolSet(path, ref.params)
        for plot in (plotSobolScalar, plotSobolProfile, plotSobolField):
            plot(label, sset, plot_path=folder)
    print(f"[INFO] convergence plots saved to {folder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot a convergence comparison of Sobol-index methods")
    p.add_argument("conv_dir", help="output directory of md_sa/run_convergence.py")
    args = p.parse_args()
    plotSobolConvergence(args.conv_dir)
