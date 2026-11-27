"""
Plot Sobol indices of scalar (0D) QoIs.

Representations:
  partition   one 100% bar per QoI: first-order shares per parameter, pairwise
              and higher-order interactions; the bar length is the whole
              output variance (partition of unity), so QoIs are comparable.
  dumbbell    first-order (filled) vs total-order (hollow) index per parameter;
              the gap S_T - S_1 is the parameter's involvement in interactions.
  matrix      first order on the diagonal, second order S_ij off the diagonal.

    python3 PlotSobolScalar.py <sobols.yml> [--plot_type all] [--qois ...]
"""

import os
import argparse
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

from SobolPartition import (LoadSobolSet, Partition, PlotFolder, SaveFigure, ComponentLegendHandles,
                            InkOn, SEQUENTIAL_BLUE, INK, INK_SECONDARY)


def plotPartitionBars(sset, qois, plot_folder, combine_plots_pdf=None, label_min=0.06):
    """Horizontal 100% stacked bars, one per scalar QoI."""
    fig, ax = plt.subplots(figsize=(9.5, 0.6 + 0.55 * len(qois)))

    legend_comps = {}
    for row, qoi in enumerate(qois):
        if qoi.mean is not None and qoi.std is not None:
            mu, sd = float(qoi.mean), float(qoi.std)
            cv = f"  CV {abs(sd / mu):.1%}" if mu != 0 else ''
            ax.text(1.02, row, f"{mu:.4g} $\\pm$ {sd:.3g}{cv}", ha='left', va='center',
                    fontsize='x-small', color=INK_SECONDARY, clip_on=False)
        comps, mask = Partition(sset, qoi)
        if mask.all():
            ax.text(0.5, row, 'zero variance', ha='center', va='center', color=INK_SECONDARY)
            continue
        left = 0.0
        for c in comps:
            w = float(c.values)
            if w <= 1e-3:
                continue
            ax.barh(row, w, left=left, height=0.62, color=c.colour, hatch=c.hatch,
                    edgecolor='white', linewidth=2)
            if w >= label_min:
                ax.text(left + w / 2, row, f"{w:.0%}", ha='center', va='center',
                        fontsize='x-small', color=InkOn(c.colour))
            left += w
            legend_comps.setdefault(c.key, c)

    ax.set_yticks(range(len(qois)), [sset.label(q.name) for q in qois])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel('Share of output variance')
    if any(q.std is not None for q in qois):
        ax.text(1.02, -0.6, 'mean $\\pm$ std', ha='left', va='bottom', fontsize='x-small',
                fontweight='bold', color=INK_SECONDARY, clip_on=False)
    ax.grid(visible=True, axis='x', linestyle='-', linewidth=0.33)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.legend(handles=ComponentLegendHandles(legend_comps.values()), loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=min(len(legend_comps), 3), frameon=False, fontsize='small')
    ax.set_title(sset.describeDesign(), fontsize='x-small', color=INK_SECONDARY, loc='left', pad=48)

    SaveFigure(fig, 'Sobol_scalar_partition', plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotFirstTotal(sset, qois, plot_folder, combine_plots_pdf=None, ncols=3):
    """Dumbbells: S_1 (filled) to S_T (hollow), one panel per QoI."""
    params = sset.params
    ncols = min(ncols, len(qois))
    nrows = int(np.ceil(len(qois) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 0.5 + 0.42 * len(params) * nrows),
                             sharex=True, sharey=True, squeeze=False)

    y = np.arange(len(params))
    for ax, qoi in zip(axes.flat, qois):
        for i, p in enumerate(params):
            s1 = float(np.clip(qoi.first.get(p, np.nan), 0, 1))
            st = float(np.clip(qoi.total.get(p, np.nan), 0, 1))
            ax.plot([s1, st], [i, i], color=INK_SECONDARY, linewidth=2, zorder=1)
            ax.scatter([st], [i], s=72, facecolor='white', edgecolor=sset.colour(p), linewidth=2, zorder=2)
            ax.scatter([s1], [i], s=30, color=sset.colour(p), edgecolor='white', linewidth=1, zorder=3)
        ax.set_title(sset.label(qoi.name), fontsize='small')
        ax.set_xlim(-0.03, 1.03)
        ax.grid(visible=True, axis='x', linestyle='-', linewidth=0.33)
        ax.set_axisbelow(True)
    for k in range(len(qois), nrows * ncols):
        axes.flat[k].set_visible(False)

    axes[0, 0].set_yticks(y, [sset.label(p) for p in params])
    axes[0, 0].invert_yaxis()
    handles = [mlines.Line2D([], [], marker='o', linestyle='', color=INK, markersize=7, label='First order $S_i$'),
               mlines.Line2D([], [], marker='o', linestyle='', markerfacecolor='white', markeredgecolor=INK,
                             markersize=7, label='Total order $S_{T,i}$')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize='small')
    fig.supxlabel('Sobol index')
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    SaveFigure(fig, 'Sobol_scalar_first_total', plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotInteractionMatrix(sset, qois, plot_folder, combine_plots_pdf=None, ncols=3):
    """Lower-triangular matrix: S_i on the diagonal, S_ij below it."""
    qois = [q for q in qois if q.second]
    if not qois:
        print("[INFO] no second-order indices: skipping interaction matrices.")
        return
    params = sset.params
    n = len(params)
    ncols = min(ncols, len(qois))
    nrows = int(np.ceil(len(qois) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols + 0.8, 3.2 * nrows), squeeze=False)

    for ax, qoi in zip(axes.flat, qois):
        m = np.full((n, n), np.nan)
        for i, p in enumerate(params):
            m[i, i] = qoi.first.get(p, np.nan)
        for (p, q), v in qoi.second.items():
            i, j = sorted((params.index(p), params.index(q)), reverse=True)
            m[i, j] = v
        cmap = SEQUENTIAL_BLUE.copy()
        cmap.set_bad(alpha=0)       # upper triangle: not applicable
        im = ax.imshow(np.ma.masked_invalid(np.clip(m, 0, 1)), cmap=cmap, vmin=0, vmax=1)
        for i in range(n):
            for j in range(i + 1):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha='center', va='center', fontsize='x-small',
                            color=INK if m[i, j] < 0.5 else '#ffffff')
        ax.set_xticks(range(n), [sset.label(p) for p in params], rotation=45, ha='right', fontsize='x-small')
        ax.set_yticks(range(n), [sset.label(p) for p in params] if ax in axes[:, 0] else [], fontsize='x-small')
        ax.set_title(sset.label(qoi.name), fontsize='small')
        for s in ax.spines.values():
            s.set_visible(False)
    for k in range(len(qois), nrows * ncols):
        axes.flat[k].set_visible(False)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Sobol index ($S_i$ diagonal, $S_{ij}$ below)')
    SaveFigure(fig, 'Sobol_scalar_interaction_matrix', plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotSobolScalar(code, sset, plot_type='all', plot_path='.', qois=None, combine_plots_pdf=True):
    qois = sset.byDim(0, qois)
    if not qois:
        print("[WARNING] no scalar QoIs to plot.")
        return
    plot_folder = PlotFolder(plot_path, code)
    with PdfPages(os.path.join(plot_folder, "combined_sobol_scalar.pdf")) if combine_plots_pdf else nullcontext() as pdf:
        if plot_type in ('partition', 'all'):
            plotPartitionBars(sset, qois, plot_folder, pdf)
        if plot_type in ('dumbbell', 'all'):
            plotFirstTotal(sset, qois, plot_folder, pdf)
        if plot_type in ('matrix', 'all'):
            plotInteractionMatrix(sset, qois, plot_folder, pdf)
    print(f"[INFO] scalar Sobol plots saved to {plot_folder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot Sobol indices of scalar QoIs")
    p.add_argument("sobols_yml")
    p.add_argument("--code", default="md")
    p.add_argument("--plot_type", default="all", choices=["partition", "dumbbell", "matrix", "all"])
    p.add_argument("--qois", nargs="*", default=None)
    p.add_argument("--param_order", nargs="*", default=None, help="parameter order (fixes colours)")
    p.add_argument("--plot_path", default=None, help="output root (default: next to sobols.yml)")
    args = p.parse_args()

    sset = LoadSobolSet(args.sobols_yml, args.param_order)
    plot_path = args.plot_path or os.path.dirname(os.path.abspath(args.sobols_yml))
    plotSobolScalar(args.code, sset, args.plot_type, plot_path, qois=args.qois)
