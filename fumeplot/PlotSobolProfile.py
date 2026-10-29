"""
Plot Sobol indices of 1D QoIs (time series, spatial profiles, spectra).

Representations:
  profile   three panels on a shared axis:
              (a) ensemble mean +- std of the QoI,
              (b) partition of unity: stacked first-order shares, pairwise and
                  higher-order interactions (sums to 1 wherever variance > 0),
              (c) the same partition scaled by the variance, S_k * Var[Y](x),
                  showing where along the axis each input actually matters.
            Points with zero variance (indices undefined) are shaded.
  compare   partitions of unity of several QoIs sharing an axis, stacked in
            rows, to compare which inputs drive which QoI and when.
  lines     first (solid) / total (dashed) order per parameter, via the
            existing PlotSobolByTimestep.plotSobolLines kernel.

    python3 PlotSobolProfile.py <sobols.yml> [--plot_type all] [--qois ...]
"""

import os
import argparse
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from PlotSobolByTimestep import plotSobolLines
from SobolPartition import (LoadSobolSet, Partition, PlotFolder, SaveFigure, ComponentLegendHandles,
                            INK, INK_SECONDARY, MASK_COLOUR)


def _runs(mask):
    """(start, stop) index pairs of contiguous True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def shadeMasked(ax, x, mask, label=True):
    for i0, i1 in _runs(mask):
        lo = x[i0] - (x[1] - x[0]) / 2 if len(x) > 1 else x[i0]
        hi = x[i1 - 1] + (x[1] - x[0]) / 2 if len(x) > 1 else x[i1 - 1]
        ax.axvspan(lo, hi, color=MASK_COLOUR, zorder=0, linewidth=0)
        if label and (i1 - i0) > 0.08 * len(x):
            ax.text((lo + hi) / 2, 0.5, 'no variance', transform=ax.get_xaxis_transform(),
                    ha='center', va='center', rotation=90, fontsize='x-small', color=INK_SECONDARY)


def stackPartition(ax, x, comps, mask, scale=None):
    """Stacked areas of the partition components; `scale` multiplies all (e.g. variance)."""
    valid = ~mask
    base = np.zeros(len(x))
    for c in comps:
        v = np.nan_to_num(c.values) * (1.0 if scale is None else scale)
        if not np.any(v[valid] > 1e-12):
            continue
        ax.fill_between(x, base, base + v, where=valid, color=c.colour, hatch=c.hatch,
                        edgecolor='white', linewidth=0.8, step=None)
        base = base + v
    shadeMasked(ax, x, mask, label=scale is None)


def plotPartitionProfile(sset, qoi, plot_folder, combine_plots_pdf=None):
    axis_name, x = next(iter(qoi.axes.items()))
    comps, mask = Partition(sset, qoi)
    has_stats = qoi.mean is not None and qoi.std is not None

    nrows = 3 if has_stats else 1
    fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=(7.5, 2.2 * nrows + 0.6),
                             gridspec_kw={'height_ratios': [1, 1.3, 1][:nrows]}, squeeze=False)
    axes = axes[:, 0]

    if has_stats:
        ax = axes[0]
        ax.fill_between(x, qoi.mean - qoi.std, qoi.mean + qoi.std, color=INK_SECONDARY, alpha=0.18,
                        linewidth=0, label=r'mean $\pm$ std')
        ax.plot(x, qoi.mean, color=INK, linewidth=2, label='mean')
        ax.set_ylabel(sset.label(qoi.name), fontsize='small')
        ax.legend(loc='best', fontsize='x-small', frameon=False)
        ax.set_title('(a) Ensemble mean and spread', fontsize='small', loc='left')

    ax = axes[1] if has_stats else axes[0]
    stackPartition(ax, x, comps, mask)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel('Share of variance', fontsize='small')
    ax.set_title(('(b) ' if has_stats else '') + 'Partition of unity', fontsize='small', loc='left')

    if has_stats:
        ax = axes[2]
        stackPartition(ax, x, comps, mask, scale=qoi.std**2)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r'Var$[Y]\cdot S_k$', fontsize='small')
        ax.set_title('(c) Partition of variance', fontsize='small', loc='left')

    for ax in axes:
        ax.grid(visible=True, linestyle='-', linewidth=0.33)
        ax.set_axisbelow(True)
        ax.margins(x=0)
    axes[-1].set_xlabel(sset.label(axis_name))

    shown = [c for c in comps if np.nanmax(np.nan_to_num(c.values)) > 1e-12]
    fig.legend(handles=ComponentLegendHandles(shown), loc='upper left', bbox_to_anchor=(1.0, 0.95),
               frameon=False, fontsize='small')
    fig.suptitle(sset.label(qoi.name), x=0.02, ha='left')
    fig.text(0.02, 0.005, sset.describeDesign(), fontsize='x-small', color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    SaveFigure(fig, f"{qoi.name}_Sobol_profile", plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotPartitionCompare(sset, qois, plot_folder, combine_plots_pdf=None):
    """One partition-of-unity row per QoI, for QoIs sharing the same axis."""
    groups = {}
    for q in qois:
        groups.setdefault(next(iter(q.axes)), []).append(q)
    for axis_name, group in groups.items():
        if len(group) < 2:
            continue
        fig, axes = plt.subplots(len(group), 1, sharex=True, figsize=(7.5, 1.7 * len(group) + 0.9), squeeze=False)
        shown = {}
        for ax, qoi in zip(axes[:, 0], group):
            x = qoi.axes[axis_name]
            comps, mask = Partition(sset, qoi)
            stackPartition(ax, x, comps, mask)
            ax.set_ylim(0, 1)
            ax.margins(x=0)
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            ax.set_title(sset.label(qoi.name), fontsize='small', loc='left')
            ax.grid(visible=True, linestyle='-', linewidth=0.33)
            ax.set_axisbelow(True)
            for c in comps:
                if np.nanmax(np.nan_to_num(c.values)) > 1e-12:
                    shown.setdefault(c.key, c)
        axes[-1, 0].set_xlabel(sset.label(axis_name))
        fig.supylabel('Share of variance', fontsize='small')
        fig.legend(handles=ComponentLegendHandles(shown.values()), loc='upper left', bbox_to_anchor=(1.0, 0.95),
                   frameon=False, fontsize='small')
        fig.text(0.02, 0.005, sset.describeDesign(), fontsize='x-small', color=INK_SECONDARY)
        fig.tight_layout(rect=(0, 0.02, 1, 1))

        SaveFigure(fig, f"Sobol_compare_{'_'.join(q.name for q in group)}", plot_folder, combine_plots_pdf)
        plt.close(fig)


def plotLinesKernel(sset, qoi, plot_num, plot_folder, combine_plots_pdf=None):
    """First/total order lines through the existing FUMEplot kernel, masked where variance is 0."""
    axis_name, x = next(iter(qoi.axes.items()))
    _, mask = Partition(sset, qoi)
    entry = {o: {p: np.where(mask, np.nan, v) for p, v in getattr(qoi, o).items()} for o in ('first', 'total')}
    plotSobolLines(qoi.name, entry, sset.params, x, sset.label(axis_name), plot_num,
                   save_fig=True, plot_folder=plot_folder, combine_plots_pdf=combine_plots_pdf)


def plotSobolProfile(code, sset, plot_type='all', plot_path='.', qois=None, combine_plots_pdf=True):
    qois = sset.byDim(1, qois)
    if not qois:
        print("[WARNING] no 1D QoIs to plot.")
        return
    plot_folder = PlotFolder(plot_path, code)
    with PdfPages(os.path.join(plot_folder, "combined_sobol_profile.pdf")) if combine_plots_pdf else nullcontext() as pdf:
        for i, qoi in enumerate(qois):
            if plot_type in ('profile', 'all'):
                plotPartitionProfile(sset, qoi, plot_folder, pdf)
            if plot_type in ('lines', 'all'):
                plotLinesKernel(sset, qoi, i, plot_folder, pdf)
        if plot_type in ('compare', 'all'):
            plotPartitionCompare(sset, qois, plot_folder, pdf)
    print(f"[INFO] 1D Sobol plots saved to {plot_folder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot Sobol indices of 1D QoIs")
    p.add_argument("sobols_yml")
    p.add_argument("--code", default="md")
    p.add_argument("--plot_type", default="all", choices=["profile", "compare", "lines", "all"])
    p.add_argument("--qois", nargs="*", default=None)
    p.add_argument("--param_order", nargs="*", default=None, help="parameter order (fixes colours)")
    p.add_argument("--plot_path", default=None, help="output root (default: next to sobols.yml)")
    args = p.parse_args()

    sset = LoadSobolSet(args.sobols_yml, args.param_order)
    plot_path = args.plot_path or os.path.dirname(os.path.abspath(args.sobols_yml))
    plotSobolProfile(args.code, sset, args.plot_type, plot_path, qois=args.qois)
