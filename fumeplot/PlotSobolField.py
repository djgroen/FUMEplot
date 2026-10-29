"""
Plot Sobol indices of 2D QoIs (fields such as g(t, r), density maps, spectrograms).

Array axis 0 is drawn on the vertical axis, axis 1 on the horizontal axis.

Representations:
  components  small multiples, one heatmap per partition component (first
              order per parameter, pairwise, higher order) on a shared 0..1
              scale; at every pixel the panels sum to 1.
  dominant    categorical map of the largest component; lightness encodes its
              share (pale = no clear winner).
  mixed       every pixel coloured by the convex combination sum_k S_k c_k of
              the component colours (mixed in linear RGB): the partition of
              unity mapped directly onto colour. With --weight std, pixels fade
              to the surface where the output spread is small.
  slices      1D partitions of unity along axis 1 at a few axis-0 positions.

Zero-variance pixels (indices undefined) are drawn in a neutral grey.

    python3 PlotSobolField.py <sobols.yml> [--plot_type all] [--weight none|std]
"""

import os
import argparse
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from PlotSobolProfile import stackPartition
from SobolPartition import (LoadSobolSet, Partition, PlotFolder, SaveFigure, ComponentLegendHandles,
                            MixColours, FadeToSurface, SEQUENTIAL_BLUE, INK_SECONDARY, MASK_COLOUR)
import matplotlib.colors as mcolors


def _grid(qoi):
    (y_name, y), (x_name, x) = list(qoi.axes.items())[:2]
    return x_name, x, y_name, y


def _rgbImage(ax, x, y, rgb, mask):
    rgb = np.where(mask[..., None], np.array(mcolors.to_rgb(MASK_COLOUR)), rgb)
    ax.imshow(rgb, origin='lower', aspect='auto', interpolation='nearest',
              extent=(x[0] - (x[1] - x[0]) / 2, x[-1] + (x[1] - x[0]) / 2,
                      y[0] - (y[1] - y[0]) / 2, y[-1] + (y[1] - y[0]) / 2))


def _shown(comps):
    return [c for c in comps if np.nanmax(np.nan_to_num(c.values)) > 1e-6]


def plotComponentMaps(sset, qoi, plot_folder, combine_plots_pdf=None, ncols=3):
    x_name, x, y_name, y = _grid(qoi)
    comps, mask = Partition(sset, qoi)
    comps = _shown(comps)
    ncols = min(ncols, len(comps))
    nrows = int(np.ceil(len(comps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                             figsize=(3.3 * ncols + 1.0, 2.7 * nrows + 0.5), squeeze=False)
    cmap = SEQUENTIAL_BLUE.copy()
    cmap.set_bad(MASK_COLOUR)
    for ax, c in zip(axes.flat, comps):
        im = ax.pcolormesh(x, y, np.ma.masked_invalid(c.values), cmap=cmap, vmin=0, vmax=1, shading='nearest')
        ax.set_title(c.label, fontsize='small')
    for k in range(len(comps), nrows * ncols):
        axes.flat[k].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel(sset.label(x_name))
    for ax in axes[:, 0]:
        ax.set_ylabel(sset.label(y_name))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label='Share of variance')
    fig.suptitle(f"{sset.label(qoi.name)}: partition components (sum to 1 per pixel)", x=0.02, ha='left')

    SaveFigure(fig, f"{qoi.name}_Sobol_components", plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotDominantMap(sset, qoi, plot_folder, combine_plots_pdf=None):
    x_name, x, y_name, y = _grid(qoi)
    comps, mask = Partition(sset, qoi)
    stack = np.nan_to_num(np.array([c.values for c in comps]))
    winner = np.argmax(stack, axis=0)
    share = np.max(stack, axis=0)
    base = np.array([mcolors.to_rgb(c.colour) for c in comps])[winner]
    # share 1/K (no winner) -> surface, share 1 -> full colour
    k = len(comps)
    strength = (share - 1.0 / k) / (1.0 - 1.0 / k)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _rgbImage(ax, x, y, FadeToSurface(base, 0.25 + 0.75 * strength), mask)
    ax.set_xlabel(sset.label(x_name))
    ax.set_ylabel(sset.label(y_name))
    present = [comps[i] for i in np.unique(winner[~mask])]
    ax.legend(handles=ComponentLegendHandles(present), loc='upper left', bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize='small', title='Largest share', title_fontsize='small')
    ax.set_title(f"{sset.label(qoi.name)}: dominant contribution (paler = smaller share)",
                 fontsize='small', loc='left')

    SaveFigure(fig, f"{qoi.name}_Sobol_dominant", plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotMixedMap(sset, qoi, plot_folder, combine_plots_pdf=None, weight=None):
    x_name, x, y_name, y = _grid(qoi)
    comps, mask = Partition(sset, qoi)
    rgb = MixColours(np.array([c.values for c in comps]), [c.colour for c in comps])
    suffix = ''
    if weight == 'std' and qoi.std is not None:
        rgb = FadeToSurface(rgb, qoi.std / np.nanmax(qoi.std))
        suffix = ', faded by std'

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    _rgbImage(ax, x, y, rgb, mask)
    ax.set_xlabel(sset.label(x_name))
    ax.set_ylabel(sset.label(y_name))
    ax.legend(handles=ComponentLegendHandles(_shown(comps)), loc='upper left', bbox_to_anchor=(1.01, 1),
              frameon=False, fontsize='small', title='Pure colours', title_fontsize='small')
    ax.set_title(f"{sset.label(qoi.name)}: colour mixed by Sobol shares{suffix}", fontsize='small', loc='left')

    SaveFigure(fig, f"{qoi.name}_Sobol_mixed{'_std' if suffix else ''}", plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotSlices(sset, qoi, plot_folder, combine_plots_pdf=None, n_slices=4):
    x_name, x, y_name, y = _grid(qoi)
    comps, mask = Partition(sset, qoi)
    rows = np.unique(np.linspace(0, len(y) - 1, n_slices).round().astype(int))
    fig, axes = plt.subplots(1, len(rows), sharey=True, figsize=(2.6 * len(rows) + 1.6, 2.8), squeeze=False)
    for ax, i in zip(axes[0], rows):
        stackPartition(ax, x, [c._replace(values=c.values[i]) for c in comps], mask[i])
        ax.set_ylim(0, 1)
        ax.margins(x=0)
        ax.set_title(f"{sset.label(y_name)} = {y[i]:.3g}", fontsize='small')
        ax.set_xlabel(sset.label(x_name), fontsize='small')
        ax.grid(visible=True, linestyle='-', linewidth=0.33)
    axes[0, 0].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    axes[0, 0].set_ylabel('Share of variance')
    fig.legend(handles=ComponentLegendHandles(_shown(comps)), loc='upper left', bbox_to_anchor=(1.0, 0.9),
               frameon=False, fontsize='small')
    fig.suptitle(f"{sset.label(qoi.name)}: slices", x=0.02, ha='left')
    fig.tight_layout()

    SaveFigure(fig, f"{qoi.name}_Sobol_slices", plot_folder, combine_plots_pdf)
    plt.close(fig)


def plotSobolField(code, sset, plot_type='all', plot_path='.', qois=None, weight=None, combine_plots_pdf=True):
    qois = sset.byDim(2, qois)
    if not qois:
        print("[WARNING] no 2D QoIs to plot.")
        return
    plot_folder = PlotFolder(plot_path, code)
    with PdfPages(os.path.join(plot_folder, "combined_sobol_field.pdf")) if combine_plots_pdf else nullcontext() as pdf:
        for qoi in qois:
            if plot_type in ('components', 'all'):
                plotComponentMaps(sset, qoi, plot_folder, pdf)
            if plot_type in ('dominant', 'all'):
                plotDominantMap(sset, qoi, plot_folder, pdf)
            if plot_type in ('mixed', 'all'):
                plotMixedMap(sset, qoi, plot_folder, pdf)
                if weight == 'std' or plot_type == 'all':
                    plotMixedMap(sset, qoi, plot_folder, pdf, weight='std')
            if plot_type in ('slices', 'all'):
                plotSlices(sset, qoi, plot_folder, pdf)
    print(f"[INFO] 2D Sobol plots saved to {plot_folder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot Sobol indices of 2D QoIs")
    p.add_argument("sobols_yml")
    p.add_argument("--code", default="md")
    p.add_argument("--plot_type", default="all", choices=["components", "dominant", "mixed", "slices", "all"])
    p.add_argument("--weight", default=None, choices=[None, "std"], help="fade mixed map by output std")
    p.add_argument("--qois", nargs="*", default=None)
    p.add_argument("--param_order", nargs="*", default=None, help="parameter order (fixes colours)")
    p.add_argument("--plot_path", default=None, help="output root (default: next to sobols.yml)")
    args = p.parse_args()

    sset = LoadSobolSet(args.sobols_yml, args.param_order)
    plot_path = args.plot_path or os.path.dirname(os.path.abspath(args.sobols_yml))
    plotSobolField(args.code, sset, args.plot_type, plot_path, qois=args.qois, weight=args.weight)
