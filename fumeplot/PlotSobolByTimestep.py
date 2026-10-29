"""
Plot Sobol sensitivity indices of time-dependent QoIs as functions of a single
variable (the simulation timestep / date).

Sobol indices are not produced by plain FUMEplot ensembles (those are replica
ensembles with identical inputs). They come from an EasyVVUQ SA campaign, e.g.
the FabHomecoming SA pipeline (`fabsim localhost homecoming_analyse_SA:<config>`
with an SCSampler / PCESampler), which stores them in two ways:

  * `sobols.yml` in the campaign work dir (written when `report: True`), with
    layout
        campaign_info: {...}
        <param>:
            <qoi>_sobols_first_mean:  float
            <qoi>_sobols_first_gmean: float
            <qoi>_sobols_first:       [S_1(t_0), S_1(t_1), ...]
    (optionally also `<qoi>_sobols_total: [...]`)
  * in memory, as an EasyVVUQ AnalysisResults object
    (`campaign.get_last_analysis()`), queried with `results.sobols_first(qoi)`
    and `results.sobols_total(qoi)`.

Both sources are supported: `LoadSobolsYml` for the file, `SobolsFromResults`
for an in-process results object.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import yaml

from matplotlib.backends.backend_pdf import PdfPages
from contextlib import nullcontext

USELATEXPLOTLIB = False

if USELATEXPLOTLIB:
    import latexplotlib as lpl
    lpl.style.use('latex12pt')
    latex_doc_size = (347.12354, 549.138)
    columns_per_page = 2.
    latex_doc_size = tuple(x / columns_per_page for x in latex_doc_size)
    lpl.size.set(*latex_doc_size)

# fixed categorical order: a parameter keeps its colour in every plot
PARAM_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INTERACTION_COLOUR = '#b5b4ae'

ORDER_STYLES = {'first': '-', 'total': '--'}
ORDER_LABELS = {'first': 'First order', 'total': 'Total order'}


def _formatLabel(label):
    return label.replace('ukr-', '').replace('_', ' ').replace('-', ' ').title()


def _fileName(label):
    return str(label).replace(' ', '').replace('#', 'Num').replace('/', '_')


def _newFigure(plot_num):
    if USELATEXPLOTLIB:
        return lpl.subplots(num=plot_num+1)
    fig = plt.figure(plot_num+1)
    return fig, fig.add_subplot(111)


# ------------------------------------------------------------------ loaders
# All loaders return {qoi: {'first': {param: np.array}, 'total': {param: np.array}}}

def LoadSobolsYml(yml_path):
    """Read the `sobols.yml` report written by FabHomecoming's SA ReportGenerator."""
    with open(yml_path) as f:
        report = yaml.safe_load(f)

    sobols = {}
    for param, entries in report.items():
        if param == 'campaign_info' or not isinstance(entries, dict):
            continue
        for key, values in entries.items():
            for order in ORDER_STYLES:
                suffix = f"_sobols_{order}"
                if key.endswith(suffix) and isinstance(values, list):
                    qoi = key[:-len(suffix)]
                    sobols.setdefault(qoi, {'first': {}, 'total': {}})
                    sobols[qoi][order][param] = np.asarray(values, dtype=float).ravel()
    return sobols


def SobolsFromResults(results, qois=None):
    """Extract first/total order indices from an EasyVVUQ AnalysisResults object."""
    sobols = {}
    for qoi in (qois if qois is not None else results.qois):
        entry = {'first': {}, 'total': {}}
        for order, getter in (('first', results.sobols_first), ('total', results.sobols_total)):
            try:
                entry[order] = {p: np.asarray(v, dtype=float).ravel() for p, v in getter(qoi).items()}
            except (RuntimeError, NotImplementedError, KeyError) as e:
                print(f"[WARNING] no {order}-order Sobol indices for {qoi}: {e}")
        sobols[qoi] = entry
    return sobols


def LoadTimeAxis(out_csv, num_steps):
    """Use the first column of a run's out.csv (e.g. homecoming 'time' = MM/YYYY) as x values."""
    if out_csv is None or not os.path.isfile(out_csv):
        return np.arange(num_steps), 'Timestep'
    col = pd.read_csv(out_csv).iloc[:num_steps, 0]
    dates = pd.to_datetime(col, format='%m/%Y', errors='coerce')
    if dates.notna().all() and len(dates) == num_steps:
        return dates.values, 'Month'
    if len(col) == num_steps:
        return col.values, col.name
    return np.arange(num_steps), 'Timestep'


# ------------------------------------------------------------------ plots

def _paramColour(param, params):
    i = params.index(param)
    return PARAM_COLOURS[i % len(PARAM_COLOURS)]


def plotSobolLines(qoi, qoi_sobols, params, x, x_label, plot_num,
                   save_fig=False, plot_folder=None, combine_plots_pdf=None):
    """
    One line per parameter: first order solid, total order dashed.
    """
    fig, ax = _newFigure(plot_num)

    orders = [o for o in ORDER_STYLES if qoi_sobols[o]]
    for param in params:
        for order in orders:
            y = qoi_sobols[order].get(param)
            if y is None or np.all(np.isnan(y)):
                continue
            ax.plot(x[:len(y)], y, color=_paramColour(param, params),
                    linestyle=ORDER_STYLES[order], linewidth=2)

    # legend: colour = parameter, line style = order
    handles = [mlines.Line2D([], [], color=_paramColour(p, params), linewidth=2, label=_formatLabel(p))
               for p in params]
    if len(orders) > 1:
        handles += [mlines.Line2D([], [], color='k', linestyle=ORDER_STYLES[o], label=ORDER_LABELS[o])
                    for o in orders]
    ax.legend(handles=handles, loc='best', fontsize='small')

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Sobol index' if len(orders) > 1 else f'{ORDER_LABELS[orders[0]]} Sobol index')
    ax.set_title(_formatLabel(qoi))
    ax.grid(visible=True, which='both', axis='both', linestyle='-', linewidth=0.33)
    fig.autofmt_xdate()

    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    if save_fig:
        fig.savefig(f"{plot_folder}/{_fileName(qoi)}_Sobol_lines.png", bbox_inches='tight')

    plt.close(fig)


def plotSobolStacked(qoi, qoi_sobols, params, x, x_label, plot_num,
                     save_fig=False, plot_folder=None, combine_plots_pdf=None):
    """
    First-order indices stacked; the remainder up to 1 is the variance share of
    parameter interactions (1 - sum_i S_i).
    """
    first = qoi_sobols['first']
    present = [p for p in params if p in first]
    if not present:
        return
    n = min(len(first[p]) for p in present)
    stack = np.clip(np.nan_to_num(np.vstack([first[p][:n] for p in present])), 0, 1)
    interactions = np.clip(1.0 - stack.sum(axis=0), 0, 1)

    fig, ax = _newFigure(plot_num)
    ax.stackplot(x[:n], *stack, interactions,
                 colors=[_paramColour(p, params) for p in present] + [INTERACTION_COLOUR],
                 labels=[_formatLabel(p) for p in present] + ['Interactions'],
                 edgecolor='white', linewidth=1)

    ax.set_ylim(0, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Share of output variance')
    ax.set_title(_formatLabel(qoi))
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax.grid(visible=True, which='both', axis='y', linestyle='-', linewidth=0.33)
    fig.autofmt_xdate()

    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig, bbox_inches='tight')
    if save_fig:
        fig.savefig(f"{plot_folder}/{_fileName(qoi)}_Sobol_stacked.png", bbox_inches='tight')

    plt.close(fig)


def plotSobolGrid(sobols, qois, params, x, x_label, order='first',
                  save_fig=False, plot_folder=None, combine_plots_pdf=None, ncols=4):
    """
    Small multiples: one panel per QoI, shared axes, one line per parameter.
    """
    qois = [q for q in qois if sobols[q][order]]
    if not qois:
        return
    nrows = int(np.ceil(len(qois) / ncols))
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                             figsize=(3.2*ncols, 2.4*nrows), squeeze=False)

    for ax, qoi in zip(axes.flat, qois):
        for param in params:
            y = sobols[qoi][order].get(param)
            if y is not None:
                ax.plot(x[:len(y)], y, color=_paramColour(param, params), linewidth=1.5)
        ax.set_title(_formatLabel(qoi), fontsize='small')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(visible=True, linestyle='-', linewidth=0.33)
        ax.tick_params(axis='x', labelrotation=45, labelsize='x-small')
    for k in range(len(qois), nrows*ncols):
        axes.flat[k].set_visible(False)
        # the panel above an empty slot is the bottom of its column, so it keeps x tick labels
        if k - ncols >= 0:
            axes.flat[k - ncols].xaxis.set_tick_params(labelbottom=True)

    handles = [mlines.Line2D([], [], color=_paramColour(p, params), linewidth=2, label=_formatLabel(p))
               for p in params]
    fig.legend(handles=handles, loc='upper center', ncol=min(len(params), 8), frameon=False)
    fig.supxlabel(x_label)
    fig.supylabel(f'{ORDER_LABELS[order]} Sobol index')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5/(2.4*nrows)))

    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    if save_fig:
        fig.savefig(f"{plot_folder}/Sobol_{order}_grid.png", bbox_inches='tight')

    plt.close(fig)


# main plotting script
def plotSobolByTimestep(code, sobols, plot_type='all', plot_path='../..', qois=None,
                        param_order=None, time_file=None, combine_plots_pdf=True):
    """
    code:        flee / facs / homecoming (only names the output folder)
    sobols:      dict from LoadSobolsYml or SobolsFromResults
    plot_type:   'lines', 'stacked', 'grid' or 'all'
    qois:        subset of QoIs to plot (default: all)
    param_order: fixes the parameter -> colour mapping (default: sorted names)
    time_file:   an out.csv whose first column gives the x values
    """
    qois = [q for q in (qois or sobols.keys()) if q in sobols]
    if not qois:
        print("[WARNING] no Sobol indices to plot.")
        return

    all_params = sorted({p for q in qois for o in ORDER_STYLES for p in sobols[q][o]})
    params = [p for p in (param_order or []) if p in all_params] + \
             [p for p in all_params if p not in (param_order or [])]
    if len(params) > len(PARAM_COLOURS):
        print(f"[WARNING] {len(params)} parameters but only {len(PARAM_COLOURS)} distinct colours; "
              f"consider plotting a subset via param_order.")

    num_steps = max(len(v) for q in qois for o in ORDER_STYLES for v in sobols[q][o].values())
    x, x_label = LoadTimeAxis(time_file, num_steps)

    plotfolder = plot_path+'/EnsemblePlots/'+code+'Plots/Sobol'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)

    with PdfPages(os.path.join(plotfolder, "combined_sobol_plots.pdf")) if combine_plots_pdf else nullcontext() as pdf_pages:
        fi = 0
        for qoi in qois:
            if plot_type in ("lines", "all"):
                plotSobolLines(qoi, sobols[qoi], params, x, x_label, fi,
                               save_fig=True, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)
                fi += 1
            if plot_type in ("stacked", "all") and sobols[qoi]['first']:
                plotSobolStacked(qoi, sobols[qoi], params, x, x_label, fi,
                                 save_fig=True, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)
                fi += 1
        if plot_type in ("grid", "all"):
            for order in ORDER_STYLES:
                plotSobolGrid(sobols, qois, params, x, x_label, order=order,
                              save_fig=True, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)

    print(f"[INFO] Sobol plots saved to {plotfolder}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot Sobol indices of time-dependent QoIs")
    p.add_argument("sobols_yml", help="sobols.yml from an EasyVVUQ/FabSim3 SA campaign")
    p.add_argument("--code", default="homecoming", help="flee, facs or homecoming")
    p.add_argument("--plot_type", default="all", choices=["lines", "stacked", "grid", "all"])
    p.add_argument("--qois", nargs="*", default=None, help="QoI columns to plot (default: all)")
    p.add_argument("--param_order", nargs="*", default=None, help="parameter order (fixes colours)")
    p.add_argument("--time_file", default=None, help="out.csv whose first column is the time axis")
    p.add_argument("--plot_path", default=None, help="output root (default: next to sobols.yml)")
    args = p.parse_args()

    sobols = LoadSobolsYml(args.sobols_yml)
    plot_path = args.plot_path or os.path.dirname(os.path.abspath(args.sobols_yml))
    plotSobolByTimestep(args.code, sobols, args.plot_type, plot_path, qois=args.qois,
                        param_order=args.param_order, time_file=args.time_file)
