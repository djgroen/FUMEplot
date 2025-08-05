import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import json
import plotly.express as px
import glob 

from matplotlib.backends.backend_pdf import PdfPages
from contextlib import nullcontext


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import argparse

def EnsembleData(outdirs, sim_index):
    """
    Read column `sim_index` from each out.csv in `outdirs`.
    Returns:
      - dfTest:        list of pandas.Series (one per run)
      - ensembleSize:  int
      - outdf:         the last DataFrame read (for any reference columns)
    """
    dfTest = []
    outdf = None
    for d in outdirs:
        csv_path = Path(d) / "out.csv"
        if not csv_path.exists():
            print(f"[WARNING] {csv_path} not found; skipping.")
            continue
        outdf = pd.read_csv(csv_path)
        dfTest.append(outdf.iloc[:, sim_index])
    ensembleSize = len(dfTest)
    return dfTest, ensembleSize, outdf

def plotLocation(dfTest, ensembleSize, outdf,
                 plot_num, loc_index, data_index, loc_names, y_label,
                 save_fig=False, plot_folder=None,
                 combine_plots_pdf=None):
    """
    Plot all ensemble members + mean + optional reference timeseries.
    """
    fig, ax = plt.subplots()

    # draw each ensemble member lightly
    data_arrays = [s.values for s in dfTest]
    for arr in data_arrays:
        ax.plot(arr, 'k', alpha=0.15)

    # ensemble mean
    mean_arr = np.mean(data_arrays, axis=0)
    ax.plot(mean_arr, 'maroon', label='Ensemble Mean')

    # optional reference data
    if data_index >= 0:
        ref = outdf.iloc[:, data_index].values
        ax.plot(ref, 'b-', label='Reference')

    ax.set(xlabel='Timestep', ylabel=y_label, title=loc_names[loc_index])
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='-', linewidth=0.33)

    # combine into one PDF
    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    # save standalone if requested
    if save_fig and plot_folder:
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        fn = str(loc_names[loc_index]).replace(' ', '').replace('#', 'Num')
        fig.savefig(Path(plot_folder) / f"{fn}_Ensemble.png")

    plt.close(fig)


def plotLocationSTDBound(dfTest, ensembleSize, outdf,
                         plot_num, loc_index, data_index, loc_names, y_label,
                         save_fig=False, plot_folder=None,
                         combine_plots_pdf=None):
    """
    Plot ensemble mean ±1 std dev and the overall min/max envelope.
    """
    fig, ax = plt.subplots()

    data_arrays = np.vstack([s.values for s in dfTest])
    mean = data_arrays.mean(axis=0)
    std  = data_arrays.std(axis=0)
    t = np.arange(mean.shape[0])

    ax.fill_between(t, mean - std, mean + std,
                    alpha=0.3, color='maroon', label='Mean ±1 STD')
    ax.fill_between(t,
                    data_arrays.min(axis=0),
                    data_arrays.max(axis=0),
                    alpha=0.1, color='maroon', label='Min–Max')

    if data_index >= 0:
        ref = outdf.iloc[:, data_index].values
        ax.plot(ref, 'b-', label='Reference')

    ax.set(xlabel='Timestep', ylabel=y_label, title=loc_names[loc_index])
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='-', linewidth=0.33)

    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    if save_fig and plot_folder:
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        fn = str(loc_names[loc_index]).replace(' ', '').replace('#', 'Num')
        fig.savefig(Path(plot_folder) / f"{fn}_STDBound.png")

    plt.close(fig)


def plotLocationDifferences(outdirs, plot_num, loc_index,
                            sim_index, data_index, loc_names,
                            save_fig=False, plot_folder=None,
                            combine_plots_pdf=None):
    """
    Compute RMSE & ARD of ensemble mean vs. reference, and plot the difference.
    """
    dfTest, ensembleSize, outdf = EnsembleData(outdirs, sim_index)
    data_arrays = np.vstack([s.values for s in dfTest])
    mean_arr = data_arrays.mean(axis=0)
    ref = outdf.iloc[:, data_index].values
    diff = mean_arr - ref

    rmse = np.sqrt((diff**2).mean())
    ard  = np.abs(diff).mean()

    fig, ax = plt.subplots()
    ax.plot(diff, 'k')
    stats = f"RMSE = {rmse:.2f}\nARD = {ard:.2f}"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
            va='top', ha='left', bbox=dict(boxstyle='round', alpha=0.3))

    ax.set(xlabel='Timestep', ylabel='Sim − Obs', title=loc_names[loc_index])
    ax.grid(True, which='both', linestyle='-', linewidth=0.33)

    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    if save_fig and plot_folder:
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        fn = str(loc_names[loc_index]).replace(' ', '').replace('#', 'Num')
        fig.savefig(Path(plot_folder) / f"{fn}_Differences.png")

    plt.close(fig)


def main(input_file=None, output_folder="plots"):
    """
    Simple wrapper that:
      • Reads `input_file` as a newline-separated list of run directories (if given),
        otherwise glob("run_*/").
      • Uses sim_index=0, data_index=1, dummy loc_names/y_label,
        then calls plotLocation once per location.
    """
    # 1. figure out your runs
    if input_file and os.path.isfile(input_file):
        with open(input_file) as f:
            outdirs = [line.strip() for line in f if line.strip()]
    else:
        outdirs = glob.glob("run_*/")

    # 2. pick your columns & labels
    sim_index  = 0
    data_index = 1
    # temporary names (override as you like)
    dfTest, ensembleSize, outdf = EnsembleData(outdirs, sim_index)
    loc_names  = [f"Loc{i+1}" for i in range(ensembleSize)]
    y_label    = "Returnees"

    # 3. make output folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 4. loop & plot
    for i in range(ensembleSize):
        plotLocation(dfTest, ensembleSize, outdf,
                     plot_num=i, loc_index=i,
                     data_index=data_index, loc_names=loc_names, y_label=y_label,
                     save_fig=True, plot_folder=output_folder,
                     combine_plots_pdf=None)

    print(f"[INFO] Saved ensemble plots to '{output_folder}/'.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Produce ensemble bar charts")
    p.add_argument("--input_file",
                   help="Text file listing one run‐directory per line",
                   default=None)
    p.add_argument("--output_folder",
                   help="Where to save PNGs",
                   default="plots")
    args = p.parse_args()
    main(args.input_file, args.output_folder)