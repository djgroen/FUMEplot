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
from typing import List, Sequence


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


USELATEXPLOTLIB = False
if USELATEXPLOTLIB:
    import latexplotlib as lpl
    lpl.style.use('latex12pt')
    latex_doc_size = (347.12354, 549.138)
    columns_per_page = 2.
    latex_doc_size = tuple(x / columns_per_page for x in latex_doc_size)
    lpl.size.set(*latex_doc_size)

def adjacent_values(vals: np.ndarray, q1: float, q3: float):
    """Tukey whisker helper (kept here because violins call it)."""
    upper = q3 + (q3 - q1) * 1.5
    upper = np.clip(upper, q3, vals[-1])
    lower = q1 - (q3 - q1) * 1.5
    lower = np.clip(lower, vals[0], q1)
    return lower, upper


def animateLocationHistogram(
    dfTest: Sequence[pd.Series],
    ensembleSize: int,
    outdf: pd.DataFrame,
    *,
    plot_num: int,
    loc_index: int,
    data_index: int,
    loc_names: Sequence[str],
    x_label: str,
    save_fig: bool = False,
    plot_folder: str | os.PathLike | None = None,
):
    """Build an animated GIF of histograms across runs at every time‑step."""
    maxPop = max(s.max() for s in dfTest)
    if data_index >= 0:
        maxPop = max(maxPop, outdf.iloc[:, data_index].max())

    # --- figure set‑up -------------------------------------------------------
    if USELATEXPLOTLIB:  # pragma: no cover –  optional path
        import latexplotlib as lpl  # type: ignore
        fig, ax = lpl.subplots(num=plot_num + 1)
    else:
        fig, ax = plt.subplots(num=plot_num + 1)

    # ---------------------------------------------------------------------
    def update_frame(i: int):
        ax.cla()
        ax.hist([s.iloc[i] for s in dfTest], bins=10, edgecolor="k", alpha=0.65,
                label="Ensemble Data")
        if data_index >= 0:
            ax.axvline(outdf.iloc[i, data_index], color="k", ls="--", lw=1,
                        label="Reference")
        ax.set_title(f"{loc_names[loc_index]} – Day {i}")
        ax.set(xlim=(0, 1.1 * maxPop), ylim=(0, ensembleSize),
                xlabel=x_label, ylabel="Ensemble Observations")
        ax.grid(True, which="both", lw=0.33)
        ax.legend()
        return tuple()

    # frames = length of the time axis (assume all runs same length)
    n_frames = len(dfTest[0])
    ani = animation.FuncAnimation(fig, update_frame, frames=n_frames, blit=False)

    if save_fig and plot_folder is not None:
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        outfile = Path(plot_folder) / f"{loc_names[loc_index].replace(' ', '').replace('#','Num')}_Histogram.gif"
        ani.save(outfile, writer="pillow")
        print(f"[INFO] saved {outfile}")

    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. Animated violin plot (USES EnsembleData arrays for *all* categories)
# -----------------------------------------------------------------------------

def _gather_dfFull(outdirs: Sequence[str], sim_indices: Sequence[int]):
    """Return list[list[Series]] shaped as [len(sim_indices)][n_runs]."""
    # We call EnsembleData once per *category* (sim column) so we still only
    # read each CSV len(sim_indices) times – acceptable for a small count,
    # much less than original per‑plot loops.
    return [EnsembleData(outdirs, si)[0] for si in sim_indices]


def animateLocationViolins(
    dfFull: Sequence[Sequence[pd.Series]],  # shape cat × run
    ensembleSize: int,
    outdf: pd.DataFrame,
    *,
    plot_num: int,
    loc_index: int,
    sim_indices: Sequence[int],
    data_indices: Sequence[int],
    loc_names: Sequence[str],
    y_label: str,
    save_fig: bool = False,
    plot_folder: str | os.PathLike | None = None,
):
    """Animated violin plot summarising run‑to‑run spread for each category."""
    dataAvailable = any(di >= 0 for di in data_indices)
    if dataAvailable:
        dataValues = outdf.iloc[:, data_indices]

    # convert dfFull (cat × run) to the shape expected in frames:------------
    #    locData[j][i] = value of category *j* in run *r* at time *i*
    cats = len(sim_indices)
    time_len = len(dfFull[0][0])

    if USELATEXPLOTLIB:  # pragma: no cover
        import latexplotlib as lpl  # type: ignore
        fig, ax = lpl.subplots(num=plot_num + 1)
    else:
        fig, ax = plt.subplots(num=plot_num + 1)

    def update_frame(i: int):
        ax.cla()
        locData = [[run_series.iloc[i] for run_series in dfFull[cat]]
                   for cat in range(cats)]
        quartile1, medians, quartile3 = np.percentile(locData, [25, 50, 75], axis=1)
        whiskers = np.array([adjacent_values(np.sort(arr), q1, q3)
                             for arr, q1, q3 in zip(locData, quartile1, quartile3)])
        ax.violinplot(locData, showmeans=False, showmedians=True, showextrema=False)

        if dataAvailable:
            ax.scatter(np.arange(1, cats + 1), dataValues.iloc[i].values,
                       color="r", label="Reference")
            ax.vlines(np.arange(1, cats + 1), quartile1, quartile3, color="k", lw=1)
            handles, _ = ax.get_legend_handles_labels()
            handles.append(mpatches.Patch(color="C0", label="Simulations"))
            ax.legend(handles=handles)

        ax.set_title(f"{y_label} – Day {i}")
        ax.set(xlabel="Category", ylabel=y_label.replace("Number", ""))
        ax.set_xticks(np.arange(1, cats + 1))
        ax.set_xticklabels(loc_names, rotation=45, ha="right")
        ax.yaxis.grid(True)
        plt.tight_layout()
        return tuple()

    ani = animation.FuncAnimation(fig, update_frame, frames=time_len, blit=False)

    if save_fig and plot_folder is not None:
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        outfile = Path(plot_folder) / "Overall_Violin.gif"
        ani.save(outfile, writer="pillow")
        print(f"[INFO] saved {outfile}")

    plt.close(fig)

