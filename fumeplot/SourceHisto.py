from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from contextlib import nullcontext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import latexplotlib as lpl

# --- latex‑friendly styling (safe even if latexplotlib absent) --------
try:
    lpl.style.use("latex12pt")
    lpl.size.set(347.12354, 549.138)
except Exception:  # noqa: E722
    pass


def _fmt(lbls):                       # small label helper
    return [s.replace("_", " ").title() for s in lbls]


def _plot_counts(idx, series_list, save, folder, pdf=None):
    """Internal – draw a single boxplot panel."""
    df = pd.DataFrame(series_list).fillna(0)
    fig, ax = plt.subplots(num=idx + 1, figsize=(10, 6))

    ax.boxplot(
        df,
        labels=_fmt(df.columns),
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="blue", linewidth=.9),
        meanprops=dict(color="black", linewidth=.9, linestyle="dashed"),
        boxprops=dict(facecolor="skyblue", linewidth=.9),
        whiskerprops=dict(color="blue", linewidth=.75),
        capprops=dict(color="blue", linewidth=.75),
        flierprops=dict(marker="o", markerfacecolor="blue",
                        markersize=2, linestyle="none"),
    )
    ax.set_title("Entries by Source")
    ax.set_ylabel("Number of Entries")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", linestyle="-", linewidth=.3)

    if pdf:
        pdf.savefig(fig)
    if save:
        Path(folder).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(folder) / "EntriesBySource.png", dpi=150)


def _filter_counts(csvs, query: str):
    """Return list‑of‑Series after applying a simple 'col op value' filter."""
    col, op, val = query.split(maxsplit=2)
    val = float(val) if val.replace(".", "", 1).isdigit() else val
    out = []
    for f in csvs:
        df = pd.read_csv(f)
        if op == "==":
            df = df[df[col] == val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]
        out.append(df["source"].value_counts())
    return out


# ------------------------------------------------------------------ #
#  PUBLIC FUNCTION
# ------------------------------------------------------------------ #
def plotSourceHist(outdirs, filters=None, *, save_fig=True,
                   plot_folder="plots", combine_pdf=False):
    """
    Box‑and‑whisker of entry counts per *source* for each ensemble run.

    Parameters
    ----------
    outdirs : list[str | Path]
        Directories that each contain a ``migration.log``.
    filters : list[str] | None
        Each string like 'gender == f' or 'age > 17'.
    """
    if filters is None:
        filters = []

    csvs = [Path(d) / "migration.log" for d in outdirs]
    series0 = [pd.read_csv(c)["source"].value_counts() for c in csvs]

    pdf_ctx = (
        PdfPages(Path(plot_folder) / "combined_box_plots.pdf")
        if combine_pdf else nullcontext()
    )
    with pdf_ctx as pdf:
        _plot_counts(0, series0, save_fig, plot_folder, pdf)
        for i, flt in enumerate(filters, 1):
            _plot_counts(i, _filter_counts(csvs, flt),
                         save_fig, plot_folder, pdf)
    if not combine_pdf:
        plt.show()