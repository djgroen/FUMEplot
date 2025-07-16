import sys
import os
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import ReadHeaders
from TimeSeriesPlots import (
    plotLocation,
    plotLocationSTDBound,
    plotLocationDifferences,
    EnsembleData,
)

# helper module that houses the animated GIF makers
from AnimatedCharts import (
    animateLocationHistogram,
    animateLocationViolins,
)

"""High-level driver that produces static time‑series plots + animated GIFs
for the FUME ‘homecoming’ use‑case (or any other code supported by
ReadHeaders).

The script now passes **in‑memory ensemble arrays** to the GIF helpers, so no
function in animatedcharts ever touches the filesystem.
"""

# ---------------------------------------------------------------------------
# 0.  Utility
# ---------------------------------------------------------------------------

def _formatLabels(labels):
    return [lbl.replace('_', ' ').title() for lbl in labels]

# ---------------------------------------------------------------------------
# 1.  Final‑timestep bar‑chart helper 
# ---------------------------------------------------------------------------

def plotFinalOblastBarHTML(outdirs, plot_folder="plots"):
    run_vals, oblast_cols = [], None
    for d in outdirs:
        csv = Path(d) / "out.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if oblast_cols is None:
            oblast_cols = [c for c in df.columns if c.startswith("ukr-")]
        run_vals.append(df.iloc[-1][oblast_cols].astype(float).values)

    if not run_vals:
        return

    arr = np.vstack(run_vals)
    mean_counts = arr.mean(axis=0)
    names = [c.replace("ukr-", "").replace("-", " ").title() for c in oblast_cols]
    df_plot = pd.DataFrame({"Oblast": names, "Mean Returnees": mean_counts})

    fig = px.bar(df_plot, x="Oblast", y="Mean Returnees",
                 title="Mean Returnees by Ukrainian Oblast (final timestep)",
                 labels={"Mean Returnees": "Returnees", "Oblast": "Oblast"})
    fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=20, t=50, b=120))

    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    fig.write_html(Path(plot_folder) / "final_oblast_bar.html")

# ---------------------------------------------------------------------------
# 2.  Main plotting routine
# ---------------------------------------------------------------------------

def plotNamedCountByTimestep(code, outdirs, plot_type, FUMEheader, plot_path="../.."):
    sim_indices  = FUMEheader.sim_indices
    data_indices = FUMEheader.data_indices
    loc_names    = FUMEheader.loc_names
    y_label      = FUMEheader.y_label.replace('#', 'Number')
    combine_pdf  = FUMEheader.combine_plots_pdf

    plotfolder = Path(plot_path) / "EnsemblePlots" / f"{code}Plots"
    plotfolder.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # helper to build dfFull once
    # ----------------------------
    def _build_dfFull():
        return [EnsembleData(outdirs, si)[0] for si in sim_indices]

    pdf_ctx = PdfPages(plotfolder / "combined_time_plots.pdf") if combine_pdf else nullcontext()

    with pdf_ctx as pdf:
        fi = 0  # figure index
        for i, (sim_i, data_i) in enumerate(zip(sim_indices, data_indices)):
            # load the ensemble once for this column
            dfTest, ensembleSize, outdf = EnsembleData(outdirs, sim_i)

            # ---- static plots -----------------------------------------
            if plot_type in ("loc_lines", "all"):
                plotLocation(dfTest, ensembleSize, outdf,
                             plot_num=fi, loc_index=i, data_index=data_i,
                             loc_names=loc_names, y_label=y_label,
                             save_fig=True, plot_folder=str(plotfolder),
                             combine_plots_pdf=pdf)
                fi += 1

            if plot_type in ("loc_stdev", "all"):
                plotLocationSTDBound(dfTest, ensembleSize, outdf,
                                     plot_num=fi, loc_index=i, data_index=data_i,
                                     loc_names=loc_names, y_label=y_label,
                                     save_fig=True, plot_folder=str(plotfolder),
                                     combine_plots_pdf=pdf)
                fi += 1

            if data_i >= 0 and plot_type in ("loc_diff", "all"):
                plotLocationDifferences(outdirs, fi, i, sim_i, data_i, loc_names,
                                         save_fig=True, plot_folder=str(plotfolder),
                                         combine_plots_pdf=pdf)
                fi += 1

            # ---- animated histogram ----------------------------------
            if plot_type in ("loc_hist_gif", "all"):
                animateLocationHistogram(dfTest, ensembleSize, outdf,
                                         plot_num=fi, loc_index=i,
                                         data_index=data_i, loc_names=loc_names,
                                         x_label=y_label, save_fig=True,
                                         plot_folder=str(plotfolder))
                fi += 1

            # ---- animated violins (needs all categories) -------------
            if plot_type in ("loc_violin_gif", "all") and i == 0:  # only once
                dfFull = _build_dfFull()
                animateLocationViolins(dfFull, ensembleSize, outdf,
                                        plot_num=fi, loc_index=0,
                                        sim_indices=sim_indices,
                                        data_indices=data_indices,
                                        loc_names=loc_names,
                                        y_label=y_label, save_fig=True,
                                        plot_folder=str(plotfolder))

            # ---- final bar chart -------------------------------------
            if plot_type in ("bar_chart", "all") and i == 0:
                plotFinalOblastBarHTML(outdirs, plot_folder=str(plotfolder))

    if not combine_pdf:
        plt.show()

# ---------------------------------------------------------------------------
# 3.  CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    code      = sys.argv[1] if len(sys.argv) > 1 else "homecoming"
    plot_type = sys.argv[2] if len(sys.argv) > 2 else "all"
    outdir    = f"../sample_{code}_output"
    outdirs   = ReadHeaders.GetOutDirs(outdir)
    FUMEheader= ReadHeaders.ReadOutHeaders(outdirs, mode=code)
    plotNamedCountByTimestep(code, outdirs, plot_type, FUMEheader)
    print(f"Plots saved to {Path(outdir) / 'EnsemblePlots' / f'{code}Plots'}")
   