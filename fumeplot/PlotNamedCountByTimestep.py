# ───────── std lib ─────────
from __future__ import annotations
import sys, numpy as np, pandas as pd
from pathlib import Path
from contextlib import nullcontext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ───────── 3rd‑party ───────
import plotly.express as px

# ───────── project helpers –
import ReadHeaders
from TimeSeriesPlots import (plotLocation, plotLocationSTDBound,
                             plotLocationDifferences, EnsembleData)
from AnimatedCharts  import (animateLocationHistogram,
                             animateLocationViolins)

# NEW: separate helper modules you created earlier
from SourceHisto      import plotSourceHist
from MigrationSankey import plotMigrationSankey
from TimeSeriesFan import plotLineOverTime
from CountBarCharts import plotStackedBar          

# ───────────────────────────────────────────────────────────────────────


def _format_labels(lbls):          # tiny util for Matplotlib ticks
    return [s.replace("_", " ").title() for s in lbls]


# Final‑timestep bar (unchanged)
# ---------------------------------------------------------------------
def plot_final_oblast_bar(outdirs, folder: Path):
    run_vals, cols = [], None
    for d in outdirs:
        csv = Path(d) / "out.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if cols is None:
            cols = [c for c in df.columns if c.startswith("ukr-")]
        run_vals.append(df.iloc[-1][cols].astype(float).values)

    if not run_vals:
        return

    mean_counts = np.vstack(run_vals).mean(axis=0)
    names = [c.replace("ukr-", "").replace("-", " ").title() for c in cols]
    fig = px.bar(
        {"Oblast": names, "Mean Returnees": mean_counts},
        x="Oblast", y="Mean Returnees",
        title="Mean Returnees by Ukrainian Oblast (final timestep)",
        labels={"Mean Returnees": "Returnees"},
    )
    fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=20, t=50, b=120))
    folder.mkdir(parents=True, exist_ok=True)
    fig.write_html(folder / "final_oblast_bar.html")


# ---------------------------------------------------------------------
# master driver
# ---------------------------------------------------------------------
def plotNamedCountByTimestep(code: str, outdirs, plot_type: str,
                             header, *, root="../.."):

    loc_names    = header.loc_names
    sim_idx      = header.sim_indices
    data_idx     = header.data_indices
    y_label      = header.y_label.lstrip("\\").replace("#", "Number")
    combine_pdf  = header.combine_plots_pdf

    plotdir = Path(root) / "EnsemblePlots" / f"{code}Plots"
    plotdir.mkdir(parents=True, exist_ok=True)

    def _assemble_full():
        return [EnsembleData(outdirs, sidx)[0] for sidx in sim_idx]

    pdf_ctx = PdfPages(plotdir / "combined_time_plots.pdf") if combine_pdf else nullcontext()

    # ─── LOCATION‑based time‑series & GIFs ────────────────────────────
    with pdf_ctx as pdf:
        fi = 0
        for i, (sidx, didx) in enumerate(zip(sim_idx, data_idx)):
            df, ens_size, outdf = EnsembleData(outdirs, sidx)

            if plot_type in ("loc_lines", "all"):
                plotLocation(df, ens_size, outdf, fi, i, didx,
                             loc_names, y_label, True, str(plotdir), pdf)
                fi += 1

            if plot_type in ("loc_stdev", "all"):
                plotLocationSTDBound(df, ens_size, outdf, fi, i, didx,
                                     loc_names, y_label, True, str(plotdir), pdf)
                fi += 1

            if didx >= 0 and plot_type in ("loc_diff", "all"):
                plotLocationDifferences(outdirs, fi, i, sidx, didx, loc_names,
                                         True, str(plotdir), pdf)
                fi += 1

            if plot_type in ("loc_hist_gif", "all"):
                animateLocationHistogram(df, ens_size, outdf,
                                            plot_num=fi,
                                            loc_index=i,
                                            data_index=didx,
                                            loc_names=loc_names,
                                            x_label=y_label,
                                            save_fig=True,
                                            plot_folder=str(plotdir),)
                fi += 1

                

            if plot_type in ("loc_violin_gif", "all") and i == 0:
                animateLocationViolins(_assemble_full(),          # dfFull
                                        ens_size,                  # ensembleSize
                                        outdf,                     # outdf
                                        plot_num=fi,
                                        loc_index=0,               # keep 0 as in your original logic
                                        sim_indices=sim_idx,
                                        data_indices=data_idx,
                                        loc_names=loc_names,
                                        y_label=y_label,
                                        save_fig=True,
                                        plot_folder=str(plotdir),)
                fi += 1

            if plot_type in ("bar_chart", "all") and i == 0:
                plot_final_oblast_bar(outdirs, plotdir)

    if not combine_pdf:
        plt.show()

    # ─── GROUP‑level helpers (separate modules) ───────────────────────
    if plot_type in ("source_hist", "all"):
        plotSourceHist(outdirs, filters=[], save_fig=True,
                       plot_folder=str(plotdir), combine_pdf=True)

    if plot_type in ("single_sankey", "all"):
        plotMigrationSankey(outdirs, save_fig=True, plot_folder=str(plotdir))

    if plot_type in ("stacked_bar", "all"):
        for col in ("gender", "age", "education", "property_in_ukraine"):
            plotStackedBar(outdirs, disaggregator=col,
                           filters=[], save_fig=True,
                           plot_folder=str(plotdir))

    if plot_type in ("line_chart", "all"):
        for col in ("gender", "age_binned", "education", "property_in_ukraine"):
            plotLineOverTime(outdirs,
                             primary_filter_column="source",
                             primary_filter_value=None,
                             line_disaggregator=col,
                             filters=[], save_fig=True,
                             plot_folder=str(plotdir))


# ---------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    code      = sys.argv[1] if len(sys.argv) > 1 else "homecoming"
    plot_type = sys.argv[2] if len(sys.argv) > 2 else "all"

    outdir    = f"../sample_{code}_output"
    outdirs   = ReadHeaders.GetOutDirs(outdir)
    header    = ReadHeaders.ReadOutHeaders(outdirs, mode=code)

    plotNamedCountByTimestep(code, outdirs, plot_type, header)
    print(f"[DONE] Plots saved in {Path(outdir) / 'EnsemblePlots' / f'{code}Plots'}")
