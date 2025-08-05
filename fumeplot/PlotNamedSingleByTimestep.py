from __future__ import annotations

# ────────── std lib / scientific ──────────
import sys, numpy as np, pandas as pd
from pathlib import Path
from contextlib import nullcontext

# (You can trim imports further once you know exactly what you need)
import matplotlib.pyplot as plt                     # still handy for new plots
from matplotlib.backends.backend_pdf import PdfPages
from dateutil.relativedelta import relativedelta

# ────────── project helpers (data I/O only) ──────────
import ReadHeaders                                  # ← keep: loads ensemble meta
from TimeSeriesPlots  import EnsembleData           # ← matrix builder only


# (latexplotlib config left in case you need it later)
import latexplotlib as lpl
lpl.style.use("latex12pt")
lpl.size.set(347.12354, 549.138)


# ───────────────────────── helper utilities ───────────────────────────
def _format_labels(labels):
    """Title‑case + replace underscores → spaces."""
    return [lbl.replace("_", " ").title() for lbl in labels]


# ──────────────────────────── *STUB* plots ────────────────────────────
# Replace each of these with your real implementation when ready.
# Until then they do nothing and will not raise.
# ---------------------------------------------------------------------

def plotSourceHist(*args, **kwargs) -> None:
    """Box‑and‑whisker of source counts – placeholder."""
    pass

def plotMigrationSankey(*args, **kwargs) -> None:
    """Mean origin→destination Sankey – placeholder."""
    pass

def plotLineOverTime(*args, **kwargs) -> None:
    """Fan/line plot over time – placeholder."""
    pass


# ─────────────────────── high‑level orchestrator ──────────────────────
def plotNamedSingleByTimestep(
    code: str,
    outdirs: list[str] | list[Path],
    plot_type: str,
    header,
    *,                           # all remaining args are keyword‑only
    filters: list[tuple] | None = None,
    disaggregator: str | None = None,
    primary_filter_column: str | None = None,
    primary_filter_value: str | None = None,
    plot_path: str | Path = "../..",
) -> None:
    """Dispatch to whichever plot helpers you later plug in."""

    # ---------------- default arguments ----------------
    filters = filters or []
    disaggregator = disaggregator or getattr(header, "disaggregator", "gender")
    primary_filter_column = primary_filter_column or getattr(
        header, "primary_filter_column", "source"
    )
    primary_filter_value = primary_filter_value or getattr(
        header, "primary_filter_value", None
    )

    # -------------- output directory ------------------
    plot_dir = Path(plot_path) / "EnsemblePlots" / f"{code}Plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # -------------- call requested plot ----------------
    if plot_type in ("source_hist", "all"):
        plotSourceHist(outdirs, filters, save_fig=True, plot_folder=str(plot_dir))

    if plot_type in ("single_sankey", "all"):
        plotMigrationSankey(outdirs, save_fig=True, plot_folder=str(plot_dir))

    if plot_type in ("line_chart", "all"):
        plotLineOverTime(
            outdirs,
            primary_filter_column=primary_filter_column,
            primary_filter_value=primary_filter_value,
            line_disaggregator=disaggregator,
            filters=filters,
            save_fig=True,
            plot_folder=str(plot_dir),
        )

    # Future bundles (stacked‑bars, gifs, etc.) can be slotted in here.


# ─────────────────────────────── CLI ──────────────────────────────────
if __name__ == "__main__":
    # Positional args:  code  plot_type
    code      = sys.argv[1] if len(sys.argv) > 1 else "homecoming"
    plot_type = sys.argv[2] if len(sys.argv) > 2 else "all"

    # Resolve ensemble directories + meta header (no plotting yet)
    outdir   = f"../sample_{code}_agentlog"
    outdirs  = ReadHeaders.GetOutDirs(outdir)
    header   = ReadHeaders.ReadOutHeaders(outdirs, mode=code)

    # Drive the (currently stubbed) dispatcher
    plotNamedSingleByTimestep(code, outdirs, plot_type, header)

    print(
        "[DONE]  No plots were generated because all plot helpers are stubs.\n"
        "        Replace them with real implementations when ready!"
    )