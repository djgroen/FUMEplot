from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# 0.  Labels, palettes, helper functions
# ---------------------------------------------------------------------------

EDU_LABELS: dict[int, str] = {
    0: "No Education",
    1: "Primary",
    2: "Secondary",
    4: "Vocational",
    3: "Bachelor",      # adjust if your code→label mapping differs
    5: "Specialist",
    6: "Master Plus",
}

PROP_LABELS: dict[int, str] = {
    0: "No Property",
    1: "Unknown Status",
    2: "Fully Damaged",
    3: "Partially Damaged",
    5: "Intact",
}

DISAGG_LABELS: dict[str, str] = {
    "age":                  "Age",
    "age_binned":           "Age",
    "gender":               "Gender",
    "education":            "Education",
    "property_in_ukraine":  "Property In Ukraine",
}

AGE_PALETTE: list[str] = px.colors.sequential.Plasma  # or Viridis, Turbo, …

GENDER_COLORS: dict[str, str] = {
    "f": "#e05779",   # pinkish
    "m": "#4a78b5",   # steel‑blue
}

GENDER_NAMES: dict[str, str] = {
    "f": "Female",
    "m": "Male",
}

def pretty(name: str) -> str:
    """Human‑readable version of a disaggregator column name."""
    return DISAGG_LABELS.get(name, name.replace("_", " ").title())


def _formatLabels(labels: List[str]) -> List[str]:
    """Helper for Matplotlib boxplots (kept for potential reuse)."""
    nice = [lbl.replace("_", " ").title() for lbl in labels]
    return [lbl.split()[-1] for lbl in nice]

# ---------------------------------------------------------------------------
# 1.  Core bar‑chart routine (refactored)
# ---------------------------------------------------------------------------

def plotStackedBar(
    outdirs: Sequence[Union[str, os.PathLike]],
    disaggregator: str | Sequence[str] = "age_binned",
    *,
    filters: Sequence[Tuple[str, str, Any]] | None = None,
    save_fig: bool = True,
    plot_folder: str | os.PathLike = "plots",
) -> None:
    """Create a **stacked bar chart of *net* arrivals** for each destination.

    Parameters
    ----------
    outdirs : list[str] | list[Path]
        One directory per ensemble run — each must contain *migration.log*.
    disaggregator : str
        Column to split the bars by (`age`, `gender`, `education`, …).
        If `'age'` we auto‑create an `'age_binned'` column (0‑17,18‑29…).
    filters : list[tuple[str, str, Any]] | None
        Optional (column, operator, value) filters to apply before grouping.
    save_fig : bool
        Write an HTML file when *True*. Always returns the `plotly` Figure.
    plot_folder : str or Path
        Output directory for the HTML file.
    """
    if filters is None:
        filters = []

    # pick first element if user passed a sequence
    if isinstance(disaggregator, (list, tuple)) and disaggregator:
        disaggregator = disaggregator[0]

    per_run_tables: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Step 1  Read each run and compute destination×category net arrivals
    # ------------------------------------------------------------------
    for run_dir in outdirs:
        log_path = os.path.join(run_dir, "migration.log")
        if not os.path.exists(log_path):
            print(f"[WARNING] No migration.log in {run_dir} – skipping.")
            continue

        df = pd.read_csv(log_path)

        # apply user filters
        for col, op, val in filters:
            if op == "==":
                df = df[df[col] == val]
            elif op == "!=":
                df = df[df[col] != val]
            elif op == ">":
                df = df[df[col] > val]
            elif op == "<":
                df = df[df[col] < val]
            elif op == ">=":
                df = df[df[col] >= val]
            elif op == "<=":
                df = df[df[col] <= val]
            else:
                print(f"[WARNING] Unsupported filter {col}{op}{val}, skipping.")

        # special case: create bins when disaggregator == 'age'
        if disaggregator == "age":
            if "age" not in df.columns:
                print("[ERROR] Cannot bin 'age' – no 'age' column.")
                return
            bins = [0, 17, 29, 49, 64, 200]
            labels = ["0-17", "18-29", "30-49", "50-64", "65+"]
            df["age_binned"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
            agg_col = "age_binned"
        else:
            agg_col = disaggregator

        if agg_col not in df.columns or "destination" not in df.columns:
            print(f"[ERROR] Missing '{agg_col}' or 'destination' – aborting.")
            return

        # --- net arrivals per (destination × agg_col) ------------------
        arrivals = (
            df.assign(arrival=1)
            .groupby(["destination", agg_col])["arrival"].sum()
            .reset_index()
        )
        departures = (
            df.assign(departure=1)
            .groupby(["source", agg_col])["departure"].sum()
            .reset_index()
            .rename(columns={"source": "destination"})
        )
        net = pd.merge(arrivals, departures, on=["destination", agg_col], how="outer")
        net[["arrival", "departure"]] = net[["arrival", "departure"]].fillna(0)
        net["net"] = net["arrival"] - net["departure"]
        pivot = net.pivot_table(index="destination", columns=agg_col, values="net", fill_value=0)
        per_run_tables.append(pivot)

    if not per_run_tables:
        print("[INFO] No valid data found – nothing to plot.")
        return

    # ------------------------------------------------------------------
    # Step 2  Average across runs and filter to UA oblasts
    # ------------------------------------------------------------------
    all_dest = sorted(set().union(*(t.index   for t in per_run_tables)))
    all_cats = sorted(set().union(*(t.columns for t in per_run_tables)))

    R, D, C = len(per_run_tables), len(all_dest), len(all_cats)
    cube = np.zeros((R, D, C), dtype=float)
    for i, tbl in enumerate(per_run_tables):
        cube[i] = tbl.reindex(index=all_dest, columns=all_cats, fill_value=0).values

    mean_df = pd.DataFrame(cube.mean(axis=0), index=all_dest, columns=all_cats)
    mean_df.index = mean_df.index.str.strip().str.lower().str.replace("_", "-")

    ukr_oblasts = [
        "kyivska", "zakarpatska", "ivano-frankivska", "ternopilska",
        "rivnenska", "volynska", "zhytomyrska", "khmelnytska",
        "vinnytska", "chernivetska", "kyiv", "chernihivska",
        "sumska", "cherkaska", "poltavska", "kharkivska",
        "dnipropetrovska", "kirovohradska", "odeska",
        "mykolaiivska", "khersonska", "donetska", "zaporizka",
        "luhanska", "autonomous-republic-of-crimea",
    ]
    mean_df = mean_df.loc[mean_df.index.isin(ukr_oblasts)]
    if mean_df.empty:
        print("[INFO] No Ukrainian oblast destinations to plot – skipping.")
        return

    # label remapping (for legend readability)
    if disaggregator == "education":
        mean_df.rename(columns=EDU_LABELS, inplace=True)
    elif disaggregator == "property_in_ukraine":
        mean_df.rename(columns=PROP_LABELS, inplace=True)

    # ------------------------------------------------------------------
    # Step 3  Plotly stacked bar
    # ------------------------------------------------------------------
    mean_df.index.name = "destination"
    df_long = mean_df.reset_index().melt(
        id_vars="destination", value_name="mean_count", var_name=disaggregator
    )

    # gender needs nicer names & fixed colours
    color_map = None
    if disaggregator == "gender":
        df_long["gender"] = df_long["gender"].map(GENDER_NAMES)
        color_map = {GENDER_NAMES[k]: GENDER_COLORS[k] for k in GENDER_COLORS}

    pretty_name = pretty(disaggregator)
    fig = px.bar(
        df_long,
        x="destination",
        y="mean_count",
        color=disaggregator,
        barmode="stack",
        color_discrete_map=color_map,
        title=f"Net Arrivals by Destination, Stacked by {pretty_name}",
        labels={"mean_count": "Net No. of Returnees (×100)", "destination": "Destination", disaggregator: pretty_name},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        title_font_size=25,
        font=dict(size=18),
        legend=dict(font=dict(size=18), title_text=pretty_name),
        margin=dict(l=120, r=20, t=80, b=80),
    )

    # auto y‑axis tick step
    max_val = df_long["mean_count"].max()
    dtick_val = 100 if max_val > 100 else 25
    fig.update_yaxes(tickmode="linear", tick0=0, dtick=dtick_val, showgrid=True, gridcolor="LightGray")

    # ------------------------------------------------------------------
    # Step 4  Save and/or return the figure
    # ------------------------------------------------------------------
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    html_out = Path(plot_folder) / f"stacked_bar_{disaggregator}.html"
    fig.write_html(html_out)
    if save_fig:
        print(f"[INFO] Saved HTML to {html_out}")

    return fig

# ---------------------------------------------------------------------------
# Public symbols (for * import)
# ---------------------------------------------------------------------------

__all__ = [
    "plotStackedBar",
    "EDU_LABELS",
    "PROP_LABELS",
    "GENDER_COLORS",
    "GENDER_NAMES",
    "pretty",
]
