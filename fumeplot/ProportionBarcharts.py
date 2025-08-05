
""" This script generates various plots for named Ratios by Timestep. 
    By Ratio it is a propotion of the total number of agents being plotted, 
    Eg: 1:3 or 20% went to a location and 80% in another out of 100%. """
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
from matplotlib.backends.backend_pdf import PdfPages
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hard-coded list of Ukrainian oblast names used repeatedly in the original code.
UKR_OBLASTS: List[str] = [
    "kyivska", "zakarpatska", "ivano-frankivska", "ternopilska", "rivnenska",
    "volynska", "zhytomyrska", "khmelnytska", "vinnytska", "chernivetska",
    "kyiv", "chernihivska", "sumska", "cherkaska", "poltavska", "kharkivska",
    "dnipropetrovska", "kirovohradska", "odeska", "mykolaiivska", "khersonska",
    "donetska", "zaporizka", "luhanska", "autonomous-republic-of-crimea",
]

# ---------------------------------------------------------------------------
# Default color maps & label maps (override as needed)
# ---------------------------------------------------------------------------

PROP_COLORS_DEFAULT: Dict[str, str] = {
    "No Property":       "#a6cee3",
    "Unknown Status":    "#1f78b4",
    "Fully Damaged":     "#b2df8a",
    "Partially Damaged": "#33a02c",
    "Intact":            "#fb9a99",
}

GENDER_COLORS_DEFAULT: Dict[str, str] = {
    "f": "#e377c2",
    "m": "#17becf",
}

EDU_COLOR_MAP_DEFAULT: Dict[str, str] = {
    "No Education": "#8dd3c7",
    "Primary": "#ffffb3",
    "Secondary": "#bebada",
    "Vocational": "#fb8072",
    "Bachelor": "#80b1d3",
    "Specialist": "#fdb462",
    "Master Plus": "#b3de69",
}

AGE_COLOR_MAP_DEFAULT: Dict[str, str] = {
    "0-17":  "#1f77b4",
    "18-29": "#ff7f0e",
    "30-49": "#2ca02c",
    "50-64": "#d62728",
    "65+":   "#9467bd",
}

# Example label maps (replace with your real ones if needed)
PROP_LABELS_DEFAULT: Dict[Any, str] = {
    # Provide identity fallback; override with real mapping in caller.
    # e.g., 0:"No Property", 1:"Unknown Status", ...
}

EDU_LABELS_DEFAULT: Dict[Any, str] = {
    0: "No Education",
    1: "Primary",
    2: "Secondary",
    3: "Bachelor",
    4: "Vocational",
    5: "Specialist",
    6: "Master Plus",
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: Optional[Sequence[Tuple[str, str, Any]]]) -> pd.DataFrame:
    """Apply user-specified (col, op, val) filters to *df*.

    Supported ops: ==, !=, >, <, >=, <=.
    Missing columns are ignored with a warning.
    Returns a *new* filtered DataFrame.
    """
    if not filters:
        return df

    out = df
    for col, op, val in filters:
        if col not in out.columns:
            print(f"[WARN] Filter column '{col}' not in DataFrame; skipping.")
            continue
        if   op == "==": out = out[out[col] ==  val]
        elif op == "!=": out = out[out[col] !=  val]
        elif op == ">":  out = out[out[col] >   val]
        elif op == "<":  out = out[out[col] <   val]
        elif op == ">=": out = out[out[col] >=  val]
        elif op == "<=": out = out[out[col] <=  val]
        else:
            print(f"[WARN] Unsupported filter op '{op}' for column '{col}'; skipping.")
    return out


def _map_labels_if_provided(df: pd.DataFrame, col: str, label_map: Optional[Mapping[Any, str]]) -> pd.DataFrame:
    """Return a copy of *df* with *col* mapped via *label_map* if provided.
    If *label_map* is falsy, original DataFrame is returned unchanged.
    """
    if not label_map:
        return df
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].map(label_map).fillna(out[col].astype(str))
    return out


def compute_net_pivot(
    df: pd.DataFrame,
    group_col: str,
    *,
    dest_col: str = "destination",
    source_col: str = "source",
) -> pd.DataFrame:
    """Compute a destination×group_col pivot of *net arrivals* (arrivals − departures).

    Parameters
    ----------
    df : DataFrame containing at least *dest_col*, *source_col*, and *group_col*.
    group_col : Name of the categorical column to aggregate.
    dest_col, source_col : Column names for destination & source (customizable).

    Returns
    -------
    DataFrame (index=destination, columns=group categories, values=net counts).
    Missing categories/dests filled with 0.
    """
    if group_col not in df.columns or dest_col not in df.columns or source_col not in df.columns:
        missing = [c for c in (group_col, dest_col, source_col) if c not in df.columns]
        raise KeyError(f"Missing required columns: {missing}")

    # Count arrivals
    arr = (
        df.assign(_arrival=1)
          .groupby([dest_col, group_col])['_arrival']
          .sum()
          .rename('arrival')
          .reset_index()
    )
    # Count departures (rename source→destination to align)
    dep = (
        df.assign(_departure=1)
          .groupby([source_col, group_col])['_departure']
          .sum()
          .rename('departure')
          .reset_index()
          .rename(columns={source_col: dest_col})
    )

    # Merge & compute net
    net = pd.merge(arr, dep, on=[dest_col, group_col], how='outer')
    net['arrival'] = net['arrival'].fillna(0)
    net['departure'] = net['departure'].fillna(0)
    net['net'] = net['arrival'] - net['departure']

    pivot = (
        net.groupby([dest_col, group_col])['net']
           .sum()
           .unstack(fill_value=0)
    )
    return pivot


def average_pivots(per_run_tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Align, average across runs, and return the *mean* wide DataFrame.

    All missing values are filled with 0 before averaging.
    """
    if not per_run_tables:
        raise ValueError("No per-run tables supplied.")

    # Universe of destinations & categories
    all_dest = sorted(set().union(*(tbl.index for tbl in per_run_tables)))
    all_cats = sorted(set().union(*(tbl.columns for tbl in per_run_tables)))

    R, D, C = len(per_run_tables), len(all_dest), len(all_cats)
    arr_3d = np.zeros((R, D, C), dtype=float)
    for i, tbl in enumerate(per_run_tables):
        tbl_full = tbl.reindex(index=all_dest, columns=all_cats, fill_value=0)
        arr_3d[i] = tbl_full.values

    mean_df = pd.DataFrame(arr_3d.mean(axis=0), index=all_dest, columns=all_cats)
    return mean_df


def row_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Convert counts to row percentages (sum=100). Zero-rows remain zero."""
    row_sums = df.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = df.div(row_sums.replace({0: np.nan}), axis=0) * 100.0
    pct = pct.fillna(0.0)
    return pct


def tidy_destinations(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, underscore→hyphen, and filter to UKR_OBLASTS."""
    out = df.copy()
    out.index = (
        out.index.astype(str)
           .str.strip()
           .str.lower()
           .str.replace('_', '-', regex=False)
    )
    out = out.loc[out.index.isin(UKR_OBLASTS)]
    return out


def melt_for_plot(
    percent_df: pd.DataFrame,
    *,
    category_name: str,
    value_name: str = "percentage",
    pct_label_col: str = "pct_label",
) -> pd.DataFrame:
    """Wide→long melt + add formatted percent label column."""
    percent_df = percent_df.copy()
    percent_df.index.name = "destination"
    df_long = percent_df.reset_index().melt(
        id_vars="destination",
        var_name=category_name,
        value_name=value_name,
    )
    df_long[pct_label_col] = df_long[value_name].apply(lambda v: f"{v:.1f}%")
    return df_long


def make_stacked_bar(
    df_long: pd.DataFrame,
    *,
    category_name: str,
    color_map: Optional[Mapping[str, str]] = None,
    title: str = "Stacked Proportions",
    pretty_name: Optional[str] = None,
    value_name: str = "percentage",
    pct_label_col: str = "pct_label",
) -> "plotly.graph_objs._figure.Figure":
    """Build a 100%-stacked bar chart Figure from a long DataFrame."""
    if pretty_name is None:
        pretty_name = category_name.replace('_', ' ').title()

    fig = px.bar(
        df_long,
        x="destination",
        y=value_name,
        color=category_name,
        barmode="stack",
        color_discrete_map=color_map or {},
        text=pct_label_col,
        title=f"{pretty_name} Proportions by Destination (100 % stacked)",
        labels={
            value_name: "Proportion (%)",
            "destination": "Destination",
            category_name: pretty_name,
        },
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        xaxis_tickangle=-45,
        title_font_size=25,
        font=dict(size=18),
        legend=dict(font=dict(size=18)),
        legend_title_text=pretty_name,
        margin=dict(l=120, r=20, t=80, b=80),
        yaxis=dict(range=[0, 100], dtick=20, showgrid=True, gridcolor="LightGray"),
    )
    return fig


def save_plot(fig, html_name: str, plot_folder: Union[str, os.PathLike]) -> str:
    """Ensure *plot_folder* exists; write *fig* to *html_name*; return path."""
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    html_out = os.path.join(str(plot_folder), html_name)
    fig.write_html(html_out)
    print(f"[INFO] Saved HTML to {html_out}")
    return html_out

# ---------------------------------------------------------------------------
# Generic high-level pipeline
# ---------------------------------------------------------------------------

def stack_prop_plot(
    outdirs: Sequence[Union[str, os.PathLike]],
    *,
    group_col: str,
    filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
    label_map: Optional[Mapping[Any, str]] = None,
    color_map: Optional[Mapping[str, str]] = None,
    pretty_name: Optional[str] = None,
    html_basename: Optional[str] = None,
    plot_folder: Union[str, os.PathLike] = "plots",
    save_fig: bool = True,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """Full pipeline: read logs, filter, net arrivals, average, % normalize, plot.

    Parameters mirror those in the original family of functions. The return value
    is the (wide) *percent_df* (dest × category) table, or *None* if no data.
    """
    if filters is None or filters == ["None"]:
        filters = []

    per_run_tables: List[pd.DataFrame] = []

    for run_dir in outdirs:
        log_path = os.path.join(run_dir, "migration.log")
        if not os.path.exists(log_path):
            if verbose:
                print(f"[INFO] Missing migration.log in {run_dir}; skipping.")
            continue

        df = pd.read_csv(log_path)
        df = apply_filters(df, filters)
        df = _map_labels_if_provided(df, group_col, label_map)

        # compute net pivot; skip run if fails
        try:
            pivot = compute_net_pivot(df, group_col)
        except KeyError as exc:  # missing columns
            if verbose:
                print(f"[WARN] {exc}; skipping run {run_dir}.")
            continue
        per_run_tables.append(pivot)

    if not per_run_tables:
        if verbose:
            print(f"[INFO] No data for {group_col} proportions; skipping plot.")
        return None

    # Average across runs & convert to percentages
    mean_df = average_pivots(per_run_tables)
    percent_df = row_percent(mean_df)

    # Clean destination names + filter to oblasts
    percent_df = tidy_destinations(percent_df)
    if percent_df.empty:
        if verbose:
            print(f"[INFO] {group_col} plot: nothing left after oblast filter.")
        return None

    # Melt + plot
    df_long = melt_for_plot(percent_df, category_name=group_col)
    fig = make_stacked_bar(
        df_long,
        category_name=group_col,
        color_map=color_map,
        title="",  # make_stacked_bar constructs the title
        pretty_name=pretty_name,
    )

    # Save
    if save_fig:
        if html_basename is None:
            html_basename = f"stacked_bar_{group_col}_proportions.html"
        save_plot(fig, html_basename, plot_folder)

    return percent_df

# ---------------------------------------------------------------------------
# Thin wrappers matching the original API names
# ---------------------------------------------------------------------------

def plotStackedBarProportions_Property(
    outdirs: Sequence[Union[str, os.PathLike]],
    filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
    save_fig: bool = True,
    plot_folder: Union[str, os.PathLike] = "plots",
    *,
    prop_labels: Optional[Mapping[Any, str]] = None,
    prop_colors: Optional[Mapping[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Drop-in replacement for the original property-status plot function."""
    return stack_prop_plot(
        outdirs,
        group_col="property_in_ukraine",
        filters=filters,
        label_map=prop_labels or PROP_LABELS_DEFAULT,
        color_map=prop_colors or PROP_COLORS_DEFAULT,
        pretty_name="Property in Ukraine",
        html_basename="stacked_bar_property_proportions.html",
        plot_folder=plot_folder,
        save_fig=save_fig,
    )


def plotStackedBarProportions_Gender(
    outdirs: Sequence[Union[str, os.PathLike]],
    filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
    save_fig: bool = True,
    plot_folder: Union[str, os.PathLike] = "plots",
    *,
    gender_colors: Optional[Mapping[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Drop-in replacement for the original gender plot function."""
    return stack_prop_plot(
        outdirs,
        group_col="gender",
        filters=filters,
        label_map=None,  # assume values already 'f'/'m' or text
        color_map=gender_colors or GENDER_COLORS_DEFAULT,
        pretty_name="Gender (f / m)",
        html_basename="stacked_bar_gender_proportions.html",
        plot_folder=plot_folder,
        save_fig=save_fig,
    )


def plotStackedBarProportions_Education(
    outdirs: Sequence[Union[str, os.PathLike]],
    filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
    save_fig: bool = True,
    plot_folder: Union[str, os.PathLike] = "plots",
    *,
    edu_labels: Optional[Mapping[Any, str]] = None,
    edu_colors: Optional[Mapping[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Drop-in replacement for the original education plot function."""
    return stack_prop_plot(
        outdirs,
        group_col="education",
        filters=filters,
        label_map=edu_labels or EDU_LABELS_DEFAULT,
        color_map=edu_colors or EDU_COLOR_MAP_DEFAULT,
        pretty_name="Education",
        html_basename="stacked_bar_education_proportions.html",
        plot_folder=plot_folder,
        save_fig=save_fig,
    )


def plotStackedBarProportions_Age(
    outdirs: Sequence[Union[str, os.PathLike]],
    filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
    save_fig: bool = True,
    plot_folder: Union[str, os.PathLike] = "plots",
    *,
    age_colors: Optional[Mapping[str, str]] = None,
    # If caller already supplies age bins & labels, skip auto-binning by calling the generic
) -> Optional[pd.DataFrame]:
    """Convenience wrapper expecting pre-binned age categories in column 'age_binned'.

    If your logs only have raw 'age', call `plotStackedBarProportions_AgeFromRaw`.
    """
    return stack_prop_plot(
        outdirs,
        group_col="age_binned",
        filters=filters,
        label_map=None,
        color_map=age_colors or AGE_COLOR_MAP_DEFAULT,
        pretty_name="Age Group",
        html_basename="stacked_bar_age_binned_proportions.html",
        plot_folder=plot_folder,
        save_fig=save_fig,
    )

# ---------------------------------------------------------------------------
# Age binning from raw 'age' column
# ---------------------------------------------------------------------------

AGE_BINS = [0, 17, 29, 49, 64, 200]
AGE_LABELS = ["0-17", "18-29", "30-49", "50-64", "65+"]

def _ensure_age_binned(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'age_binned' column (categorical) from 'age' if present."""
    if 'age_binned' in df.columns:
        return df
    if 'age' not in df.columns:
        raise KeyError("No 'age' column available to create age_binned")
    out = df.copy()
    out['age_binned'] = pd.cut(out['age'], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    return out


def plotStackedBarProportions_AgeFromRaw(
    outdirs: Sequence[Union[str, os.PathLike]],
    filters: Optional[Sequence[Tuple[str, str, Any]]] = None,
    save_fig: bool = True,
    plot_folder: Union[str, os.PathLike] = "plots",
    *,
    age_colors: Optional[Mapping[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Like `plotStackedBarProportions_Age`, but bins raw 'age' within this call."""
    if filters is None or filters == ["None"]:
        filters = []

    per_run_tables: List[pd.DataFrame] = []

    for run_dir in outdirs:
        log_path = os.path.join(run_dir, "migration.log")
        if not os.path.exists(log_path):
            print(f"[INFO] Missing migration.log in {run_dir}; skipping.")
            continue

        df = pd.read_csv(log_path)
        df = apply_filters(df, filters)
        try:
            df = _ensure_age_binned(df)
        except KeyError as exc:
            print(f"[WARN] {exc}; skipping run {run_dir}.")
            continue

        try:
            pivot = compute_net_pivot(df, 'age_binned')
        except KeyError as exc:
            print(f"[WARN] {exc}; skipping run {run_dir}.")
            continue
        per_run_tables.append(pivot)

    if not per_run_tables:
        print("[INFO] No data for age proportions; skipping plot.")
        return None

    mean_df = average_pivots(per_run_tables)
    percent_df = row_percent(mean_df)
    percent_df = tidy_destinations(percent_df)
    if percent_df.empty:
        print("[INFO] Age plot: nothing left after oblast filter.")
        return None

    df_long = melt_for_plot(percent_df, category_name='age_binned')
    fig = make_stacked_bar(
        df_long,
        category_name='age_binned',
        color_map=age_colors or AGE_COLOR_MAP_DEFAULT,
        pretty_name="Age Group",
    )
    if save_fig:
        save_plot(fig, "stacked_bar_age_proportions.html", plot_folder)
    return percent_df