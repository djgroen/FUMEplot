from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import plotly.express as px
from dateutil.relativedelta import relativedelta



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

GENDER_COLORS: dict[str, str] = {
    "f": "#e05779",   # pinkish
    "m": "#4a78b5",   # steel‑blue
}

GENDER_NAMES: dict[str, str] = {
    "f": "Female",
    "m": "Male",
}

DISAGG_LABELS: dict[str, str] = {
    "age":                 "Age",
    "age_binned":          "Age",
    "gender":              "Gender",
    "education":           "Education",
    "property_in_ukraine": "Property In Ukraine",
}

def pretty(name: str) -> str:
    """
    Convert a raw column/disaggregator key to a human‑readable label
    (used in chart titles and axis labels).
    """
    return DISAGG_LABELS.get(name, name.replace("_", " ").title())
# ---------------------------------------------------------------------------------

def _prep_runs(outdirs, agg_col, primary_col, primary_val, extra_filters):
    per_run = []
    for d in outdirs:
        fp = Path(d) / "migration.log"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if primary_val is not None:
            df = df[df[primary_col] == primary_val]
        for col, op, val in extra_filters:
            if op == "==":
                df = df[df[col] == val]
            elif op == ">":
                df = df[df[col] > val]
            elif op == "<":
                df = df[df[col] < val]
        per_run.append(df.groupby(["time", agg_col])
                         .size().unstack(fill_value=0))
    return per_run


def plotLineOverTime(outdirs, *, primary_filter_column="source",
                     primary_filter_value=None, line_disaggregator="gender",
                     filters=None, save_fig=True, plot_folder="plots",
                     show_quartiles=True):
    """
    Median ± IQR fan plot of net returns through time.

    line_disaggregator ∈ {"gender","age_binned","education",
                          "property_in_ukraine"}
    """
    if filters is None:
        filters = []
    if isinstance(line_disaggregator, (list, tuple)):
        line_disaggregator = line_disaggregator[0]

    # derive / build the correct aggregator column
    if line_disaggregator == "age_binned":
        raw_runs = _prep_runs(outdirs, "age", primary_filter_column,
                              primary_filter_value, filters)
        # add bins on‑the‑fly
        bins   = [0, 17, 29, 49, 64, 200]
        labels = ["0-17", "18-29", "30-49", "50-64", "65+"]
        new_runs = []
        for df in raw_runs:
            df = df.copy()
            df["age_bin"] = pd.cut(df.index.get_level_values("time") * 0 + 1,
                                   bins=[0, 5, 10, 15, 20, 25],  # dummy split
                                   labels=labels)
            new_runs.append(df.groupby(["time", "age_bin"])
                              .size().unstack(fill_value=0))
        runs = new_runs
        agg = "Age"
    else:
        runs = _prep_runs(outdirs, line_disaggregator,
                          primary_filter_column, primary_filter_value, filters)
        agg = pretty(line_disaggregator)

    if not runs:
        print("[INFO] No data to plot.")
        return

    times = sorted(set().union(*(df.index for df in runs)))
    cats  = sorted(set().union(*(df.columns for df in runs)))

    R, T, C = len(runs), len(times), len(cats)
    cube = np.zeros((R, T, C))
    for i, df in enumerate(runs):
        cube[i] = df.reindex(index=times, columns=cats, fill_value=0).values

    med = np.median(cube, axis=0)
    q25 = np.percentile(cube, 25, axis=0)
    q75 = np.percentile(cube, 75, axis=0)

    # nice labels
    if line_disaggregator == "education":
        lb_map = EDU_LABELS
    elif line_disaggregator == "property_in_ukraine":
        lb_map = PROP_LABELS
    elif line_disaggregator == "gender":
        lb_map = GENDER_NAMES
    else:
        lb_map = {}

    rows = []
    for ti, t in enumerate(times):
        for cj, cat in enumerate(cats):
            rows.append({
                "time": t,
                agg: lb_map.get(cat, cat),
                "median": med[ti, cj],
                "q25": q25[ti, cj],
                "q75": q75[ti, cj],
            })
    dfp = pd.DataFrame(rows)
    start_dt = pd.to_datetime("2025-03-01")
    dfp["month"] = dfp["time"].astype(int).apply(
        lambda m: start_dt + relativedelta(months=m)
    )

    color_map = ({GENDER_NAMES["f"]: GENDER_COLORS["f"],
                  GENDER_NAMES["m"]: GENDER_COLORS["m"]}
                 if line_disaggregator == "gender" else None)

    fig = px.line(dfp, x="month", y="median",
                  color=agg, color_discrete_map=color_map,
                  labels={"median": "Net No. of Returnees (×100)"},
                  title=f"Returnees Over Time "
                        f"({primary_filter_column}="
                        f"{primary_filter_value or 'All'})")
    if show_quartiles:
        for name in dfp[agg].unique():
            sub = dfp[dfp[agg] == name]
            fig.add_traces(
                px.scatter(sub, x="month", y="q25")
                .update_traces(mode="lines", line=dict(width=0),
                               showlegend=False).data
            )
            fig.add_traces(
                px.scatter(sub, x="month", y="q75")
                .update_traces(mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor="rgba(0,0,0,0.1)",
                               showlegend=False).data
            )

    fig.update_xaxes(dtick="M1", tickformat="%b %Y", tickangle=45)
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    fig.write_html(Path(plot_folder) /
                   f"fan_{primary_filter_column}_{primary_filter_value}_"
                   f"{line_disaggregator}.html")