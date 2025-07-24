from __future__ import annotations
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go


def plotMigrationSankey(outdirs, *, save_fig=True, plot_folder="plots"):
    flows = []
    for d in outdirs:
        log = Path(d) / "migration.log"
        if log.exists():
            df = pd.read_csv(log)
            flows.append(df.groupby(["source", "destination"])
                           .size().reset_index(name="cnt"))
    if not flows:
        print("[INFO] No migration.log files found.")
        return

    mean_flows = (pd.concat(flows)
                    .groupby(["source", "destination"])["cnt"]
                    .mean().reset_index(name="mean_cnt"))

    origins = sorted(mean_flows["source"].unique())
    dests   = sorted(mean_flows["destination"].unique())
    labels  = origins + dests
    index   = {lbl: i for i, lbl in enumerate(labels)}

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=20,
                  line=dict(width=0.5)),
        link=dict(
            source=[index[s] for s in mean_flows["source"]],
            target=[index[t] for t in mean_flows["destination"]],
            value=mean_flows["mean_cnt"].tolist(),
        ),
    ))
    fig.update_layout(title_text="Mean Migration Sankey", font_size=10)

    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    fig.write_html(Path(plot_folder) / "migration_sankey_mean.html")
    if save_fig:
        try:
            fig.write_image(Path(plot_folder) / "migration_sankey_mean.png")
        except Exception as e:
            print(f"[WARNING] PNG export failed: {e}")