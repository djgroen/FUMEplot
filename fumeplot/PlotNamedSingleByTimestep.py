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

# font = {'family': 'serif',
#         'weight': 'bold',
#         'size': 12}
# plt.rc('font', **font)
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['xtick.labelsize'] = 10
# plt.rcParams['ytick.labelsize'] = 10
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['legend.handlelength'] = 1.5
# plt.rcParams['legend.handleheight'] = 0.5
# plt.rcParams['legend.borderpad'] = 0.5
# plt.rcParams['legend.borderaxespad'] = 0.5
# plt.rcParams['legend.frameon'] = True
# plt.rcParams['legend.framealpha'] = 0.5
# plt.rcParams['legend.loc'] = 'best'
# plt.rcParams['legend.labelspacing'] = 0.5
# plt.rcParams['legend.columnspacing'] = 1.0
# plt.rcParams['legend.markerscale'] = 1.0

import latexplotlib as lpl
lpl.style.use('latex12pt')
latex_doc_size = (347.12354, 549.138) 
columns_per_page = 1.
latex_doc_size = tuple(x / columns_per_page for x in latex_doc_size)
lpl.size.set(*latex_doc_size)

EDU_LABELS = {
    0: "No Education",
    1: "Primary",
    2: "Secondary",
    4: "Vocational",
    3: "Bachelor",      # adjust if your code→label mapping differs
    5: "Specialist",
    6: "Master Plus"
}

PROP_LABELS = {
    0: "No Property",
    1: "Unknown Status",
    2: "Fully Damaged",
    3: "Partially Damaged",
    5: "Intact"
}

# give every disaggregator a nice display name
DISAGG_LABELS = {
    'age':                'Age',
    'age_binned':         'Age',
    'gender':             'Gender',
    'education':          'Education',
    'property_in_ukraine':'Property In Ukraine'
}

# sequential palette for Age
AGE_PALETTE = px.colors.sequential.Plasma  # or Viridis, Turbo, etc.

GENDER_COLORS = {
    "f": "#e05779",    # pinkish
    "m": "#4a78b5"     # steel‐blue
}

GENDER_NAMES = {
    "f": "Female",
    "m": "Male"
}
def pretty(name: str) -> str:
    """Turn a raw aggregator key into its user-friendly label."""
    return DISAGG_LABELS.get(name, name.replace('_',' ').title())

def _formatLabels(labels):

    # Capitalize and replace underscores with spaces
    labels =  [label.replace('_', ' ').title() for label in labels]

    # Leave only the last words of the labels
    labels = [label.split()[-1] for label in labels]

    return labels

def plotCounts(plot_num, all_counts, save_fig, plot_folder, combine_plots_pdf):
    # Combine counts into a DataFrame, filling missing values with 0
    combined_counts = pd.DataFrame(all_counts).fillna(0)

    # Compute mean and standard deviation for each source
    mean_counts = combined_counts.mean()
    std_counts = combined_counts.std()

    # Plot histogram with error bars
    # #fig= plt.figure(plot_num+1, figsize=(10,6))
    # #ax = fig.add_subplot(111)
    fig, ax = plt.subplots(num=plot_num+1, figsize=(10,6))
    # - LatexPlotLib version
    #fig, ax = lpl.subplots(num=plot_num+1)

    # plt.bar(mean_counts.index, 
    #         mean_counts.values, 
    #         yerr=std_counts.values, 
    #         color='skyblue', 
    #         capsize=5,
    #         )

    ax.boxplot(
                combined_counts,
                #tick_labels=_formatLabels(mean_counts.index), # matplotlib >=3.9
                labels=_formatLabels(mean_counts.index), # matplotlib <3.9
                patch_artist=True, 
                #notch=True,
                bootstrap=1000,
                showmeans=True, 
                meanline=True, 
                showbox=True,
                showcaps=True,
                showfliers=True,
                medianprops=dict(color='blue', linewidth=0.85),
                meanprops=dict(color='black', linewidth=0.85, linestyle='dashed'),
                boxprops=dict(facecolor='skyblue', linewidth=0.85),
                whiskerprops=dict(color='blue', linewidth=0.75),
                capprops=dict(color='blue', linewidth=0.75),
                capwidths=0.3,
                flierprops=dict(marker='o', markerfacecolor='blue', markersize=2, linestyle='none'),
                #label='Entries', # matplotlib >=3.9
                )

    # Labels and title
    ax.legend(
        [mlines.Line2D([], [], color='black', linestyle='dashed', label='Mean'),
         mlines.Line2D([], [], color='blue', label='Median'),
         mpatches.Patch(color='skyblue', label='Q1-Q3'),
         mlines.Line2D([], [], marker='|', color='blue', linestyle='-', label='One and half inter-quartile range'),
         mlines.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Outliers')], 
        ['Mean', 'Median', '25th-75th percentile (IQR)', r'$\pm$ $1.5$ IQR', 'Outliers'],
        # ['Mean', 'Median', '25th-75th percentile (IQR)', '1.5 IQR', 'Outliers'],
        loc='upper right',
        )
    
    ax.set_xlabel('Source Location')
    ax.set_ylabel('Number of Entries')
    ax.set_title('Boxplot of Entries Grouped by Source')
    ax.tick_params('x', rotation=60)
    ax.grid(axis='y', linestyle='-', linewidth=0.33)
    # plt.axis('tight')  # Uncommented to set axis to tight

    # - LatexPlotLib version
    #lpl.tight_layout()  # Uncommented to apply tight layout

    # save plot    import os
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from pathlib import Path
    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
        print(f"Saved plot {plot_num} to PDF.")
    if save_fig:
        fig.savefig(plot_folder+'/EntriesBySource.png')
    #plt.close()

def _getFilteredCounts(csv_files, query, var_type="str"):

    parts = query.split(" ")
    col = parts[0]
    comparison_operator = parts[1]
    val = parts[2]
    if var_type == "int":
        var = int(parts[2])
    if var_type == "float":
        var = float(parts[2])

    all_counts = []
    for file in csv_files:
        df = pd.read_csv(file)

        if comparison_operator == "==":
            df = df[df[col] == val]
        elif comparison_operator == ">":
            df = df[df[col] > val]
        elif comparison_operator == "<":
            df = df[df[col] < val]
        else:
            print(f"ERROR: unsupported comparison operator {comparison_operator} in _getFilteredCounts.", file=sys.stderr)
            sys.exit()

        source_counts = df['source'].value_counts()
        all_counts.append(source_counts)

    return all_counts

def plotSourceHist(outdirs, filters, save_fig=False, plot_folder=None, combine_plots_pdf=False):

    # Read and aggregate data from multiple CSV files
    all_counts = []
    
    csv_files = []
    for d in outdirs:
        csv_files.append(f"{d}/migration.log")

    for file in csv_files:
        df = pd.read_csv(file)
        source_counts = df['source'].value_counts()
        all_counts.append(source_counts)

    with PdfPages(os.path.join(plot_folder, "combined_bar_plots.pdf")) if combine_plots_pdf else nullcontext() as pdf_pages:    
   
        plotCounts(0, all_counts, save_fig, plot_folder, combine_plots_pdf=pdf_pages)

        i = 1
        for f in filters:
            all_counts = _getFilteredCounts(csv_files, f)
            plotCounts(i, all_counts, save_fig, plot_folder, combine_plots_pdf=pdf_pages)
            i += 1

    if not combine_plots_pdf:
        plt.show()
    #plt.close()
    
def plotMigrationSankey(outdirs, save_fig=True, plot_folder="plots"): # CHANGE SAVEFIG AND PLOTFOLDER TO NOT HARDCODE
    """
    Creates a Sankey diagram showing the *average* number of migrations
    per origin→destination link, aggregated across all ensemble runs.

    Each run’s migrations.log is read, row-counts for each (source, destination)
    pair are computed, and then we take the mean count for each link.
    """
    # 1. Read each run’s migration.log and compute link counts
    per_run_flows = []  # will hold one DataFrame per run
    for run_dir in outdirs:
        log_path = os.path.join(run_dir, "migration.log")
        if not os.path.exists(log_path):
            print(f"[WARNING] No migration.log in {run_dir}, skipping.")
            continue

        # read the raw log
        df = pd.read_csv(log_path)

        # count how many rows for each (source, destination) in current run
        flows_this_run = (
            df
            .groupby(['source', 'destination'])
            .size()
            .reset_index(name='count')  # 'count' is just the number of rows
        )
        per_run_flows.append(flows_this_run)
    
    # if nothing was read, send error
    if not per_run_flows:
        print("[INFO] No valid migration.log files found.")
        return
    
    # 2. Concatenate all runs and compute mean per link
    all_flows = pd.concat(per_run_flows, ignore_index=True)

    mean_flows = (
        all_flows
        .groupby(['source', 'destination'])['count']
        .mean()
        .reset_index(name='mean_count')  # average count across runs
    )
    
    # 3. Build node lists and index mapping
    origin_list = sorted(mean_flows['source'].unique())
    region_list = sorted(mean_flows['destination'].unique())
    node_labels = origin_list + region_list
    label_to_idx = {lbl: i for i, lbl in enumerate(node_labels)}
    
    # 4. Turn each row of mean_flows into a Sankey link
    src_indices = []
    tgt_indices = []
    values     = []
    for _, row in mean_flows.iterrows():
        s = label_to_idx[row['source']]
        t = label_to_idx[row['destination']]
        v = row['mean_count']
        src_indices.append(s)
        tgt_indices.append(t)
        values.append(v)
    
    # 5. Manually position nodes: origins left, destinations right
    n_orig = len(origin_list)
    n_dest = len(region_list)
    node_x, node_y = [], []
    # spread origins evenly at x=0.1
    for i in range(n_orig):
        node_x.append(0.1)
        node_y.append((i+1)/(n_orig+1))
    # spread destinations at x=0.9
    for i in range(n_dest):
        node_x.append(0.9)
        node_y.append((i+1)/(n_dest+1))
        
    # 6. Build the Plotly Sankey
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad=15, thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels, x=node_x, y=node_y
        ),
        link = dict(
            source=src_indices,
            target=tgt_indices,
            value=values
        )
    )])

    fig.update_layout(
        title_text="Mean Migration Sankey (across ensemble runs)",
        font_size=10
    )
    
    # 7. Save outputs and create png
    os.makedirs(plot_folder, exist_ok=True)

    html_path = os.path.join(plot_folder, "migration_sankey_mean.html")
    fig.write_html(html_path)
    print(f"[INFO] Sankey HTML saved to {html_path}")

    if save_fig:
        png_path = os.path.join(plot_folder, "migration_sankey_mean.png")
        try:
            fig.write_image(png_path)
            print(f"[INFO] Sankey PNG saved to {png_path}")
        except Exception as e:
            print(f"[ERROR] Couldn't write PNG: {e}")
            
import os
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

# colour mapping for property status
PROP_COLORS = {
    "No Property":       "#a6cee3",
    "Unknown Status":    "#1f78b4",
    "Fully Damaged":     "#b2df8a",
    "Partially Damaged": "#33a02c",
    "Intact":            "#fb9a99"
}

def plotStackedBarProportions_Property(outdirs,
                                       filters=None,
                                       save_fig=True,
                                       plot_folder="plots"):
    """
    100 %-stacked bar of net-arrival *property_in_ukraine* proportions
    for each destination, averaged across ensemble runs.

    Saves HTML to  <plot_folder>/stacked_bar_property_proportions.html
    """
    import pandas as pd, numpy as np, os
    from pathlib import Path
    import plotly.express as px

    if filters is None or filters == ["None"]:
        filters = []

    per_run_tables = []

    
    for d in outdirs:
        fp = os.path.join(d, "migration.log")
        if not os.path.exists(fp):
            continue

        df = pd.read_csv(fp)

        # user filters
        for col, op, val in filters:
            if   op == "==": df = df[df[col] ==  val]
            elif op == "!=": df = df[df[col] !=  val]
            elif op == ">":  df = df[df[col] >   val]
            elif op == "<":  df = df[df[col] <   val]
            elif op == ">=": df = df[df[col] >=  val]
            elif op == "<=": df = df[df[col] <=  val]

        if "property_in_ukraine" not in df.columns or "destination" not in df.columns:
            continue

        df["property_in_ukraine"] = df["property_in_ukraine"].map(PROP_LABELS)

        arr = (df.assign(arrival=1)
                 .groupby(["destination", "property_in_ukraine"])["arrival"]
                 .sum().reset_index())

        dep = (df.assign(departure=1)
                 .groupby(["source", "property_in_ukraine"])["departure"]
                 .sum().reset_index()
                 .rename(columns={"source": "destination"}))

        net = (pd.merge(arr, dep,
                        on=["destination", "property_in_ukraine"],
                        how="outer")
                 .fillna(0))
        net["net"] = net["arrival"] - net["departure"]

        pivot = (net.groupby(["destination", "property_in_ukraine"])["net"]
                       .sum().unstack(fill_value=0))
        per_run_tables.append(pivot)

    if not per_run_tables:
        print("[INFO] No data for property proportions; skipping plot.")
        return

    all_dest = sorted(set().union(*(t.index   for t in per_run_tables)))
    all_prop = sorted(set().union(*(t.columns for t in per_run_tables)))

    R, D, C = len(per_run_tables), len(all_dest), len(all_prop)
    arr_3d  = np.zeros((R, D, C), dtype=float)
    for i, tbl in enumerate(per_run_tables):
        arr_3d[i] = tbl.reindex(index=all_dest,
                                columns=all_prop,
                                fill_value=0).values

    mean_df    = pd.DataFrame(arr_3d.mean(axis=0), index=all_dest, columns=all_prop)
    percent_df = mean_df.div(mean_df.sum(axis=1), axis=0) * 100

    # tidy destination names
    percent_df.index = (percent_df.index.astype(str)
                                         .str.strip()
                                         .str.lower()
                                         .str.replace("_", "-"))

    # keep Ukrainian oblasts
    ukr_oblasts = [
        "kyivska","zakarpatska","ivano-frankivska","ternopilska","rivnenska",
        "volynska","zhytomyrska","khmelnytska","vinnytska","chernivetska",
        "kyiv","chernihivska","sumska","cherkaska","poltavska","kharkivska",
        "dnipropetrovska","kirovohradska","odeska","mykolaiivska","khersonska",
        "donetska","zaporizka","luhanska","autonomous-republic-of-crimea"
    ]
    percent_df = percent_df.loc[percent_df.index.isin(ukr_oblasts)]
    if percent_df.empty:
        print("[INFO] Property plot: nothing left after oblast filter.")
        return

    df_long = percent_df.reset_index()
    df_long = df_long.rename(columns={df_long.columns[0]: "destination"})
    df_long = df_long.melt(
        id_vars="destination",
        var_name="property",
        value_name="percentage"
    )
    df_long["pct_label"] = df_long["percentage"].apply(lambda v: f"{v:.1f}%")

    fig = px.bar(
        df_long,
        x="destination",
        y="percentage",
        color="property",
        barmode="stack",
        color_discrete_map=PROP_COLORS,
        text="pct_label",
        title="Property-Status Proportions by Destination (100 % stacked)",
        labels={"percentage": "Proportion (%)",
                "destination": "Destination",
                "property": "Property in Ukraine"}
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        xaxis_tickangle=-45,
        title_font_size=25,
        font=dict(size=18),
        legend=dict(font=dict(size=18)),
        margin=dict(l=120, r=20, t=80, b=80),
        yaxis=dict(range=[0, 100],
                   dtick=20,
                   showgrid=True,
                   gridcolor="LightGray")
    )

    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    html_out = os.path.join(plot_folder, "stacked_bar_property_proportions.html")
    if save_fig:
        fig.write_html(html_out)
        print(f"[INFO] Saved property-proportions plot to {html_out}")


def plotStackedBarProportions_Gender(outdirs,
                                     filters=None,
                                     save_fig=True,
                                     plot_folder="plots"):
    """
    100 %-stacked bar of net-arrival gender proportions (f / m)
    for every destination, averaged across all ensemble runs.
    """
    import pandas as pd, numpy as np, os
    from pathlib import Path
    import plotly.express as px

    if filters is None or filters == ["None"]:
        filters = []

    
    per_run_tables = []
    for d in outdirs:
        fp = os.path.join(d, "migration.log")
        if not os.path.exists(fp):
            continue

        df = pd.read_csv(fp)

        # user filters
        for col, op, val in filters:
            if   op == "==": df = df[df[col] ==  val]
            elif op == "!=": df = df[df[col] !=  val]
            elif op == ">":  df = df[df[col] >   val]
            elif op == "<":  df = df[df[col] <   val]
            elif op == ">=": df = df[df[col] >=  val]
            elif op == "<=": df = df[df[col] <=  val]

        if "gender" not in df.columns or "destination" not in df.columns:
            continue

        arr = (df.assign(arrival=1)
                 .groupby(["destination", "gender"])["arrival"]
                 .sum().reset_index())

        dep = (df.assign(departure=1)
                 .groupby(["source", "gender"])["departure"]
                 .sum().reset_index()
                 .rename(columns={"source": "destination"}))

        net = (pd.merge(arr, dep, on=["destination", "gender"], how="outer")
                 .fillna(0))
        net["net"] = net["arrival"] - net["departure"]

        pivot = (net.groupby(["destination", "gender"])["net"]
                       .sum().unstack(fill_value=0))
        per_run_tables.append(pivot)

    if not per_run_tables:
        print("[INFO] No data for gender proportions; skipping plot.")
        return

    # 
    all_dest = sorted(set().union(*(t.index   for t in per_run_tables)))
    all_gen  = sorted(set().union(*(t.columns for t in per_run_tables)))

    R, D, C = len(per_run_tables), len(all_dest), len(all_gen)
    arr_3d  = np.zeros((R, D, C), dtype=float)
    for i, tbl in enumerate(per_run_tables):
        arr_3d[i] = tbl.reindex(index=all_dest,
                                columns=all_gen,
                                fill_value=0).values

    mean_df    = pd.DataFrame(arr_3d.mean(axis=0), index=all_dest, columns=all_gen)
    percent_df = mean_df.div(mean_df.sum(axis=1), axis=0) * 100

    
    percent_df.index = (percent_df.index.astype(str)
                                         .str.strip()
                                         .str.lower()
                                         .str.replace("_", "-"))

    ukr_oblasts = [
        "kyivska","zakarpatska","ivano-frankivska","ternopilska","rivnenska",
        "volynska","zhytomyrska","khmelnytska","vinnytska","chernivetska",
        "kyiv","chernihivska","sumska","cherkaska","poltavska","kharkivska",
        "dnipropetrovska","kirovohradska","odeska","mykolaiivska","khersonska",
        "donetska","zaporizka","luhanska","autonomous-republic-of-crimea"
    ]
    percent_df = percent_df.loc[percent_df.index.isin(ukr_oblasts)]
    if percent_df.empty:
        print("[INFO] Gender plot: nothing left after oblast filter.")
        return

    
    df_long = percent_df.reset_index()                          # column 'index'
    df_long = df_long.rename(columns={df_long.columns[0]: "destination"})
    df_long = df_long.melt(id_vars="destination",
                           var_name="gender",
                           value_name="percentage")
    df_long["pct_label"] = df_long["percentage"].apply(lambda v: f"{v:.1f}%")

    
    fig = px.bar(
        df_long,
        x="destination",
        y="percentage",
        color="gender",
        barmode="stack",
        color_discrete_map=GENDER_COLORS,
        text="pct_label",
        title="Gender Proportions by Destination (100 % stacked)",
        labels={"percentage": "Proportion (%)",
                "destination": "Destination",
                "gender": "Gender (f / m)"}
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        xaxis_tickangle=-45,
        title_font_size=25,
        font=dict(size=18),
        legend=dict(font=dict(size=18)),
        margin=dict(l=120, r=20, t=80, b=80),
        yaxis=dict(range=[0, 100],
                   dtick=20,
                   showgrid=True,
                   gridcolor="LightGray")
    )

    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    html_out = os.path.join(plot_folder, "stacked_bar_gender_proportions.html")
    if save_fig:
        fig.write_html(html_out)
        print(f"[INFO] Saved gender-proportions plot to {html_out}")

# A mapping from each education label to a specific color for the stacked bars
color_map = {
    "No Education": "#8dd3c7",
    "Primary": "#ffffb3",
    "Secondary": "#bebada",
    "Vocational": "#fb8072",
    "Bachelor": "#80b1d3",
    "Specialist": "#fdb462",
    "Master Plus": "#b3de69"
}

EDU_LABELS = {
    0: "No Education",
    1: "Primary",
    2: "Secondary",
    4: "Vocational",
    3: "Bachelor",
    5: "Specialist",
    6: "Master Plus"
}

def plotStackedBarProportions_Education(outdirs, filters=None, save_fig=True, plot_folder="plots"):
    """
    Reads every migration.log in `outdirs`, applies optional filters,
    then groups by `destination` and the 'education' column.
    For each run, it builds a destination×education table of NET arrivals (arrivals minus departures).
    Averages those tables across runs, normalizes each row to sum to 100%,
    and plots a 100%‐stacked bar chart of percentages by destination.
    """
    if filters is None or filters == ["None"]:
        filters = []

    per_run_tables = []

    for run_dir in outdirs:
        log_path = os.path.join(run_dir, "migration.log")
        if not os.path.exists(log_path):
            continue

        df = pd.read_csv(log_path)

        # Apply any user‐specified filters
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

        agg_col = "education"

        # Abort if required columns are missing
        if agg_col not in df.columns or 'destination' not in df.columns:
            return

        # Map education codes to readable labels
        df['education'] = df['education'].map(EDU_LABELS)

        # Count arrivals by (destination × education)
        arr = (
            df.assign(arrival=1)
              .groupby(["destination", agg_col])["arrival"]
              .sum()
              .reset_index()
        )
        # Count departures by (source × education), rename "source" → "destination"
        dep = (
            df.assign(departure=1)
              .groupby(["source", agg_col])["departure"]
              .sum()
              .reset_index()
              .rename(columns={"source": "destination"})
        )
        # Merge, fill NaNs with 0, compute net = arrival − departure
        net = pd.merge(arr, dep, on=["destination", agg_col], how="outer")
        net["arrival"]   = net["arrival"].fillna(0)
        net["departure"] = net["departure"].fillna(0)
        net["net"]  = net["arrival"] - net["departure"]

        # Pivot: rows = destination, columns = education label, values = net
        pivot = net.groupby(["destination", agg_col])["net"].sum().unstack(fill_value=0)
        per_run_tables.append(pivot)

    if not per_run_tables:
        return

    # Align all runs onto the same set of destinations & education categories
    all_dest = sorted(set().union(*(tbl.index for tbl in per_run_tables)))
    all_cats = sorted(set().union(*(tbl.columns for tbl in per_run_tables)))

    R, D, C = len(per_run_tables), len(all_dest), len(all_cats)
    arr_3d = np.zeros((R, D, C), dtype=float)
    for i, tbl in enumerate(per_run_tables):
        tbl_full = tbl.reindex(index=all_dest, columns=all_cats, fill_value=0)
        arr_3d[i] = tbl_full.values

    # Compute mean across runs
    mean_df = pd.DataFrame(arr_3d.mean(axis=0), index=all_dest, columns=all_cats)
    # Normalize each row to sum to 100%
    percent_df = mean_df.div(mean_df.sum(axis=1), axis=0) * 100
    percent_df.index = (
        percent_df.index.astype(str).str.strip().str.lower().str.replace("_", "-")
    )

    # Filter to known Ukrainian oblasts
    ukr_oblasts = [
        "kyivska", "zakarpatska", "ivano-frankivska", "ternopilska",
        "rivnenska", "volynska", "zhytomyrska", "khmelnytska",
        "vinnytska", "chernivetska", "kyiv", "chernihivska",
        "sumska", "cherkaska", "poltavska", "kharkivska",
        "dnipropetrovska", "kirovohradska", "odeska",
        "mykolaiivska", "khersonska", "donetska", "zaporizka",
        "luhanska", "autonomous-republic-of-crimea"
    ]
    percent_df = percent_df.loc[percent_df.index.isin(ukr_oblasts)]
    if percent_df.empty:
        return

    # Melt wide→long for Plotly
    percent_df.index.name = "destination"
    df_long = percent_df.reset_index().melt(
        id_vars="destination",
        var_name="education",
        value_name="percentage"
    )
    pretty_name = "Education"

    # Create a new column that contains the percentage formatted as “xx.x%”
    df_long['pct_label'] = df_long['percentage'].apply(lambda v: f"{v:.1f}%")

    fig = px.bar(
        df_long,
        x="destination",
        y="percentage",
        color="education",
        barmode="stack",
        color_discrete_map=color_map,
        text="pct_label",
        title=f"Education Group Proportions by Destination (Stacked by {pretty_name})",
        labels={
            "percentage": "Proportion (%)",
            "destination": "Destination",
            "education": pretty_name
        }
    )
    fig.update_traces(textposition="inside")

    # Save the plot
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    html_out = os.path.join(plot_folder, f"stacked_bar_education_proportions.html")
    if save_fig:
        fig.write_html(html_out)
        print(f"[INFO] Saved HTML to {html_out}")

# A mapping from each age‐bin label to a specific color for the stacked bars
color_map = {
    "0-17":  "#1f77b4",
    "18-29": "#ff7f0e",
    "30-49": "#2ca02c",
    "50-64": "#d62728",
    "65+":   "#9467bd",
}

def plotStackedBarProportions(outdirs, disaggregator='age_binned', filters=None, save_fig=True, plot_folder="plots"):
    """
    Reads every migration.log in `outdirs`, applies optional filters,
    then groups by `destination` and the chosen `disaggregator` column (e.g. age_binned).
    For each run, it builds a destination×category table of NET arrivals (arrivals minus departures).
    Finally, it averages those tables across runs, normalizes each row to sum to 100%,
    and plots a 100%‐stacked bar chart of percentages by destination.
    """
    if filters is None or filters == ["None"]:
        filters = []
    # If disaggregator was passed as a list/tuple, grab its first element
    if isinstance(disaggregator, (list, tuple)) and disaggregator:
        disaggregator = disaggregator[0]

    per_run_tables = []

    for run_dir in outdirs:
        log_path = os.path.join(run_dir, "migration.log")
        if not os.path.exists(log_path):
            continue

        df = pd.read_csv(log_path)

        # Apply any user‐specified filters
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

        # --- BINNING FOR 'age' OR 'age_binned' ---
        # Treat "age_binned" exactly like "age": always create the age_binned column
        if disaggregator in ("age", "age_binned"):
            if "age" not in df.columns:
                return  # no raw 'age' column, so cannot bin
            bins   = [0, 17, 29, 49, 64, 200]
            labels = ["0-17", "18-29", "30-49", "50-64", "65+"]
            df['age_binned'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
            agg_col = "age_binned"
        else:
            agg_col = disaggregator

        # Abort if required columns are missing
        if agg_col not in df.columns or 'destination' not in df.columns:
            return

        # Count arrivals by (destination × category)
        arr = (
            df.assign(arrival=1)
              .groupby(["destination", agg_col])["arrival"]
              .sum()
              .reset_index()
        )
        # Count departures by (source × category), rename "source" → "destination"
        dep = (
            df.assign(departure=1)
              .groupby(["source", agg_col])["departure"]
              .sum()
              .reset_index()
              .rename(columns={"source": "destination"})
        )
        # Merge, fill NaNs with 0, compute net = arrival − departure
        net = pd.merge(arr, dep, on=["destination", agg_col], how="outer")

        # Only fill the numeric columns; do NOT call fillna(0) on the categorical column
        net["arrival"]   = net["arrival"].fillna(0)
        net["departure"] = net["departure"].fillna(0)
        net["net"]  = net["arrival"] - net["departure"]


        # Pivot: rows = destination, columns = category, values = net
        pivot = net.groupby(["destination", agg_col])["net"].sum().unstack(fill_value=0)
        per_run_tables.append(pivot)

    if not per_run_tables:
        return

    # Align all runs onto the same set of destinations & categories
    all_dest = sorted(set().union(*(tbl.index for tbl in per_run_tables)))
    all_cats = sorted(set().union(*(tbl.columns for tbl in per_run_tables)))

    R, D, C = len(per_run_tables), len(all_dest), len(all_cats)
    arr_3d = np.zeros((R, D, C), dtype=float)
    for i, tbl in enumerate(per_run_tables):
        tbl_full = tbl.reindex(index=all_dest, columns=all_cats, fill_value=0)
        arr_3d[i] = tbl_full.values

    # Compute mean across runs
    mean_df = pd.DataFrame(arr_3d.mean(axis=0), index=all_dest, columns=all_cats)
    # Normalize each row to sum to 100%
    percent_df = mean_df.div(mean_df.sum(axis=1), axis=0) * 100
    # Standardize index: lowercase, hyphens instead of underscores
    percent_df.index = (
        percent_df.index
                 .astype(str)
                 .str.strip()
                 .str.lower()
                 .str.replace("_", "-")
    )

    # Filter to known Ukrainian oblasts
    ukr_oblasts = [
        "kyivska", "zakarpatska", "ivano-frankivska", "ternopilska",
        "rivnenska", "volynska", "zhytomyrska", "khmelnytska",
        "vinnytska", "chernivetska", "kyiv", "chernihivska",
        "sumska", "cherkaska", "poltavska", "kharkivska",
        "dnipropetrovska", "kirovohradska", "odeska",
        "mykolaiivska", "khersonska", "donetska", "zaporizka",
        "luhanska", "autonomous-republic-of-crimea"
    ]
    percent_df = percent_df.loc[percent_df.index.isin(ukr_oblasts)]
    if percent_df.empty:
        return

    # Melt wide→long for Plotly
    percent_df.index.name = "destination"
    df_long = percent_df.reset_index().melt(
        id_vars="destination",
        var_name=agg_col,
        value_name="percentage"
    )
    pretty_name = agg_col.replace('_', ' ').title()

    # Create a new column that contains the percentage formatted as “xx.x%”
    df_long['pct_label'] = df_long['percentage'].apply(lambda v: f"{v:.1f}%")

    fig = px.bar(
        df_long,
        x="destination",
        y="percentage",
        color=agg_col,
        barmode="stack",
        color_discrete_map=color_map,
        text="pct_label",   # use our pre‐formatted strings
        title=f"Age Group Proportions by Destination (Stacked by {pretty_name})",
        labels={
            "percentage": "Proportion (%)",
            "destination": "Destination",
            agg_col: pretty_name
        }
    )

    # Force text labels inside each bar segment
    fig.update_traces(textposition="inside")

    
# Format the text to show one decimal and a “%” sign, and force it inside each segment

    fig.update_layout(
        xaxis_tickangle=-45,
        title_font_size=25,
        font=dict(size=18),
        legend=dict(font=dict(size=18)),
        legend_title_text=pretty_name,
        margin=dict(l=120, r=20, t=80, b=80),
        yaxis=dict(range=[0, 100], tickmode="linear", tick0=0, dtick=20, showgrid=True, gridcolor="LightGray")
    )

    # Ensure output folder exists and save HTML
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    html_out = os.path.join(plot_folder, f"stacked_bar_{agg_col}_proportions.html")
    if save_fig:
        fig.write_html(html_out)
        print(f"[INFO] Saved HTML to {html_out}")


        
def plotStackedBar(outdirs, disaggregator='age_binned', filters=None, save_fig=True, plot_folder="plots"):
    """
    Reads every migration.log in `outdirs`, applies optional filters,
    then groups by `destination` and the chosen `disaggregator` column (e.g. age bin, gender).
    For each run it builds a destination×category table of row-counts (one row = one individual).
    Finally it averages those tables across runs and plots.
    
    Parameters
    ----------
    disaggregator : str, optional
        The column used for stacking the bars (e.g., 'age' or 'gender').
        This is supplied at runtime via your config (FUMEheader); default is 'age'.
    filters : list of tuples or None
        Optional filters to apply, formatted as (column, operator, value),
        e.g. [('gender', '==', 'f'), ('age', '>=', 18)].
    save_fig : bool, optional
        Whether to save the figure as a PNG file; default is True.
    """
    # default empty filters list
    if filters is None:
        filters = []
    if isinstance(disaggregator, (list, tuple)) and disaggregator:
        disaggregator = disaggregator[0]
    
    per_run_tables = []  # will collect one pivot‐table DataFrame per run

    # 1. Process each run individually
    for run_dir in outdirs:
        log_path = os.path.join(run_dir, "migration.log")
        if not os.path.exists(log_path):
            print(f"[WARNING] No migration.log in {run_dir} – skipping.")
            continue

        df = pd.read_csv(log_path)  # read this run’s log

        # apply each user‐specified filter in turn
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

        # if disaggregator is 'age', create age bins
        if disaggregator == "age":
            if "age" not in df.columns:
                print("[ERROR] Cannot bin 'age' – no 'age' column.")
                return
            # define custom bins and labels
            bins   = [0, 17, 29, 49, 64, 200]
            labels = ["0-17", "18-29", "30-49", "50-64", "65+"]
            df['age_binned'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
            agg_col = 'age_binned'
        else:
            agg_col = disaggregator  # use the column

        # check if both columns present
        if agg_col not in df.columns or 'destination' not in df.columns:
            print(f"[ERROR] Missing '{agg_col}' or 'destination' in data – aborting.")
            return

        # --- NET-ARRIVALS MAGIC ---
        # 1) count arrivals by (destination × category)
        arr = (
            df.assign(arrival=1)
              .groupby(["destination", agg_col])["arrival"]
              .sum()
              .reset_index()
        )
        # 2) count departures by (source × category)
        dep = (
            df.assign(departure=1)
              .groupby(["source", agg_col])["departure"]
              .sum()
              .reset_index()
              .rename(columns={"source": "destination"})
        )
        # 3) merge, fill missing with 0, compute net = arr - dep
        net = pd.merge(
        arr, dep,
        on=["destination", agg_col],
        how="outer"
    )
        # only fill the two numeric columns
        net["arrival"]   = net["arrival"].fillna(0)
        net["departure"] = net["departure"].fillna(0)

        # now compute net arrivals
        net["net"] = net["arrival"] - net["departure"]

        # 4) pivot into destination×category net arrivals
        pivot = (
            net.groupby(["destination", agg_col])["net"]
               .sum()
               .unstack(fill_value=0)
        )
        per_run_tables.append(pivot)
        # --- end net-arrivals ---

    # if no valid runs give error
    if not per_run_tables:
        print("[INFO] No valid data found in any run.")
        return

    # 2. Align all runs onto the same destinations & categories
    all_destinations = sorted(set().union(*(t.index for t in per_run_tables)))
    all_categories   = sorted(set().union(*(t.columns for t in per_run_tables)))

    # 3. Stack into a 3D array (runs × destinations × categories)
    R = len(per_run_tables)
    D = len(all_destinations)
    C = len(all_categories)
    arr = np.zeros((R, D, C), dtype=float)

    for i, tbl in enumerate(per_run_tables):
        # re-index to full grid, fill missing with 0
        tbl_full = tbl.reindex(index=all_destinations,
                               columns=all_categories,
                               fill_value=0)
        arr[i, :, :] = tbl_full.values

    # 4. Compute the mean across runs
    mean_matrix = arr.mean(axis=0)
    mean_df = pd.DataFrame(mean_matrix, index=all_destinations, columns=all_categories)
    
    #NORMALISE
    mean_df.index = (
        mean_df.index.astype(str).str.strip().str.lower().str.replace("_", "-"))
    
    # keep only the 25 known Ukrainian oblasts
    ukr_oblasts = [
        "kyivska", "zakarpatska", "ivano-frankivska", "ternopilska",
        "rivnenska", "volynska", "zhytomyrska", "khmelnytska",
        "vinnytska", "chernivetska", "kyiv", "chernihivska",
        "sumska", "cherkaska", "poltavska", "kharkivska",
        "dnipropetrovska", "kirovohradska", "odeska",
        "mykolaiivska", "khersonska", "donetska", "zaporizka",
        "luhanska", "autonomous-republic-of-crimea"
    ]


    mask = mean_df.index.isin(ukr_oblasts)
    # debug print so you can see what you actually have:
    print("[DEBUG] all destinations:", list(mean_df.index))
    print(f"[DEBUG] keeping {mask.sum()} oblasts:", list(mean_df.index[mask]))

    mean_df = mean_df.loc[mask]

    #if nothing left, bail out cleanly
    if mean_df.empty:
        print("[INFO] No Ukrainian-oblast destinations to plot after filtering — skipping.")
        return
        
    plot_df = mean_df.copy()
    if disaggregator == "education":
        plot_df.rename(columns=EDU_LABELS, inplace=True)
        
    elif disaggregator == "property_in_ukraine":
        plot_df.rename(columns=PROP_LABELS, inplace=True)
        
    '''
    # 5. Static PNG via Matplotlib
    plt.figure(figsize=(12, 7))
    plot_df.plot(kind='bar', stacked=True, ax=plt.gca())
    pretty_name = pretty(disaggregator)
    plt.title(f"Mean Arrivals by Destination, Stacked by {pretty_name}")
    plt.xlabel("Destination (Ukrainian Oblast)")
    plt.ylabel(f"Mean No. of Returnees (per run) — {pretty_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # ensure output folder exists
    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    png_out = os.path.join(plot_folder, f"stacked_bar_{disaggregator}.png")
  #  if save_fig:
   #     plt.savefig(png_out, dpi=150)
    #    print(f"[INFO] Saved PNG to {png_out}")
  #  plt.close()
    '''
    pretty_name = pretty(disaggregator)
    
    # 6. Interactive HTML via Plotly
    plot_df.index.name = "destination"
    df_long = plot_df.reset_index().melt(
        id_vars="destination",
        var_name=disaggregator,
        value_name="mean_count"
    )
    
    color_map = None
    if disaggregator=="gender":
        df_long['gender'] = df_long['gender'].map(GENDER_NAMES)
        color_map = {GENDER_NAMES[k]: GENDER_COLORS[k] for k in GENDER_COLORS}

    fig = px.bar(
        df_long,
        x="destination",
        y="mean_count",
        color=disaggregator,
        barmode="stack",
        color_discrete_map=color_map,
        title=f"Net Arrivals by Destination, Stacked by {pretty_name}",
        labels={"mean_count": "Net No. of Returnees (x100)",
                "destination": "Destination", disaggregator: pretty_name}
    )
    fig.update_layout(xaxis_tickangle=-45, yaxis=dict(tickmode='linear', tick0=0, dtick=100), title_font_size=25, font=dict(size=18), legend=dict(font=dict(size=18)), legend_title_text=pretty(disaggregator), xaxis_title_font_size=24, yaxis_title_font_size=24, xaxis_tickfont_size=20, yaxis_tickfont_size=20, margin=dict(l=120, r=20, t=80, b=80))
    
    # Choose dtick based on max mean_count value
    max_val = df_long['mean_count'].max()
    dtick_val = 100 if max_val > 100 else 25

    fig.update_yaxes(
    tickmode="linear",
    tick0=0,
    dtick=dtick_val,
    showgrid=True,
    gridcolor="LightGray",
    gridwidth=1,
    )

        
    html_out = os.path.join(plot_folder, f"stacked_bar_{disaggregator}.html")
    fig.write_html(html_out)
    print(f"[INFO] Saved HTML to {html_out}")


def plotLineOverTime(outdirs, primary_filter_column='source', primary_filter_value=None, line_disaggregator='gender', filters=None, save_fig=True, plot_folder="plots", show_quartiles=True):
    """
    Fan plot of total migrations over time (median ± IQR across ensemble runs).
    """
    if filters is None:
        filters = []
    if isinstance(line_disaggregator, (list,tuple)) and line_disaggregator:
        line_disaggregator = line_disaggregator[0]
    if isinstance(primary_filter_column, (list, tuple)) and primary_filter_column:
        primary_filter_column = primary_filter_column[0]
    if isinstance(primary_filter_value, (list, tuple)) and primary_filter_value:
        primary_filter_value = primary_filter_value[0]
        
    # 1. Read each run separately
    per_run = []
    for d in outdirs:
        fp = os.path.join(d, "migration.log")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)

        # optional primary filter (e.g. source="germany")
        if primary_filter_value is not None:
            df = df[df[primary_filter_column] == primary_filter_value]
            
            # if user asked to disaggregate by "education" but there's no such column, skip this run entirely
        if line_disaggregator == "education" and "education" not in df.columns:
            print(f"[WARNING] No 'education' column in {d}/migration.log – skipping run for education plot.")
            continue

        # optional extra filters
        for col, op, val in filters:
            if op=="==":  df = df[df[col]==val]
            elif op==">": df = df[df[col]> val]
            # ... (add your other ops)

        # if we want age‐bins instead of raw ages:
        if line_disaggregator=='age_binned':
            bins  = [0,17,29,49,64,200]
            labels= ["0-17","18-29","30-49","50-64","65+"]
            df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
            agg = 'age_bin'
        else:
            agg = line_disaggregator

        # now group by time × category
        grp = df.groupby(['time', agg]).size().unstack(fill_value=0)
        per_run.append(grp)
    if not per_run:
        print("No data found.")
        return

    # 2. build common time‐axis and category list
    all_times = sorted(set().union(*(df.index for df in per_run)))
    all_cats  = sorted(set().union(*(df.columns for df in per_run)))

    # 3. build a runs × times × cats array
    R = len(per_run)
    T = len(all_times)
    C = len(all_cats)
    arr = np.zeros((R, T, C), dtype=float)

    for i, df in enumerate(per_run):
        # reindex to full grid
        df2 = df.reindex(index=all_times, columns=all_cats, fill_value=0)
        arr[i,:,:] = df2.values

    # 4. compute median & IQR across runs (axis=0 → (T×C) arrays)
    med = np.median(arr, axis=0)
    q25 = np.percentile(arr, 25, axis=0)
    q75 = np.percentile(arr, 75, axis=0)
    
    # prepare display‐names if education or property
    if line_disaggregator == "education":
        name_map = EDU_LABELS
    elif line_disaggregator == "property_in_ukraine":
        name_map = PROP_LABELS
    elif line_disaggregator == "gender":
        palette = GENDER_COLORS
        name_map = GENDER_NAMES
    else:
        name_map = {}
    
    pretty_name = pretty(line_disaggregator)
    '''
    # Static Matplotlib fan plot
    plt.figure(figsize=(12,6))
    for j, cat in enumerate(all_cats):
        disp = name_map.get(cat, cat)
        # shade between q25 and q75 for this category
        if show_quartiles:
            plt.fill_between(all_times,
                             q25[:,j],
                             q75[:,j],
                             alpha=0.2,
                             label=f"{disp} IQR")
        # median line
        plt.plot(all_times,
                 med[:,j],
                 marker='o',
                 linewidth=2,
                 label=f"{disp} median")

    plt.title(f"Returns Over Time\n"
              f"(filtered {primary_filter_column}="
              f"{primary_filter_value or 'All'})")
    pretty_name = pretty(line_disaggregator)
    plt.xlabel("Time (months)")
    plt.ylabel("Number of Returnees (x100)")
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()

    Path(plot_folder).mkdir(exist_ok=True, parents=True)
    png = os.path.join(plot_folder,
                       f"fan_{primary_filter_column}_{primary_filter_value}_{line_disaggregator}.png")
  #  if save_fig:
  #      plt.savefig(png, dpi=150)
  #      print(f"[INFO] saved PNG {png}")
    #plt.show()
  #  plt.close()
  '''
    
    # 6. Interactive Plotly fan plot
    # melt into long form: columns = time, category, median, q25, q75
    rows = []
    for ti, t in enumerate(all_times):
        for j, cat in enumerate(all_cats):
            # for gender, cat is 'f' or 'm'; map it now to full name
            disp = name_map.get(cat, cat)
            if line_disaggregator=='gender':
                disp = GENDER_NAMES.get(cat, cat)
            rows.append({
                'time': all_times[ti],  # FIXED: use the actual time value
                pretty(line_disaggregator): disp,
                'median': med[ti,j],
                'q25':    q25[ti,j],
                'q75':    q75[ti,j]
            })
    dfp = pd.DataFrame(rows)
    
    # Define the real‐date start:
    start = pd.to_datetime("2025-03-01")     

    # Convert the integer “month offsets” into real datetimes:
    dfp['month'] = dfp['time'].astype(int).apply(
    lambda m: start + relativedelta(months=(m))
    )


    # pick up which column we’re coloring by
    color_col = pretty(line_disaggregator)
    color_map = None
    if line_disaggregator=='gender':
        # map the human labels to your exact hexes
        color_map = {
            GENDER_NAMES['f']: GENDER_COLORS['f'],
            GENDER_NAMES['m']: GENDER_COLORS['m']
        }

    fig = px.line(
        dfp,
        x='month',
        y='median',
        color=color_col,
        color_discrete_map=color_map,
        title=(f"Returnees Over Time<br>"
               f"(filtered {primary_filter_column}="
               f"{primary_filter_value or 'All'})"),
        labels={'median':'Net No. of Returnees (×100)','time':'Time (Months)'}
    )

    # add the shading for each category
    if show_quartiles:
        for disp in dfp[color_col].unique():
            dsub = dfp[dfp[color_col] == disp]
            fig.add_traces(
                px.scatter(dsub, x='month', y='q25')
                .update_traces(line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,0,0)', name=f"{disp} 25th pct").data)
            fig.add_traces(
                px.scatter(dsub, x='month', y='q75')
                .update_traces(line=dict(width=0), fill='tonexty', fillcolor='rgba(0,0,0,0.1)', name=f"{disp} 75th pct").data)
    
    # Format the axis tick
        fig.update_xaxes(
        dtick="M1",            # tick every month
        tickformat="%b %Y",     # abbreviated month + year
        tickangle=45
        )

    # remove/gray‐out all of the point markers on the median traces
    for trace in fig.data:
        # keep the original key ("f" or "m") in legendgroup
        key = trace.name
        trace.legendgroup = key
        # rename only the label that shows in the legend
        if key in GENDER_NAMES:
            trace.name = GENDER_NAMES[key]
        # draw only lines (no markers)
        trace.update(mode="lines")        
    
    fig.update_layout(title_font_size=25, font=dict(size=18), legend=dict(font=dict(size=18)), yaxis=dict(tickmode='linear', dtick=200, range = [0, dfp['median'].max() * 1.1], autorange=False), legend_title_text=pretty(line_disaggregator), xaxis_title_font_size=24, yaxis_title_font_size=24, xaxis_tickfont_size=20, yaxis_tickfont_size=20)

    html = os.path.join(plot_folder,
                        f"fan_{primary_filter_column}_{primary_filter_value}_{line_disaggregator}.html")
    fig.write_html(html)
    print(f"[INFO] saved HTML {html}")
    #fig.show()



def plotNamedSingleByTimestep(code, outdirs, plot_type, FUMEheader, filters=[], disaggregator=None, primary_filter_column=None, primary_filter_value=None, plot_path='../..'):
   
    print(f"[FUMEplot]: plot_type set to {plot_type}.", file=sys.stderr)

    if isinstance(disaggregator, (list, tuple)) and disaggregator:
        disaggregator = disaggregator[0]
    if disaggregator is None:
        disaggregator = getattr(FUMEheader, 'disaggregator', 'gender')
        if isinstance(disaggregator, (list, tuple)) and disaggregator:
            disaggregator = disaggregator[0]
    
    # e.g. primary_filter_column = "source" or "destination"
    if primary_filter_column is None:
        primary_filter_column = getattr(FUMEheader, 'primary_filter_column', 'source')
    if isinstance(primary_filter_column, (list, tuple)) and primary_filter_column:
        primary_filter_column = primary_filter_column[0]
    
    # e.g. primary_filter_value = "germany" or "ukr_kyivska" 
    if primary_filter_value is None:
        primary_filter_value = getattr(FUMEheader, 'primary_filter_value', None)
    if isinstance(primary_filter_value, (list, tuple)) and primary_filter_value:
        primary_filter_value = primary_filter_value[0]
       
    filters = getattr(FUMEheader, 'filters', [])
    
    saving=True
    plotfolder=plot_path+'/EnsemblePlots/'+code+'Plots'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)
    
    fi=0
    
     # ----- SPECIAL CASE: default run outputs 3 plots in one go -----
    if plot_type == "default":
        # 1) Sankey
        '''
        plotMigrationSankey(outdirs,
                            save_fig=saving,
                            plot_folder=plotfolder)
        '''
        # 2) Stacked bar by gender
        plotStackedBar(outdirs=outdirs,
                        disaggregator="gender",
                        filters=filters,
                        save_fig=saving,
                        plot_folder=plotfolder)
        # 3) Stacked bar by age‐bins
        plotStackedBar(outdirs=outdirs,
                        disaggregator="age",
                        filters=filters,
                        save_fig=saving,
                        plot_folder=plotfolder)
        
        # 3.1) 100%-stacked (proportional) bar by age_binned
        plotStackedBarProportions(outdirs=outdirs,
                                    disaggregator="age_binned",
                                    filters=filters,
                                    save_fig=saving,
                                    plot_folder=plotfolder)
                        
       # 3.2) 100% stacked bar proportions for education level
        plotStackedBarProportions_Education(outdirs=outdirs,
                                                filters=filters,
                                                save_fig=saving,
                                                plot_folder=plotfolder)
         
       # plot gender proportions
        plotStackedBarProportions_Gender(outdirs=outdirs,
                                            filters=filters,
                                            save_fig=saving,
                                            plot_folder=plotfolder)
 
        # plot property proportions
        plotStackedBarProportions_Property(outdirs=outdirs,
                                                filters=filters,
                                                save_fig=saving,
                                                plot_folder=plotfolder)
        
                        
        plotStackedBar(outdirs=outdirs,
                        disaggregator="education",
                        filters=filters,
                        save_fig=saving,
                        plot_folder=plotfolder)
                        
        plotStackedBar(outdirs=outdirs,
                        disaggregator="property_in_ukraine",
                        filters=filters,
                        save_fig=saving,
                        plot_folder=plotfolder)
                        
        plotLineOverTime(outdirs=outdirs, primary_filter_column=primary_filter_column, primary_filter_value=primary_filter_value, line_disaggregator="age_binned", filters=filters, save_fig=saving, plot_folder=plotfolder
        )
        
        plotLineOverTime(outdirs=outdirs, primary_filter_column=primary_filter_column, primary_filter_value=primary_filter_value, line_disaggregator="property_in_ukraine", filters=filters, save_fig=saving, plot_folder=plotfolder
        )
        
        plotLineOverTime(outdirs=outdirs, primary_filter_column=primary_filter_column, primary_filter_value=primary_filter_value, line_disaggregator="education", filters=filters, save_fig=saving, plot_folder=plotfolder
        )
        
        plotLineOverTime(outdirs=outdirs, primary_filter_column=primary_filter_column, primary_filter_value=primary_filter_value, line_disaggregator="gender", filters=filters, save_fig=saving, plot_folder=plotfolder
        )
        return
    # ---------------------------------------------------------------
    
    if plot_type == "source_hist" or plot_type == "all":
        plotSourceHist(outdirs, filters=[], save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=True)
    
    if plot_type == "single_sankey" or plot_type == "all":
        plotMigrationSankey(outdirs, save_fig=saving, plot_folder=plotfolder)
        
    if plot_type == "stacked_bar" or plot_type == "all":
        plotStackedBar(outdirs=outdirs, disaggregator=disaggregator, filters=filters, save_fig=saving, plot_folder=plotfolder
        )
    if plot_type == "line_chart" or plot_type == "all":
        plotLineOverTime(outdirs=outdirs, primary_filter_column=primary_filter_column, primary_filter_value=primary_filter_value, line_disaggregator=disaggregator, filters=filters, save_fig=saving, plot_folder=plotfolder
        )

    # Show plot
    plt.show()


if __name__ == "__main__":

    import ReadHeaders
    code = "homecoming" 
    
    plot_type = "all" # Plot type is set to 'all' by default
    if len(sys.argv) > 1: # If at least one argument is passed, the first argument becomes 'code'
        code = sys.argv[1]
        if len(sys.argv) > 2: # Second argument becomes 'plot_type'
            plot_type = sys.argv[2]

    outdir = f"../sample_{code}_agentlog"

    outdirs = ReadHeaders.GetOutDirs(outdir)
    
    '''
    #FUMEheader = ReadHeaders.ReadOutHeaders(outdirs, mode=code) #FOR CSV
    FUMEheader = ReadHeaders.ReadMovelogHeaders(outdirs, mode=code)
    headers = ReadHeaders.ReadMovelogHeaders(outdirs, mode=code)
    
    plotNamedSingleByTimestep(code, outdirs, plot_type, FUMEheaders)
    '''
    # CONFIG HEADER (reads your .yml disaggregator / filters / modes)
    FUMEheader = ReadHeaders.ReadOutHeaders(outdirs, mode=code)
    # if you still need move‑log metadata later, you can load it too:
    move_header = ReadHeaders.ReadMovelogHeaders(outdirs, mode=code)

    plotNamedSingleByTimestep(code, outdirs, plot_type, FUMEheader)


# ISSUES:
# - Flee and Facs require _agentlog folders to be present

# error bar: add max-min interval, 25-75 percentiles, std(?)
