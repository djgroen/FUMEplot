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
lpl.size.set(*latex_doc_size)


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
    # fig, ax = plt.subplots(num=plot_num+1, figsize=(10,6))
    # - LatexPlotLib version
    fig, ax = lpl.subplots(num=plot_num+1)

    # plt.bar(mean_counts.index, 
    #         mean_counts.values, 
    #         yerr=std_counts.values, 
    #         color='skyblue', 
    #         capsize=5,
    #         )

    ax.boxplot(
                combined_counts,
                tick_labels=_formatLabels(mean_counts.index),
                patch_artist=True, 
                #notch=True,
                bootstrap=1000,
                showmeans=True, 
                meanline=True, 
                showbox=True,
                showcaps=True,
                showfliers=True,
                medianprops=dict(color='blue'),
                meanprops=dict(color='black'),
                boxprops=dict(facecolor='skyblue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue',),
                capwidths=0.3,
                flierprops=dict(marker='o', markerfacecolor='blue', markersize=2, linestyle='none'),
                label='Entries',
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

    # save plot
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
    Creates a single Sankey diagram by aggregating migration.log data
    from all ensemble run directories (outdirs). In each log file, it reads:
        - 'source': the country of origin (e.g., Germany, Poland, etc.)
        - 'destination': the Ukrainian region (e.g., ukr_kyivska, ukr_odessa, etc.)
        - 'sizeF': number of migrants in that record
    It then groups the data (summing 'sizeF') per (source, destination) pair
    and builds one combined Sankey diagram.
    
    Parameters
    ----------
    outdirs : list
        List of run directories (each containing a migration.log file).
    save_fig : bool, optional
        Whether to save the resulting figure as an HTML file. Default is True.
    plot_folder : str, optional
        Folder in which to save the output file. Default is "plots".
    """

    all_data = []
    # Loop over every run directory and try to read migration.log from it
    for d in outdirs:
        migration_log_path = os.path.join(d, "migration.log")
        if not os.path.exists(migration_log_path):
            print(f"[WARNING] Migration log not found in {d}, skipping this folder.")
            continue
        try:
            df_run = pd.read_csv(migration_log_path)
            all_data.append(df_run)
        except Exception as e:
            print(f"[ERROR] Could not read {migration_log_path}: {e}")
            continue

    if not all_data:
        print("[INFO] No migration data found in any run directory.")
        return

    # Concatenate data from all runs into one DataFrame.
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Group the data by 'source' and 'destination' and sum up the 'sizeF' counts.
    flows = df_all.groupby(['source', 'destination'])['sizeE'].sum().reset_index()

    # Create a sorted list of unique source countries and destination regions.
    origin_list = sorted(list(flows['source'].unique()))
    region_list = sorted(list(flows['destination'].unique()))
    
    # Create the complete node label list: sources on left, destinations on right.
    node_labels = origin_list + region_list
    label_to_index = {label: idx for idx, label in enumerate(node_labels)}

    # Build the link data arrays for the Sankey (source indices, target indices, and values).
    source_indices = []
    target_indices = []
    values = []
    for _, row in flows.iterrows():
        src_label = row['source']
        dest_label = row['destination']
        count = row['sizeE']
        src_idx = label_to_index.get(src_label)
        tgt_idx = label_to_index.get(dest_label)
        # Only add the link if both indices are found.
        if src_idx is None or tgt_idx is None:
            continue
        source_indices.append(src_idx)
        target_indices.append(tgt_idx)
        values.append(count)
    
    # Create manual node positions: put sources (origin_list) on the left (x=0.1) 
    # and destinations (region_list) on the right (x=0.9).
    n_origins = len(origin_list)
    n_regions = len(region_list)
    node_x = []
    node_y = []
    # For each origin, place it at x=0.1, with evenly spaced y-values.
    for i in range(n_origins):
        node_x.append(0.1)
        node_y.append((i + 1) / (n_origins + 1))
    # For each destination, place it at x=0.9, with evenly spaced y-values.
    for i in range(n_regions):
        node_x.append(0.9)
        node_y.append((i + 1) / (n_regions + 1))

    # Create the Sankey diagram using Plotly.
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values
        )
    )])
    
    fig.update_layout(
        title_text="Combined Migration Sankey Diagram (All Ensemble Runs)",
        font_size=10
    )
    
    # Save the figure if requested.
    os.makedirs(plot_folder, exist_ok=True)
    out_path = os.path.join(plot_folder, "migration_sankey_combined.html")
    fig.write_html(out_path)
    print(f"[INFO] Combined Sankey diagram saved to {out_path}")
    
    png_out_path = os.path.join(plot_folder, "migration_sankey_combined.png")
    try:
        fig.write_image(png_out_path)
        print(f"[INFO] Combined Sankey diagram image saved to {png_out_path}")
    except Exception as e:
        print(f"[ERROR] Unable to save PNG image: {e}")
        
def plotStackedBar(outdirs, aggregator='age_binned', filters=None, save_fig=True, plot_folder="plots"):
    """
    Creates a stacked bar chart showing total arrivals by destination 
    (Ukrainian Oblast) with each bar stacked by the categories 
    found in the 'aggregator' column (e.g., age, gender, etc.).
    
    This function reads migration.log files from each run directory in outdirs,
    computes a 'total_size' as the sum of SizeE (employable) and SizeD (dependant),
    applies optional filters, groups the data by destination and aggregator, 
    and then produces a stacked bar chart.
    
    Parameters
    ----------
    outdirs : list of str
        List of run directories (each containing a migration.log file).
    aggregator : str, optional
        The column used for stacking the bars (e.g., 'age' or 'gender').
        This is supplied at runtime via your config (FUMEheader); default is 'age'.
    filters : list of tuples or None
        Optional filters to apply, formatted as (column, operator, value),
        e.g. [('gender', '==', 'f'), ('age', '>=', 18)].
    save_fig : bool, optional
        Whether to save the figure as a PNG file; default is True.
    plot_folder : str, optional
        The folder in which to save the resulting figure; default is "plots".
    """
    if filters is None:
        filters = []
    
    per_run = []
    # 1. build one pivot table per run
    for d in outdirs:
        migration_file = os.path.join(d, "migration.log")
        if not os.path.exists(migration_file):
            print(f"[WARNING] No migration.log in {d}, skipping.")
            continue

        df = pd.read_csv(migration_file)

        # apply any filters
        for col, op, val in filters:
            if op == "==":   df = df[df[col] == val]
            elif op == "!=": df = df[df[col] != val]
            elif op == ">":  df = df[df[col] >  val]
            elif op == "<":  df = df[df[col] <  val]
            elif op == ">=": df = df[df[col] >= val]
            elif op == "<=": df = df[df[col] <= val]
            else:
                print(f"[WARNING] unsupported filter {col}{op}{val}")

        # age‑bin special case
        if aggregator == "age":
            if "age" not in df.columns:
                print("[ERROR] need raw 'age' to bin")
                return
            bins  = [0,17,29,49,64,200]
            labels= ["0-17","18-29","30-49","50-64","65+"]
            df['age_binned'] = pd.cut(df['age'],
                                      bins=bins,
                                      labels=labels,
                                      right=True)
            agg = 'age_binned'
        else:
            agg = aggregator

        # sanity check
        if agg not in df.columns or 'destination' not in df.columns:
            print(f"[ERROR] missing '{agg}' or 'destination' in data")
            return

        # group & unstack → destination × agg
        pt = df.groupby(['destination', agg]).size()\
               .unstack(fill_value=0)

        per_run.append(pt)

    if not per_run:
        print("[INFO] no data found")
        return

    # 2. align onto common row/col index
    all_dests = sorted(set.union(*[set(pt.index) for pt in per_run]))
    all_cats  = sorted(set.union(*[set(pt.columns) for pt in per_run]))

    # 3) stack into array runs×D×C
    R = len(per_run)
    D = len(all_dests)
    C = len(all_cats)
    arr = np.zeros((R,D,C), float)
    for i, pt in enumerate(per_run):
        pt2 = pt.reindex(index=all_dests, columns=all_cats, fill_value=0)
        arr[i,:,:] = pt2.values

    # 4. mean across runs
    mean_vals = arr.mean(axis=0)
    mean_df   = pd.DataFrame(mean_vals,
                             index=all_dests,
                             columns=all_cats)

    # Matplotlib PNG
    plt.figure(figsize=(12,7))
    mean_df.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title(f"Mean Arrivals by Destination, Stacked by '{aggregator}'")
    plt.xlabel("Destination (Ukrainian Oblast)")
    plt.ylabel("Mean No. of Individuals (across ensemble)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    png_path = os.path.join(plot_folder, f"stacked_bar_{aggregator}.png")
    if save_fig:
        plt.savefig(png_path, dpi=150)
        print(f"[INFO] saved PNG {png_path}")
    plt.show()
    plt.close()

    # Plotly HTML
    mean_df.index.name = "destination"
    
    df_melt = mean_df.reset_index().melt(id_vars="destination", var_name=aggregator, value_name="mean_count")

    fig = px.bar(df_melt,
                 x="destination",
                 y="mean_count",
                 color=aggregator,
                 barmode="stack",
                 title=f"Mean Arrivals by Destination, Stacked by '{aggregator}'",
                 labels={"mean_count":"Mean No. of Individuals",
                         "destination":"Destination"})
    fig.update_layout(xaxis_tickangle=-45)

    html_path = os.path.join(plot_folder, f"stacked_bar_{aggregator}.html")
    fig.write_html(html_path)
    print(f"[INFO] saved HTML {html_path}")
    fig.show()


def plotLineOverTime(outdirs, primary_filter_column='source', primary_filter_value=None, line_aggregator='gender', filters=None, save_fig=True, plot_folder="plots", show_quartiles=False):
    """
    Fan plot of total migrations over time (median ± IQR across ensemble runs).
    """
    if filters is None:
        filters = []
        
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

        # optional extra filters
        for col, op, val in filters:
            if op=="==":  df = df[df[col]==val]
            elif op==">": df = df[df[col]> val]
            # ... (add your other ops)

        # if we want age‐bins instead of raw ages:
        if line_aggregator=='age_binned':
            bins  = [0,17,29,49,64,200]
            labels= ["0-17","18-29","30-49","50-64","65+"]
            df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
            agg = 'age_bin'
        else:
            agg = line_aggregator

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

    # Static Matplotlib fan plot
    plt.figure(figsize=(12,6))
    for j, cat in enumerate(all_cats):
        # shade between q25 and q75 for this category
        if show_quartiles:
            plt.fill_between(all_times,
                             q25[:,j],
                             q75[:,j],
                             alpha=0.2,
                             label=f"{cat} IQR")
        # median line
        plt.plot(all_times,
                 med[:,j],
                 marker='o',
                 linewidth=2,
                 label=f"{cat} median")

    plt.title(f"Migrations Over Time\n"
              f"(filtered {primary_filter_column}="
              f"{primary_filter_value or 'All'})")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of Migrations")
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()

    Path(plot_folder).mkdir(exist_ok=True, parents=True)
    png = os.path.join(plot_folder,
                       f"fan_{primary_filter_column}_{primary_filter_value}_{line_aggregator}.png")
    if save_fig:
        plt.savefig(png, dpi=150)
        print(f"[INFO] saved PNG {png}")
    plt.show()
    plt.close()

    # Interactive Plotly fan plot
    # melt into long form: columns = time, category, median, q25, q75
    rows = []
    for ti, t in enumerate(all_times):
        for j, cat in enumerate(all_cats):
            rows.append({
                'time': t,
                line_aggregator: cat,
                'median': med[ti,j],
                'q25':    q25[ti,j],
                'q75':    q75[ti,j]
            })
    dfp = pd.DataFrame(rows)

    fig = px.line(dfp,
                  x='time',
                  y='median',
                  color=line_aggregator,
                  title=("Migrations Over Time<br>"
                         f"(filtered {primary_filter_column}="
                         f"{primary_filter_value or 'All'})"),
                  labels={'median':'Median Migrations','time':'Time'})
    # add the shading for each category
    if show_quartiles:
        for cat in all_cats:
            dsub = dfp[dfp[line_aggregator]==cat]
            fig.add_traces(px.scatter(dsub, x='time', y='q25').update_traces(
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,0,0,0)',  # invisible, but needed to anchor
                name=f"{cat} 25th pct"
            ).data)
            fig.add_traces(px.scatter(dsub, x='time', y='q75').update_traces(
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,0,0,0.1)',  # light grey shading
                name=f"{cat} 75th pct"
            ).data)

    fig.update_layout(legend=dict(itemsizing='constant'),
                      xaxis_type='linear')
    html = os.path.join(plot_folder,
                        f"fan_{primary_filter_column}_{primary_filter_value}_{line_aggregator}.html")
    fig.write_html(html)
    print(f"[INFO] saved HTML {html}")
    fig.show()


def plotNamedSingleByTimestep(code, outdirs, plot_type, FUMEheader, filters=[], aggregator=None):
    aggregator = getattr(FUMEheader, 'aggregator', 'gender')
    filters = getattr(FUMEheader, 'filters', [])
    
    # e.g. primary_filter_column = "source" or "destination"
    primary_filter_column = getattr(FUMEheader, 'primary_filter_column', 'source')
    # e.g. primary_filter_value = "germany" or "ukr_kyivska" 
    primary_filter_value = getattr(FUMEheader, 'primary_filter_value', 'poland')
    
    # ensembleSize = 0
    ensembleSize = 8
    
    saving=True
    plotfolder='../../EnsemblePlots/'+code+'Plots'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)
    
    fi=0
    if plot_type == "source_hist" or plot_type == "all":
        plotSourceHist(outdirs, filters=[], save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=True)
    
    if plot_type == "single_sankey" or plot_type == "all":
        plotMigrationSankey(outdirs, save_fig=saving, plot_folder=plotfolder)
        
    if plot_type == "stacked_bar" or plot_type == "all":
        plotStackedBar(outdirs=outdirs, aggregator=aggregator, filters=filters, save_fig=saving, plot_folder=plotfolder
        )
    if plot_type == "line_chart" or plot_type == "all":
        plotLineOverTime(outdirs=outdirs, primary_filter_column=primary_filter_column, primary_filter_value=primary_filter_value, line_aggregator=aggregator, filters=filters, save_fig=saving, plot_folder=plotfolder
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
    # CONFIG HEADER (reads your .yml aggregator / filters / modes)
    FUMEheader = ReadHeaders.ReadOutHeaders(outdirs, mode=code)
    # if you still need move‑log metadata later, you can load it too:
    move_header = ReadHeaders.ReadMovelogHeaders(outdirs, mode=code)

    plotNamedSingleByTimestep(code, outdirs, plot_type, FUMEheader)


# ISSUES:
# - Flee and Facs require _agentlog folders to be present

# error bar: add max-min interval, 25-75 percentiles, std(?)
