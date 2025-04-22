import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
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
    
    # 1. Read all migration.log files from each run directory in outdirs.
    all_data = []
    for d in outdirs:
        migration_file = os.path.join(d, "migration.log")
        if not os.path.exists(migration_file):
            print(f"[WARNING] No migration.log in {d}, skipping.")
            continue
        df_run = pd.read_csv(migration_file)
        all_data.append(df_run)
            
    if not all_data:
        print("[INFO] No migration data found in any run directory.")
        return
        
    # 2. Concatenate data from all runs into one DataFrame.
    df_all = pd.concat(all_data, ignore_index=True)
    
    # 3. Apply optional filters.
    for (col, op, val) in filters:
        if op == "==":
            df_all = df_all[df_all[col] == val]
        elif op == "!=":
            df_all = df_all[df_all[col] != val]
        elif op == ">":
            df_all = df_all[df_all[col] > val]
        elif op == "<":
            df_all = df_all[df_all[col] < val]
        elif op == ">=":
            df_all = df_all[df_all[col] >= val]
        elif op == "<=":
            df_all = df_all[df_all[col] <= val]
        else:
            print(f"[WARNING] Unsupported operator '{op}' for filter {col} {op} {val}")
        
    if aggregator == "age":
        # Define your bins and labels as desired.
        bins = [0, 17, 29, 49, 64, 200]  # adjust upper bound as needed
        labels = ["0-17", "18-29", "30-49", "50-64", "65+"]
        if "age" not in df_all.columns:
            print("[ERROR] Column 'age' not found in data. Cannot bin ages.")
            return
        # Create the age_bin column using pd.cut()
        df_all['age_bin'] = pd.cut(df_all['age'], bins=bins, labels=labels, right=True)
        # Update aggregator to use the new column
        aggregator = "age_bin"
            
    # 4. Check necessary columns: aggregator, 'destination', and total_size.
    for needed in [aggregator, 'destination']:
        if needed not in df_all.columns:
            print(f"[ERROR] Column '{needed}' not found in data.")
            return
    
    # 5. Group the data by destination and aggregator column, summing the total_size counts.
    group_df = df_all.groupby(['destination', aggregator]).size().reset_index(name='counts')

    
    # 6. Pivot the grouped DataFrame so that rows are destinations and columns are the values of the aggregator.
    pivot_df = group_df.pivot(index='destination', columns=aggregator, values='counts').fillna(0)

    
    # 7. Create a stacked bar chart.
    plt.figure(figsize=(12, 7))
    pivot_df.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title(f"Arrivals by Destination, Stacked by '{aggregator}'")
    plt.xlabel("Destination (Ukrainian Oblast)")
    plt.ylabel("Number of Individuals (Row Count)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


    # 8. Save the figure if required.
    os.makedirs(plot_folder, exist_ok=True)
    out_file = os.path.join(plot_folder, f"stacked_bar_{aggregator}.png")
    if save_fig:
        plt.savefig(out_file, dpi=150)
        print(f"[INFO] Stacked bar chart saved to {out_file}")
    
    plt.show()
    plt.close()
    
    # To prepare data for Plotly, melt the pivoted DataFrame into long format.
    df_melted = pivot_df.reset_index().melt(id_vars="destination", 
                                            var_name=aggregator, 
                                            value_name="counts")
    # Create a Plotly Express stacked bar chart.
    fig = px.bar(df_melted,
                 x="destination",
                 y="counts",
                 color=aggregator,
                 barmode="stack",
                 title=f"Arrivals by Destination, Stacked by '{aggregator}'",
                 labels={"destination": "Destination (Ukrainian Oblast)",
                         "counts": "Number of Individuals (Row Count)"})
    fig.update_layout(xaxis_tickangle=-45)
    
    html_out_file = os.path.join(plot_folder, f"stacked_bar_{aggregator}.html")
    fig.write_html(html_out_file)
    print(f"[INFO] Interactive stacked bar chart (HTML) saved to {html_out_file}")
    
    # Optionally, display the interactive figure in a browser if desired.
    fig.show()

def plotLineOverTime(outdirs, primary_filter_column='source', primary_filter_value=None, line_aggregator='age_binned', filters=None, save_fig=True, plot_folder="plots"):
    """
    Creates a line chart showing the number of migrations over time.
    
    You can specify a primary filtering criterion (for example, only show data
    for a given source or destination) by setting:
      - primary_filter_column: the column to filter on (e.g., "source" or "destination")
      - primary_filter_value: the value in that column (e.g., "Germany" or "ukr_kyivska")
      
    In addition, you can specify which aggregator to use to split the lines (e.g., "age_binned",
    "gender", etc.). If "age_binned" is chosen, the function will bin the raw age data.
    
    The function creates both:
      1. A static PNG file (using matplotlib), and
      2. An interactive HTML file (using Plotly) with hover tooltips.
    
    Parameters
    ----------
    outdirs : list of str
        List of run directories (each containing a migration.log file).
    primary_filter_column : str, optional
        Column used for primary filtering (e.g., "source" or "destination"). Default is "source".
    primary_filter_value : str or None, optional
        If provided, only rows where primary_filter_column == primary_filter_value are used.
        If None, data from all values is included.
    line_aggregator : str, optional
        The column used to differentiate the line series (e.g., "age_binned", "gender", etc.).
    filters : list of tuples or None, optional
        Additional filters, each as a tuple (column, operator, value), e.g.,
        [('age', '>=', 18)]. Default is None.
    save_fig : bool, optional
        Whether to save output files. Default is True.
    plot_folder : str, optional
        Folder in which to save the output files. Default is "plots".
    """
    if filters is None:
        filters = []

    # 1. Read all migration.log files from each run directory.
    all_data = []
    for d in outdirs:
        migration_file = os.path.join(d, "migration.log")
        if not os.path.exists(migration_file):
            print(f"[WARNING] No migration.log in {d}; skipping.")
            continue
        try:
            df_run = pd.read_csv(migration_file)
            all_data.append(df_run)
        except Exception as e:
            print(f"[ERROR] Could not read {migration_file}: {e}")
            continue

    if not all_data:
        print("[INFO] No migration data found.")
        return

    # 2. Concatenate data from all runs.
    df_all = pd.concat(all_data, ignore_index=True)

    # 3. Apply additional filters.
    for (col, op, val) in filters:
        if op == "==":
            df_all = df_all[df_all[col] == val]
        elif op == "!=":
            df_all = df_all[df_all[col] != val]
        elif op == ">":
            df_all = df_all[df_all[col] > val]
        elif op == "<":
            df_all = df_all[df_all[col] < val]
        elif op == ">=":
            df_all = df_all[df_all[col] >= val]
        elif op == "<=":
            df_all = df_all[df_all[col] <= val]
        else:
            print(f"[WARNING] Unsupported operator '{op}' for filter {col} {op} {val}")

    # 4. Apply the primary filter (if a value is specified).
    if primary_filter_value is not None:
        if primary_filter_column not in df_all.columns:
            print(f"[ERROR] Column '{primary_filter_column}' not found in data.")
            return
        df_all = df_all[df_all[primary_filter_column] == primary_filter_value]
    
    # 5. If the line aggregator is "age_binned", bin ages.
    if line_aggregator == "age_binned":
        bins = [0, 17, 29, 49, 64, 200]   # Customize as desired
        labels = ["0-17", "18-29", "30-49", "50-64", "65+"]
        if "age" not in df_all.columns:
            print("[ERROR] Column 'age' not found; cannot bin ages.")
            return
        df_all['age_bin'] = pd.cut(df_all['age'], bins=bins, labels=labels, right=True)
        line_aggregator = "age_bin"  # Update to use binned ages

    # 6. Check that required columns exist.
    for needed in [line_aggregator, 'time']:
        if needed not in df_all.columns:
            print(f"[ERROR] Column '{needed}' not found in data.")
            return

    # 7. Group the data by time and line_aggregator (count rows for number of migrations).
    group_df = df_all.groupby(['time', line_aggregator]).size().reset_index(name='counts')

    # 8. Pivot so that each row is a time step and each column is a line series.
    pivot_df = group_df.pivot(index='time', columns=line_aggregator, values='counts').fillna(0)

    """
    # Optionally, sort the DataFrame by time.
    try:
        pivot_df.index = pd.to_datetime(pivot_df.index)
        pivot_df.sort_index(inplace=True)
    except Exception as e:
        # If time isn't a date, no problem.
        pass
    """

    # ******************************
    # Static Line Chart with Matplotlib
    # ******************************
    plt.figure(figsize=(12, 7))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], marker='o', label=str(column))
    plt.title(f"Number of Migrations Over Time\n(Filtered by {primary_filter_column}={primary_filter_value if primary_filter_value is not None else 'All'})\nGrouped by '{line_aggregator}'")
    plt.xlabel("Time")
    plt.ylabel("Number of Migrations")
    plt.legend(title=line_aggregator)
    plt.xticks(rotation=45)
    plt.tight_layout()

    Path(plot_folder).mkdir(parents=True, exist_ok=True)
    png_file = os.path.join(plot_folder, f"line_{primary_filter_column}_{primary_filter_value}_{line_aggregator}.png")
    if save_fig:
        plt.savefig(png_file, dpi=150)
        print(f"[INFO] Line chart (PNG) saved to {png_file}")
    plt.show()
    plt.close()


    # Interactive Line Chart with Plotly
    # Melt the pivot table to long format.
    df_melted = pivot_df.reset_index().melt(id_vars='time', var_name=line_aggregator, value_name='counts')

    # Create a Plotly line chart.
    fig = px.line(df_melted,
                  x='time',
                  y='counts',
                  color=line_aggregator,
                  title=f"Number of Migrations Over Time\n(Filtered by {primary_filter_column}={primary_filter_value if primary_filter_value is not None else 'All'})",
                  labels={'time': 'Time', 'counts': 'Number of Migrations'})
    # Forces x-axis to stay linear, not as dates (UNIX from 1970)
    fig.update_layout(xaxis_type='linear')
    fig.update_xaxes(tickmode='linear')
    
    '''
    # If time is a date, you can enforce date formatting:
    try:
        fig.update_xaxes(type='date')
    except:
        pass
    '''

    html_file = os.path.join(plot_folder, f"line_{primary_filter_column}_{primary_filter_value}_{line_aggregator}.html")
    fig.write_html(html_file)
    print(f"[INFO] Interactive line chart (HTML) saved to {html_file}")
    fig.show()
    
    
def plotNamedSingleByTimestep(code, outdirs, plot_type, FUMEheader, filters=[], aggregator=None):
    aggregator = getattr(FUMEheader, 'aggregator', 'age_binned')
    filters = getattr(FUMEheader, 'filters', [])
    
    # e.g. primary_filter_column = "source" or "destination"
    primary_filter_column = getattr(FUMEheader, 'primary_filter_column', 'destination')
    # e.g. primary_filter_value = "germany" or "ukr_kyivska" 
    primary_filter_value = getattr(FUMEheader, 'primary_filter_value', 'kyiv')
    
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
