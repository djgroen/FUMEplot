import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from matplotlib.backends.backend_pdf import PdfPages
from contextlib import nullcontext


def _formatLabels(labels):

    # Capitalize and replace underscores with spaces
    labels =  [label.replace('_', ' ').title() for label in labels]

    return labels

def plotCounts(plot_num, all_counts, save_fig, plot_folder, combine_plots_pdf):
    # Combine counts into a DataFrame, filling missing values with 0
    combined_counts = pd.DataFrame(all_counts).fillna(0)

    # Compute mean and standard deviation for each source
    mean_counts = combined_counts.mean()
    std_counts = combined_counts.std()

    # Plot histogram with error bars
    fig = plt.figure(plot_num+1, figsize=(15, 9))

    # plt.bar(mean_counts.index, 
    #         mean_counts.values, 
    #         yerr=std_counts.values, 
    #         color='skyblue', 
    #         capsize=5,
    #         )

    plt.boxplot(
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
    plt.legend(
        [mlines.Line2D([], [], color='black', linestyle='dashed', label='Mean'),
         mlines.Line2D([], [], color='blue', label='Median'),
         mpatches.Patch(color='skyblue', label='Q1-Q3'),
         mlines.Line2D([], [], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Outliers')], 
        ['Mean', 'Median', '25th-75th percentile', 'Outliers'],)
    plt.xlabel('Source Location')
    plt.ylabel('Number of Entries')
    plt.title('Boxplot of Entries Grouped by Source')
    plt.xticks(rotation=45)
    #plt.axis('tight')
    plt.tight_layout()

    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
        print(f"Saved plot {plot_num} to PDF.")
    if save_fig:
        plt.savefig(plot_folder+'/EntriesBySource.png')
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
    
def plotNamedSingleByTimestep(code, outdir, plot_type, FUMEheader, filters=[]):
    plotSourceHist(outdir, filters, save_fig=False, plot_folder=None, combine_plots_pdf=FUMEheader.combine_plots_pdf)


if __name__ == "__main__":

    import ReadHeaders
    code = "homecoming" 
    
    plot_type = "all"
    if len(sys.argv) > 1:
        code = sys.argv[1]
        if len(sys.argv) > 2:
            plot_type = sys.argv[2]

    outdir = f"../sample_{code}_agentlog"

    outdirs = ReadHeaders.GetOutDirs(outdir)

    headers = ReadHeaders.ReadMovelogHeaders(outdirs, mode=code)

    # ensembleSize = 0
    ensembleSize = 8
    
    saving=True
    plotfolder='../../EnsemblePlots/'+code+'Plots'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)
    
    fi=0
    if plot_type == "source_hist" or plot_type == "all":
        plotSourceHist(outdirs, filters=[], save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=True)

    # Show plot
    plt.show()


# ISSUES:
# - Flee and Facs require _agentlog folders to be present

# error bar: add max-min interval, 25-75 percentiles, std(?)