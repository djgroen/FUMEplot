import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches

def plotCounts(plot_num, all_counts, save_fig, plot_folder):
    # Combine counts into a DataFrame, filling missing values with 0
    combined_counts = pd.DataFrame(all_counts).fillna(0)

    # Compute mean and standard deviation for each source
    mean_counts = combined_counts.mean()
    std_counts = combined_counts.std()

        # Plot histogram with error bars
    plt.figure(plot_num+1, figsize=(10, 6))
    plt.bar(mean_counts.index, mean_counts.values, yerr=std_counts.values, color='skyblue', capsize=5)

    # Labels and title
    plt.xlabel('Source Country')
    plt.ylabel('Number of Entries')
    plt.title('Histogram of Entries Grouped by Source')
    plt.xticks(rotation=45)

    if save_fig:
        plt.savefig(plot_folder+'/EntriesBySource.png')



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


def plotSourceHist(outdir, save_fig=False, plot_folder=None):
    
    # Read and aggregate data from multiple CSV files
    all_counts = []
    
    csv_files = []
    for name in os.listdir(outdir):
        csv_files.append(f"{outdir}/{name}/migration.log")

    for file in csv_files:
        df = pd.read_csv(file)
        source_counts = df['source'].value_counts()
        all_counts.append(source_counts)
   
    plotCounts(0, all_counts, save_fig, plot_folder)

    all_counts = _getFilteredCounts(csv_files, "gender == f")
    plotCounts(1, all_counts, save_fig, plot_folder)

    all_counts = _getFilteredCounts(csv_files, "gender == m")
    plotCounts(2, all_counts, save_fig, plot_folder)

    plt.show()
    
def plotNamedSingleByTimestep(code, outdir, plot_type, FUMEheader):
    plotSourceHist(outdir, save_fig=False, plot_folder=None)



if __name__ == "__main__":

    import ReadHeaders
    code = "homecoming" 
    
    plot_type = "all"
    if len(sys.argv) > 1:
        code = sys.argv[1]
        if len(sys.argv) > 2:
            plot_type = sys.argv[2]

    outdir = f"../sample_{code}_agentlog"

    headers = ReadHeaders.ReadMovelogHeaders(outdir, mode=code)

    ensembleSize = 0
    
    saving=True
    plotfolder='../../EnsemblePlots/'+code+'Plots'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)
    
    fi=0
    if plot_type == "source_hist" or plot_type == "all":
        plotSourceHist(outdir, save_fig=saving, plot_folder=plotfolder)

    

    
    # Show plot
    plt.show()

