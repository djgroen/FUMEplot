import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import ReadHeaders

def plotSourceHist(outdir, save_fig=False, plot_folder=None):
    
    # Read and aggregate data from multiple CSV files
    all_counts = []
    
    csv_files = []
    for name in os.listdir(outdir):
        csv_files.append(f"{outdir}/{name}/migration.log")
#    for i in range(1,11):
#       csv_files.append(f"../sample_homecoming_agentlog/{i}/migration.log")

    for file in csv_files:
        df = pd.read_csv(file)
        source_counts = df['source'].value_counts()
        all_counts.append(source_counts)
    
    # Combine counts into a DataFrame, filling missing values with 0
    combined_counts = pd.DataFrame(all_counts).fillna(0)
    
    # Compute mean and standard deviation for each source
    mean_counts = combined_counts.mean()
    std_counts = combined_counts.std()
    
        # Plot histogram with error bars
    plt.figure(figsize=(10, 6))
    plt.bar(mean_counts.index, mean_counts.values, yerr=std_counts.values, color='skyblue', capsize=5)

    # Labels and title
    plt.xlabel('Source Country')
    plt.ylabel('Number of Entries')
    plt.title('Histogram of Entries Grouped by Source')
    plt.xticks(rotation=45)
    
    if save_fig:
        plt.savefig(plot_folder+'/EntriesBySource.png')



if __name__ == "__main__":

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

