
""" This script generates various plots for named Ratios by Timestep. 
    By Ratio it is a propotion of the total number of agents being plotted, 
    Eg: 1:3 or 20% went to a location and 80% in another out of 100%. """
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

""" This script generates various plots for named single by Timestep. 
    By single it is indiviudal agents being plotted, 
    Eg: a geospace plot showing the location of individual agents at a time step and comparing them. """

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


def plotNamedRatioByTimestep(code, outdirs, plot_type, FUMEheader, filters=[], disaggregator=None, primary_filter_column=None, primary_filter_value=None, plot_path='../..'):
   
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

    plotNamedRatioByTimestep(code, outdirs, plot_type, FUMEheader)


# ISSUES:
# - Flee and Facs require _agentlog folders to be present

# error bar: add max-min interval, 25-75 percentiles, std(?)
