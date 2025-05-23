import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import json
import plotly.express as px
import glob 

from matplotlib.backends.backend_pdf import PdfPages
from contextlib import nullcontext

USELATEXPLOTLIB = False

if USELATEXPLOTLIB:
    import latexplotlib as lpl
    lpl.style.use('latex12pt')
    latex_doc_size = (347.12354, 549.138)
    columns_per_page = 2.
    latex_doc_size = tuple(x / columns_per_page for x in latex_doc_size)
    lpl.size.set(*latex_doc_size)


def _formatLabels(labels):

    # Capitalize and replace underscores with spaces
    labels =  [label.replace('_', ' ').title() for label in labels]

    return labels

def plotLocation(outdirs, plot_num, loc_index, sim_index, data_index, loc_names, y_label, save_fig=False, plot_folder=None, combine_plots_pdf=False):    
    ensembleSize = 0
    dfTest = []

    # loop through each ensemble job extracting sim data and assigning to df for each location
    for d in outdirs:
        df = pd.read_csv(f"{d}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        ensembleSize += 1
    
    # plot all waveforms
    # - LatexPlotlib version
    if USELATEXPLOTLIB:
        fig, ax = lpl.subplots(num=plot_num+1)
    else:
        fig = plt.figure(plot_num+1)
        ax = fig.add_subplot(111)
    
    # plot individual ensemble members
    for i in range(ensembleSize):
        ax.plot(dfTest[i],'k', alpha=0.15)
        #print(f"size of dfTest[i]: {len(dfTest[i])}") #debugging

    # plot ensemble mean
    ax.plot(np.mean(dfTest,axis=0), 'maroon', label='Ensemble Mean')

    # plot the reference data if available
    if data_index > 0:
        ax.plot(df.iloc[:, data_index], 'b-', label='UN Data')
        #print(f"sizes of dfTest and data: {len(dfTest[0])}, {len(df.iloc[:, data_index])}") #debugging

    # set up legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mlines.Line2D([], [], color='black', linestyle='-', label='Ensemble Members'))
    labels.append('Ensemble Members')
    ax.legend(handles=handles, labels=labels, loc='best')
    
    # set up formatting
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Returnees')
    ax.set_title(str(loc_names[loc_index]))

    ax.grid(visible=True, which='both', axis='both', linestyle='-', linewidth=0.33)
    #plt.tight_layout()
    
    # save the plot
    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    if save_fig:
        fig.savefig(plot_folder+'/'+str(loc_names[loc_index]).replace(" ", "").replace("#", "Num")+'_Ensemble.png')

    plt.close(fig)

def plotLocationSTDBound(outdirs, plot_num, loc_index, sim_index, data_index, loc_names, y_label, save_fig=False, plot_folder=None, combine_plots_pdf=False):
    ensembleSize = 0
    dfTest = []

    # loop through each ensemble job extracting sim data and assigning to df for each location
    for d in outdirs:
        df = pd.read_csv(f"{d}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        ensembleSize += 1
    
    # plot all waveforms
    # - LatexPlotlib version
    if USELATEXPLOTLIB:
        fig, ax = lpl.subplots(num=plot_num+1)
    else:
        fig = plt.figure(plot_num+1)
        ax = fig.add_subplot(111)

    # plot simulation ensemble mean +/- 1 standard deviation
    ax.fill_between(np.linspace(0,len(dfTest[0]),len(dfTest[0])), 
                                 np.mean(dfTest,axis=0) - np.std(dfTest,axis=0), 
                                 np.mean(dfTest,axis=0) + np.std(dfTest,axis=0), 
                                 where=np.ones(len(dfTest[0])), alpha=0.3, color='maroon', label=r'Mean $\pm$ 1 STD')
    #TODO: add 2,3 std deviations, percentiles, absolute min/max, mean error etc.
    ax.fill_between(np.linspace(0,len(dfTest[0]),len(dfTest[0])),
                                np.max(dfTest,axis=0),
                                np.min(dfTest,axis=0),
                                where=np.ones(len(dfTest[0])), alpha=0.1, color='maroon', label='Min - Max')
    
    # plot reference data if available
    if data_index > 0:
        ax.plot(df.iloc[:, data_index],'b-', label='UN Data')

    # set up formatting
    ax.legend(loc='best')
    ax.set_xlabel('Day')
    ax.set_ylabel(y_label)
    ax.set_title(str(loc_names[loc_index]))

    ax.grid(visible=True, which='both', axis='both', linestyle='-', linewidth=0.33,)
    #plt.tight_layout()
     
    # save the plot
    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    if save_fig:
        fig.savefig(plot_folder+'/'+str(loc_names[loc_index]).replace(" ", "").replace("#", "Num")+'_std.png')

    plt.close(fig)

def plotLocationDifferences(outdirs, plot_num, loc_index, sim_index, data_index, loc_names, save_fig=False, plot_folder=None, combine_plots_pdf=False):    
    ensembleSize = 0
    dfTest = []

    # loop through each ensemble job extracting sim data and assigning to df for each location
    for d in outdirs:
        df = pd.read_csv(f"{d}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        ensembleSize += 1
    
    rmse = np.sqrt(((np.mean(dfTest,axis=0) - df.iloc[:, data_index]) ** 2).mean())
    ard = np.abs(np.mean(dfTest,axis=0) - df.iloc[:, data_index]).mean()
    
    # add text box for the statistics
    stats = (f'RMSE = {rmse:.2f}\n'
             f'ARD = {ard:.2f}')
    
    # plot all waveforms
    # - LatexPlotlib version
    if USELATEXPLOTLIB:
        fig, ax = lpl.subplots(num=plot_num+1)
    else:
        fig = plt.figure(plot_num+1)
        ax = fig.add_subplot(111)
    
    ax.plot(np.mean(dfTest,axis=0) - df.iloc[:, data_index],'black')
        
    ax.plot([],label=stats)
    ax.legend(handlelength=0)
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Difference (sim - observed)')
    ax.set_title(str(loc_names[loc_index]))
    #plt.tight_layout()
 
    # save the plot
    if combine_plots_pdf:
        combine_plots_pdf.savefig(fig)
    if save_fig:
        fig.savefig(plot_folder+'/'+str(loc_names[loc_index]).replace(" ", "").replace("#", "Num")+'_Differences.png')

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def animateLocationHistogram(outdirs, plot_num, loc_index, sim_index, data_index, loc_names, x_label, save_fig=False, plot_folder=None, combine_plots_pdf=False):    
    ensembleSize = 0
    maxPop = 0
    dfTest = []

    # loop through each ensemble job extracting sim data and assigning to df for each location
    for d in outdirs:
        df = pd.read_csv(f"{d}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        if data_index > 0:
            if maxPop < max(max(df.iloc[:, sim_index]),max(df.iloc[:, data_index])):
                maxPop = max(max(df.iloc[:, sim_index]),max(df.iloc[:, data_index]))
        else:
            if maxPop < max(df.iloc[:, sim_index]):
                maxPop = max(df.iloc[:, sim_index])
        ensembleSize += 1
    
    def updatehist(i):
        ax.cla()
        hist = ax.hist([item[i] for item in dfTest], bins=10, color='c', edgecolor='k', alpha=0.65, label='Ensemble Data')
        if data_index > 0:
            data = ax.axvline(df.iloc[i, data_index], color='k', linestyle='dashed', linewidth=1, label='UN Data')
        ax.set_title(str(loc_names[loc_index] + ' - Day '+ str(i)))
        ax.set(xlim=[0, 1.1*maxPop], ylim=[0, ensembleSize], xlabel=x_label, ylabel='Ensemble Observations')
        #ax.set(ylim=[0, 10], xlabel='# Refugees', ylabel='Occurances')
        ax.grid(visible=True, which='both', axis='both', linestyle='-', linewidth=0.33,)
        ax.legend() 
        if data_index > 0:
            return (hist, data)
        else:
            return (hist)
        
    # - LatexPlotlib version
    if USELATEXPLOTLIB:
        fig, ax = lpl.subplots(num=plot_num+1)
    else:   
        fig = plt.figure(plot_num+1)
        ax = fig.add_subplot(111)

    ax.hist([item[0] for item in dfTest], bins=10, color='c', edgecolor='k', alpha=0.65, label='Ensemble Data')

    # plot the reference data if available
    if data_index > 0:
        ax.axvline(df.iloc[0, data_index], color='k', linestyle='dashed', linewidth=1, label='UN Data')
    ax.set_title(str(loc_names[loc_index] + ' - Day '+ str(0)))

    # set up formatting
    ax.set(xlim=[0, 1.1*maxPop], ylim=[0, ensembleSize], xlabel=x_label, ylabel='Ensemble Observations')
    #plt.xticks(rotation=45)
    ax.grid(visible=True, which='both', axis='both', linestyle='-', linewidth=0.33,)
    ax.legend()
    
    ani = animation.FuncAnimation(fig, updatehist, len(dfTest[0]))

    # save the plot   
    if save_fig:
       ani.save(filename=plot_folder+'/'+str(loc_names[loc_index]).replace(" ", "").replace("#", "Num")+'_Histogram.gif', writer="pillow")

    plt.close(fig)

def animateLocationViolins(outdirs, plot_num, i, sim_indices, data_indices, loc_names, y_label, save_fig=False, plot_folder=None, combine_plots_pdf=False):    
    ensembleSize = 0
    maxPop = 0
    dfFull = []

    # loop through each ensemble job extracting sim data and assigning to df for each location
    for d in outdirs:
        dfTest=[]
        df = pd.read_csv(f"{d}/out.csv")
        
        for sim_index in sim_indices:
            dfTest.append(df.iloc[:, sim_index].T)

        dfFull.append(dfTest)
        ensembleSize += 1
    
    dataAvailable=False
    for d in data_indices:
        if d > 0:
            dataAvailable=True
    
    if dataAvailable:
        dataValues = df.iloc[:, data_indices] #As in flee output, data columns the same in all ensemble runs
    
    def updateviolin(i):
        ax.cla()
        locData=[]
        for j in range(ensembleSize):
            locData.append([item[i] for item in dfFull[j]])

        locData = [list(k) for k in zip(*locData)]

        quartile1, medians, quartile3 = np.percentile(locData, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(locData, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        hist = ax.violinplot(
            locData, 
            showmeans=False, 
            showmedians=True, 
            showextrema=False,
            )

        # plot the reference data if available
        if dataAvailable:
            ax.scatter([y + 1 for y in range(len(sim_indices))], dataValues.iloc[i].values, color='r', label='UN Data')
            ax.vlines([y + 1 for y in range(len(sim_indices))], quartile1, quartile3, color='k', linestyle='-', lw=1)
            #ax.vlines([y + 1 for y in range(len(sim_indices))], whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

            # where some data has already been plotted to ax
            handles, labels = ax.get_legend_handles_labels()
            handles.append(mpatches.Patch(color='C0', label='Simulations')) 
            ax.legend(handles=handles)

        # set up formatting
        ax.set_title(f"{y_label} - Day {i}")
        ax.set(xlabel='Category', ylabel='\# of Observations')
        ax.yaxis.grid(True)
        ax.set_xticks([y+1 for y in range(len(sim_indices))])
        ax.set_xticklabels(loc_names) #as old version of matplotlib
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        return (hist)
    
    if USELATEXPLOTLIB:
        # - LatexPlotlib version
        fig, ax = lpl.subplots(num=plot_num+1)
    else:
        # - Matplotlib version
        fig = plt.figure(plot_num+1)
        ax = fig.add_subplot(111)

    locData=[]
    for j in range(ensembleSize):
        locData.append([item[0] for item in dfFull[j]])
    locData = [list(i) for i in zip(*locData)]

    quartile1, medians, quartile3 = np.percentile(locData, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(locData, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    ax.violinplot(locData, showmeans=False, showmedians=True)

    # plot the reference data if available
    if dataAvailable:
        ax.scatter([y + 1 for y in range(len(sim_indices))], dataValues.iloc[0].values, color='r', label='UN Data')
        ax.vlines([y + 1 for y in range(len(sim_indices))], quartile1, quartile3, color='k', linestyle='-', lw=1)
#ax.vlines([y + 1 for y in range(len(sim_indices))], whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        # where some data has already been plotted to ax
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color='C1', label='Simulations')) 
        ax.legend(handles=handles)
        
    # set up formatting
    ax.set_title(f"{y_label} - Day {i}")
    ax.set(xlabel='Category', ylabel='\# of Observations')
    ax.yaxis.grid(True, linewidth=0.33)
    ax.set_xticks([y + 1 for y in range(len(sim_indices))])
    ax.set_xticklabels(loc_names)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, updateviolin, len(dfFull[0][0]))
    
    # save the plot
    if save_fig:
       ani.save(filename=plot_folder+'/Overall_Violin.gif', writer="pillow")

    plt.close(fig)
    
def _load_ukraine_geo(geojson_dir):
    """Merge all UA_*.geojson files in geojson_dir into one FeatureCollection."""
    files = glob.glob(os.path.join(geojson_dir, "UA_*.geojson"))
    if not files:
        raise FileNotFoundError(f"No UA_*.geojson found in {geojson_dir}")
    merged = {"type": "FeatureCollection", "features": []}
    for fn in files:
        g = json.load(open(fn, "r"))
        # some files are single‐feature, some multi‐feature
        feats = g.get("features", [g])  
        merged["features"].extend(feats)
    return merged

def plotChoropleth(outdirs, plot_folder="plots", geojson_dir=None):
    """
    Reads each out.csv, takes the final time‐step counts for every ukr-<oblast> column,
    averages across ensemble runs, then draws a Plotly choropleth of mean returnees
    by Ukrainian oblast.
    """
    # 1. gather final‐row data
    run_counts = []
    for d in outdirs:
        csv_path = os.path.join(d, "out.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] {csv_path} missing – skipping.")
            continue
        df = pd.read_csv(csv_path)
        # select only the ukr‐* columns (assumes first column is 'time')
        oblast_cols = [c for c in df.columns if c.startswith("ukr-")]
        last = df.iloc[-1][oblast_cols]
        run_counts.append(last.values.astype(float))
    if not run_counts:
        print("[INFO] No runs found for choropleth → skipping.")
        return

    # stack into array (runs × oblasts) and compute mean
    arr = np.vstack(run_counts)
    mean_vals = arr.mean(axis=0)

    # 2. build summary DataFrame
    oblast_cols = oblast_cols  # same order for every run
    # strip "ukr-", replace hyphens with spaces, title‐case to match NAME_1
    dests = [c.replace("ukr-", "").replace("-", " ").title() for c in oblast_cols]
    df_summary = pd.DataFrame({
        "destination": dests,
        "mean_count": mean_vals
    })

    # 3. load GeoJSON
    if geojson_dir is None:
        # assume located alongside this script
        here = Path(__file__).parent
        geojson_dir = here / "ukraine_geojson"
    ukraine_geo = _load_ukraine_geo(str(geojson_dir))

    # 4. make the choropleth
    fig = px.choropleth(
        df_summary,
        geojson=ukraine_geo,
        locations="destination",
        featureidkey="properties.NAME_1",
        color="mean_count",
        color_continuous_scale="Viridis",
        title="Mean Returnees by Ukrainian Oblast (final time‐step)",
        labels={"mean_count": "Mean # Returnees"}
    )
    # zoom to Ukraine, hide frame
    fig.update_geos(fitbounds="locations", visible=False)

    # styling
    fig.update_layout(
        title_font_size=24,
        font=dict(size=16),
        margin={"l":0,"r":0,"t":50,"b":0}
    )

    # 5. save
    os.makedirs(plot_folder, exist_ok=True)
    out_html = os.path.join(plot_folder, "choropleth_returnees.html")
    fig.write_html(out_html)
    print(f"[INFO] Saved choropleth to {out_html}")

def plotFinalOblastBarHTML(outdirs, plot_folder="plots"):
    """
    Reads the final row of out.csv for each ensemble in `outdirs`,
    selects only the ukr-<oblast> columns, computes the mean across runs,
    and writes an interactive HTML bar chart of mean returnees by oblast.
    """
    # 1. collect the last‐row values for each run
    run_vals = []
    oblast_cols = None
    for d in outdirs:
        csv_path = os.path.join(d, "out.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] {csv_path} missing; skipping.")
            continue
        df = pd.read_csv(csv_path)
        # on first run capture which columns to use
        if oblast_cols is None:
            oblast_cols = [c for c in df.columns if c.startswith("ukr-")]
        # grab last row
        run_vals.append(df.iloc[-1][oblast_cols].astype(float).values)

    if not run_vals:
        print("[INFO] No data for final‐bar → skipping.")
        return

    # 2. compute mean across runs
    arr = np.vstack(run_vals)        # shape: (n_runs, n_oblasts)
    mean_counts = arr.mean(axis=0)   # length = n_oblasts

    # 3. build dataframe for Plotly
    # clean up the names: drop "ukr-", hyphens→spaces, title case
    names = [c.replace("ukr-", "").replace("-", " ").title()
             for c in oblast_cols]
    df_plot = pd.DataFrame({
        "Oblast": names,
        "Mean Returnees": mean_counts
    })

    # 4. make the bar chart
    fig = px.bar(
        df_plot,
        x="Oblast",
        y="Mean Returnees",
        title="Mean Returnees by Ukrainian Oblast (final timestep)",
        labels={"Mean Returnees": "Net No. of Returnees (x1000)","Oblast": "Oblast Regions"},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=40, r=20, t=50, b=120),
        title_font_size=25,
        font=dict(size=18),
        xaxis_title_font_size=24,
        yaxis_title_font_size=24,
        xaxis_tickfont_size=20,
        yaxis_tickfont_size=20,
    )

    # 5. write out the HTML
    os.makedirs(plot_folder, exist_ok=True)
    out_html = os.path.join(plot_folder, "final_oblast_bar.html")
    fig.write_html(out_html)
    print(f"[INFO] Saved final‐bar HTML to {out_html}")

#main plotting script
def plotNamedStocksByTimestep(code, outdirs, plot_type, FUMEheader, plot_path='../..'):

    headers = FUMEheader.headers
    sim_indices = FUMEheader.sim_indices
    data_indices = FUMEheader.data_indices
    loc_names= FUMEheader.loc_names
    y_label = (FUMEheader.y_label).replace('#', 'Number')
    combine_plots_pdf = FUMEheader.combine_plots_pdf
 
    ensembleSize = 0
    
    saving=True
    plotfolder=plot_path+'/EnsemblePlots/'+code+'Plots'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)
   
    #fi = 0 #fig number #OG LINE
    
    # ----- SPECIAL CASE: default run  # ----------------------------------
    if plot_type == 'default':
        # default modes:
        sub_modes = ['loc_lines','loc_stdev','loc_hist_gif']
        # which location names we want by default:
        defaults  = {'ukr-kyivska','ukr-lvivska', 'ukr-dnipropetrovska','poland','romania','germany'}
        # find their indices
        idxs = [i for i,name in enumerate(loc_names) if name in defaults]
        
        # if no exact match, fall back to everything
        if not idxs:
            idxs = list(range(len(loc_names)))

        # bundle into one PDF if requested
        pdf_ctx = (PdfPages(plotfolder+"/combined_time_plots.pdf")
                   if combine_plots_pdf else nullcontext())
        with pdf_ctx as pdf_pages:
            fi = 0
            for mode in sub_modes:
                for i in idxs:
                    sim_i  = sim_indices[i]
                    data_i = data_indices[i]

                    if mode == 'loc_lines':
                        plotLocation(outdirs, fi, i, sim_i, data_i, loc_names, y_label,
                                     save_fig=True, plot_folder=str(plotfolder),
                                     combine_plots_pdf=pdf_pages)
                    elif mode == 'loc_stdev':
                        plotLocationSTDBound(outdirs, fi, i, sim_i, data_i, loc_names, y_label, save_fig=True, plot_folder=str(plotfolder), combine_plots_pdf=pdf_pages)
                    else:  # loc_hist_gif
                        animateLocationHistogram(outdirs, fi, i, sim_i, data_i, loc_names, y_label, save_fig=True, plot_folder=str(plotfolder), combine_plots_pdf=pdf_pages)
                    fi += 1

            # if running interactively, pop up the figures
            if not combine_plots_pdf:
                plt.show()

        return
        # ---------------------------------------------------------------

    with PdfPages(os.path.join(plotfolder, "combined_time_plots.pdf")) if combine_plots_pdf else nullcontext() as pdf_pages:
        fi = 0 #NOT OG
        for i in range(len(sim_indices)):
            if plot_type == "loc_lines" or plot_type == "all":
                plotLocation(outdirs, fi, i, sim_indices[i], data_indices[i], loc_names, y_label, save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)
                fi += 1
            if plot_type == "loc_stdev" or plot_type == "all":
                plotLocationSTDBound(outdirs, fi, i, sim_indices[i], data_indices[i], loc_names, y_label, save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)
                fi += 1
        
            if data_indices[i]>0:
                if plot_type == "loc_diff" or plot_type == "all":
                    plotLocationDifferences(outdirs, fi, i, sim_indices[i], data_indices[i], loc_names, save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)
                    fi += 1
    
            if plot_type == "loc_hist_gif" or plot_type == "all":
               animateLocationHistogram(outdirs, fi, i, sim_indices[i], data_indices[i], loc_names, y_label, save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)
               fi += 1

            if plot_type == "loc_violin_gif" or plot_type == "all":
                animateLocationViolins(outdirs, fi, i, sim_indices, data_indices, loc_names, y_label, save_fig=saving, plot_folder=plotfolder, combine_plots_pdf=pdf_pages)
                
            if plot_type == "choropleth" or plot_type == "all":
                plotChoropleth(outdirs, plot_folder=plotfolder, geojson_dir=os.path.join(os.path.dirname(__file__), "ukraine_geojson"))
            
            if plot_type in ("bar_chart", "all"):
                plotFinalOblastBarHTML(outdirs, plot_folder=plotfolder)

    if not combine_plots_pdf:
        plt.show()


if __name__ == "__main__":
    import ReadHeaders

    code = "homecoming" #flee, facs or homecoming
    plot_type = "all"
    if len(sys.argv) > 1:
        code = sys.argv[1]
        if len(sys.argv) > 2:
            plot_type = sys.argv[2]

    outdir = f"../sample_{code}_output"
   
    outdirs = ReadHeaders.GetOutDirs(outdir)
    FUMEheader = ReadHeaders.ReadOutHeaders(outdirs, mode=code)
    plotNamedStocksByTimestep(code, outdirs, plot_type, FUMEheader)


# ISSUES:
 
# - save the plots as a single pdf in a folder:
#       split histograms and violin plots into separate frames as save in a PDF

# histogram: change to log scale (break the Y axis?)
# animations: try interactive plots to slide or play though days
