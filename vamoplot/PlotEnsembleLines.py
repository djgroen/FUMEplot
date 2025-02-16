import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.patches as mpatches
import ReadHeaders

def plotCamp(outdir, plot_num, sim_index, data_index, y_label, save_fig=False, plot_folder=None):    
    ensembleSize = 0
    dfTest = []
    # loop through each ensemble job extracting sim data and assigning to df for each campsite
    for name in os.listdir(outdir):
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        ensembleSize += 1
    
    # plot all waveforms
    plt.figure(plot_num+1)
    
    for i in range(ensembleSize):
        plt.plot(dfTest[i],'k', alpha=0.2)
    plt.plot(np.mean(dfTest,axis=0),'maroon',label='ensemble mean')
    if data_index > 0:
        plt.plot(df.iloc[:, data_index],'b-', label='UN Data')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel(y_label)
    plt.title(str(headers[sim_index]))
    
    if save_fig:
        plt.savefig(plot_folder+'/'+str(headers[sim_index]).replace(" ", "")+'_Ensemble.png')

def plotCampSTDBound(outdir, plot_num, sim_index, data_index, y_label, save_fig=False, plot_folder=None):
    ensembleSize = 0
    dfTest = []
    # loop through each ensemble job extracting sim data and assigning to df for each campsite
    for name in os.listdir(outdir):
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        ensembleSize += 1
    
    # plot all waveforms
    plt.figure(plot_num+1)
    
    plt.plot(np.mean(dfTest,axis=0),'maroon',label='ensemble mean')
    plt.fill_between(np.linspace(0,len(dfTest[0]),len(dfTest[0])), 
                                 np.mean(dfTest,axis=0) - np.std(dfTest,axis=0), 
                                 np.mean(dfTest,axis=0) + np.std(dfTest,axis=0), 
                                 where=np.ones(len(dfTest[0])), alpha=0.3, color='maroon', label=r'mean $\pm$ 1 std')
    if data_index > 0:
        plt.plot(df.iloc[:, data_index],'b-', label='UN Data')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel(y_label)
    plt.title(str(headers[sim_index]))
     
    if save_fig:
        plt.savefig(plot_folder+'/'+str(headers[sim_index]).replace(" ", "")+'_std.png')


def plotCampDifferences(outdir, plot_num, sim_index, data_index, save_fig=False, plot_folder=None):    
    ensembleSize = 0
    dfTest = []
    # loop through each ensemble job extracting sim data and assigning to df for each campsite
    for name in os.listdir(outdir):
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        ensembleSize += 1
    
    rmse = np.sqrt(((np.mean(dfTest,axis=0) - df.iloc[:, data_index]) ** 2).mean())
    ard = np.abs(np.mean(dfTest,axis=0) - df.iloc[:, data_index]).mean()
    # add text box for the statistics
    stats = (f'RMSE = {rmse:.2f}\n'
             f'ARD = {ard:.2f}')
    
    # plot all waveforms
    plt.figure(plot_num+1)
    
    plt.plot(np.mean(dfTest,axis=0) - df.iloc[:, data_index],'black')
        
    plt.plot([],label=stats)
    plt.legend(handlelength=0)
    
    plt.xlabel('Day')
    plt.ylabel('Difference (sim - observed)')
    plt.title(str(headers[sim_index]))
 
    if save_fig:
        plt.savefig(plot_folder+'/'+str(headers[data_index]).replace(" ", "")+'_Differences.png')

def animateCampHistogram(outdir, plot_num, sim_index, data_index, x_label, save_fig=False, plot_folder=None):    
    ensembleSize = 0
    maxPop = 0
    dfTest = []
    # loop through each ensemble job extracting sim data and assigning to df for each campsite
    for name in os.listdir(outdir):
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
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
        hist = ax.hist([item[i] for item in dfTest], bins=10, color='c', edgecolor='k', alpha=0.65, label='Ensemble data')
        if data_index > 0:
            data = ax.axvline(df.iloc[i, data_index], color='k', linestyle='dashed', linewidth=1, label='UN data')
        ax.set_title(str(headers[sim_index] + ' - Day '+ str(i)))
        ax.set(xlim=[0, 1.1*maxPop], ylim=[0, 10], xlabel='# Refugees', ylabel='Occurances')
        #ax.set(ylim=[0, 10], xlabel='# Refugees', ylabel='Occurances')
        ax.legend()
        if data_index > 0:
            return (hist, data)
        else:
            return (hist)
        
    #fig = plt.figure(plot_num+1)
    fig, ax = plt.subplots()
    ax.hist([item[0] for item in dfTest], bins=10, color='c', edgecolor='k', alpha=0.65, label='Ensemble data')
    if data_index > 0:
        ax.axvline(df.iloc[0, data_index], color='k', linestyle='dashed', linewidth=1, label='UN data')
    ax.set_title(str(headers[sim_index] + ' - Day '+ str(0)))

    # set 
    ax.set(xlim=[0, 1.1*maxPop], ylim=[0, 10], xlabel=x_label, ylabel='Occurances')
    ax.legend()
    
    ani = animation.FuncAnimation(fig, updatehist, len(dfTest[0]))
        
    if save_fig:
       ani.save(filename=plot_folder+'/'+str(headers[sim_index]).replace(" ", "")+'_Histogram.gif', writer="pillow")

def animateCampViolins(outdir, plot_num, sim_indices, data_indices, camp_names, y_label, save_fig=False, plot_folder=None):    
    ensembleSize = 0
    maxPop = 0
    dfFull = []
    # loop through each ensemble job extracting sim data and assigning to df for each campsite
    for name in os.listdir(outdir):
        dfTest=[]
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
        
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
        campData=[]
        for j in range(ensembleSize):
            campData.append([item[i] for item in dfFull[j]])
        campData = [list(k) for k in zip(*campData)]
        hist = ax.violinplot(campData, showmeans=False, showmedians=True)
        if dataAvailable:
            ax.scatter([y + 1 for y in range(len(sim_indices))], dataValues.iloc[i].values, color='r', label='UN data')
            # where some data has already been plotted to ax
            handles, labels = ax.get_legend_handles_labels()
            handles.append(mpatches.Patch(color='C0', label='Simulations')) 
            ax.legend(handles=handles)
        ax.set_title(f"{y_label} - Day {i}")
        ax.set(xlabel='Campsite', ylabel='Observations')
        ax.yaxis.grid(True)
        ax.set_xticks([y+1 for y in range(len(sim_indices))])
        ax.set_xticklabels(camp_names) #as old version of matplotlib
        ax.tick_params(axis='x', rotation=90)
        plt.tight_layout()

        return (hist)
        
    #fig = plt.figure(plot_num+1)
    fig, ax = plt.subplots()
    campData=[]
    for j in range(ensembleSize):
        campData.append([item[0] for item in dfFull[j]])
    campData = [list(i) for i in zip(*campData)]
    ax.violinplot(campData, showmeans=False, showmedians=True)
    if dataAvailable:
        ax.scatter([y + 1 for y in range(len(sim_indices))], dataValues.iloc[0].values, color='r', label='UN data')
        
        # where some data has already been plotted to ax
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color='C1', label='Simulations')) 
        ax.legend(handles=handles)
        
    ax.set_title(f"{y_label} - Day {i}")
    ax.set(xlabel='Campsite', ylabel='Observations')
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(sim_indices))])
    ax.set_xticklabels(camp_names)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, updateviolin, len(dfFull[0][0]))
        
    if save_fig:
       ani.save(filename=plot_folder+'/Overall_Violin.gif', writer="pillow")


#main plotting script
if __name__ == "__main__":
  
    
    #code = "homecoming" #flee or homecoming
    code = "flee" #flee or homecoming
    plot_type = "all"
    if len(sys.argv) > 1:
        code = sys.argv[1]
        if len(sys.argv) > 2:
            plot_type = sys.argv[2]

    outdir = f"../sample_{code}_output"

    headers, sim_indices, data_indices, camp_names, y_label = ReadHeaders.ReadOutHeaders(outdir, mode=code)

    ensembleSize = 0
    
    saving=True
    plotfolder=code+'Plots'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)
   
    fi = 0 #fig number

    for i in range(len(sim_indices)):
        if plot_type == "camp_lines" or plot_type == "all":
            plotCamp(outdir, fi, sim_indices[i], data_indices[i], y_label, save_fig=saving, plot_folder=plotfolder)
            fi += 1
        if plot_type == "camp_stdev" or plot_type == "all":
            plotCampSTDBound(outdir, fi, sim_indices[i], data_indices[i], y_label, save_fig=saving, plot_folder=plotfolder)
            fi += 1
       
        if data_indices[i]>0:
            if plot_type == "camp_diff" or plot_type == "all":
                plotCampDifferences(outdir, fi, sim_indices[i], data_indices[i], save_fig=saving, plot_folder=plotfolder)
                fi += 1

    for i in range(len(sim_indices)): 
        if plot_type == "camp_hist_gif" or plot_type == "all":
            animateCampHistogram(outdir, fi, sim_indices[i], data_indices[i], y_label, save_fig=saving, plot_folder=plotfolder)
            fi += 1

        if plot_type == "camp_violin_gif" or plot_type == "all":
            animateCampViolins(outdir, 4*len(sim_indices), sim_indices, data_indices, camp_names, y_label, save_fig=True, plot_folder=plotfolder)

    plt.show()


