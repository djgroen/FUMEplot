import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import sys
from pathlib import Path

def ReadCampHeaders(outdir, mode="flee"):

    headers = []
    numCamps = 0
    camp_names = []
    camp_indices = []

    #Read first results file header for variable names
    for name in os.listdir(outdir):
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
        headers = list(df)
        numCamps = int((df.shape[1]-8)/3)
        break

    sim_indices = []
    data_indices = []

    for i in range(numCamps):
        if mode == "flee":
            sim_indices.append(3*i+2)
            data_indices.append(3*i+3)
        if mode == "homecoming":
            sim_indices.append(i+1)
            data_indices.append(-1) #indicates no data.


    return headers, sim_indices, data_indices


def plotCamp(outdir, plot_num, sim_index, data_index, save_fig=False, plot_folder=None):    
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
    plt.ylabel('# of asylum seekers or unrecognized refugees')
    plt.title(str(headers[data_index]))
    
    if save_fig:
        plt.savefig(plot_folder+'/'+str(headers[data_index]).replace(" ", "")+'_Ensemble.png')

def plotCampSTDBound(outdir, plot_num, sim_index, data_index, save_fig=False, plot_folder=None):
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
    plt.ylabel('# of asylum seekers or unrecognized refugees')
    plt.title(str(headers[data_index]))
     
    if save_fig:
        plt.savefig(plot_folder+'/'+str(headers[data_index]).replace(" ", "")+'_std.png')


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
    plt.title(str(headers[data_index]))
 
    if save_fig:
        plt.savefig(plot_folder+'/'+str(headers[data_index]).replace(" ", "")+'_Differences.png')

def animateCampHistogram(outdir, plot_num, sim_index, data_index, save_fig=False, plot_folder=None):    
    ensembleSize = 0
    dfTest = []
    # loop through each ensemble job extracting sim data and assigning to df for each campsite
    for name in os.listdir(outdir):
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
        dfTest.append(df.iloc[:, sim_index].T)
        ensembleSize += 1
    
    
    #min_ylim, max_ylim = plt.ylim()
    #plt.text(x.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(x.mean())) 
    # plot all waveforms
    
    def updatehist(i):
        ax.cla()
        hist = ax.hist([item[i] for item in dfTest], bins=10, color='c', edgecolor='k', alpha=0.65, label='Ensemble data')
        data = ax.axvline(df.iloc[i, data_index], color='k', linestyle='dashed', linewidth=1, label='UN data')
        ax.set_title(str(headers[data_index] + ' - Day '+ str(i)))
        #ax.set(xlim=[0, 3000], ylim=[0, 10], xlabel='# Refugees', ylabel='Occurances')
        ax.set(ylim=[0, 10], xlabel='# Refugees', ylabel='Occurances')
        ax.legend()
        return (hist, data)
        
    #fig = plt.figure(plot_num+1)
    fig, ax = plt.subplots()
    hist=ax.hist([item[0] for item in dfTest], bins=10, color='c', edgecolor='k', alpha=0.65, label='Ensemble data')
    data=ax.axvline(df.iloc[0, data_index], color='k', linestyle='dashed', linewidth=1, label='UN data')
    ax.set_title(str(headers[data_index] + ' - Day '+ str(0)))

    # set 
    ax.set(xlim=[0, 3000], ylim=[0, 10], xlabel='# Refugees', ylabel='Occurances')
    ax.set_ylim([0,10])
    ax.legend()
    
    ani = animation.FuncAnimation(fig, updatehist, len(dfTest[0]) )
        
    if save_fig:
       ani.save(filename=plot_folder+'/'+str(headers[data_index]).replace(" ", "")+'histo.gif', writer="pillow")


#main plotting script
if __name__ == "__main__":
  
    code = "flee"
    if len(sys.argv) > 1:
        code = sys.argv[1]

    outdir = f"sample_{code}_output"

    headers, sim_indices, data_indices = ReadCampHeaders(outdir)

    ensembleSize = 0
    
    saving=False
    plotfolder=code+'Plots'
    Path(plotfolder).mkdir(parents=True, exist_ok=True)

#for i in range(len(sim_indices)):
for i in range(1):
    plotCamp(outdir, 4*i, sim_indices[i], data_indices[i],save_fig=saving, plot_folder=plotfolder)
    plotCampSTDBound(outdir, 4*i+1, sim_indices[i], data_indices[i],save_fig=saving, plot_folder=plotfolder)
    plotCampDifferences(outdir, 4*i+2, sim_indices[i], data_indices[i],save_fig=saving, plot_folder=plotfolder)
    animateCampHistogram(outdir, 4*i+3, sim_indices[i], data_indices[i],save_fig=True, plot_folder=plotfolder)

    plt.show()


