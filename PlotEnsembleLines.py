import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


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


def plotFleeCamp(outdir, plot_num, sim_index, data_index):    
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

def plotFleeCampSTDBound(outdir, plot_num, sim_index, data_index):    
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

#main plotting script
if __name__ == "__main__":
    outdir_flee = "sample_flee_output"
    outdir_homecoming = "sample_homecoming_output"

    headers, sim_indices, data_indices = ReadCampHeaders(outdir_flee)

    ensembleSize = 0

    for i in range(len(sim_indices)):
        plotFleeCamp(outdir_flee, i, sim_indices[i], data_indices[i])
        plotFleeCampSTDBound(outdir_flee, 10*i, sim_indices[i], data_indices[i])

    #plot mean against quartile range for uncertainty
    plt.show()


