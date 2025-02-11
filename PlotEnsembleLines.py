import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# This is some script that plots hundreds of waveforms in greyscale and the mean in colour
# # # plot blackwaveforms
# # alpha = 0.01
# # plt.figure(1)
# # t = np.arange(100)
# # plt.plot(t,u[:462,:].T,'k',alpha=alpha)
# # plt.plot(t,np.mean(u[:729],axis=0),'b')
# # plt.plot(np.mean(u,axis=0),'r')
# # plt.title("Inlet velocity (ICA)")
# # plt.xlabel('Cardiac cycle [%]')
# # plt.ylabel('Velocity [m/s]')
# # plt.savefig("inletvel.png",format='png')




## Potential approach
## 1) identify campsites and number of ensemble runs
## 2) Collate data from each ensemble run into a single dataframe
## 3) Calculate data/statistics around the collated dataframe

outdir = "sample_flee_output"
location = "Fassala-Mbera"
sim_header = f"{location} sim"
data_header= f"{location} data"

#Read first results file header for variable names
df = pd.read_csv(f"{outdir}/1/out.csv")
headers = list(df)
numCamps = int((df.shape[1]-8)/3)

ensembleSize = 0

for n in range(numCamps):
    ensembleSize = 0
    dfTest = []
    # loop through each ensemble job extracting sim data and assigning to df for each campsite
    for name in os.listdir(outdir):
        df = pd.read_csv(f"{outdir}/{name}/out.csv")
        dfTest.append(df.iloc[:, 3*n+2].T)
        ensembleSize += 1
    
    #dfTest = pd.DataFrame(dfTest)

    # compute stats for each campsite (e.g. mean/max/min/SD/quartiles)
    
    # Plotting
    #df = pd.read_csv('out.csv')
    
    # plot all waveforms
    plt.figure(n+1)
    
    for i in range(ensembleSize):
        plt.plot(dfTest[i],'k', alpha=0.2)
    plt.plot(np.mean(dfTest,axis=0),'maroon',label='ensemble mean')
    plt.plot(df.iloc[:,3*n+3],'b-', label='UN Data')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('# of asylum seekers or unrecognized refugees')
    plt.title(str(headers[3*n+3]))

# plt.plot(df['Day'], df['Fassala-Mbera sim'],'k-')
# plt.plot(df['Day'], df['Fassala-Mbera data'],'b-')
# plt.xlabel('Day')
# plt.ylabel('Number')


#plot mean against quartile range for uncertainty
plt.show()



