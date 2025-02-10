import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

#Determine number of ensemble values
ensembleSize = 10

#Read first results file header for variable names
df = pd.read_csv('1/out.csv')
numCamps = int((df.shape[1]-8)/3)

dfTest = []
# loop through each ensemble job extracting sim data and assigning to df for each campsite
for i in range(1,ensembleSize+1):
    df = pd.read_csv(str(i)+'/out.csv')
    dfTest.append(df.iloc[:, 2].T)

#dfTest = pd.DataFrame(dfTest)

# compute stats for each campsite (e.g. mean/max/min/SD/quartiles)

# Plotting
#df = pd.read_csv('out.csv')

# plot all waveforms
plt.figure(1)

for i in range(ensembleSize):
    plt.plot(dfTest[i],'k')
plt.plot(np.mean(dfTest,axis=0),'r-',label='ensemble mean')
plt.plot(df.iloc[:,3],'b-', label='UN Data')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Number of Refugees')

# plt.plot(df['Day'], df['Fassala-Mbera sim'],'k-')
# plt.plot(df['Day'], df['Fassala-Mbera data'],'b-')
# plt.xlabel('Day')
# plt.ylabel('Number')


#plot mean against quartile range for uncertainty
plt.show()



