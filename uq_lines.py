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

#Read first results file header for variable names

# loop through each ensemble job extracting sim data and assigning to df for each campsite

# compute stats for each campsite (e.g. mean/max/min/SD/quartiles)

outdir = "sample_flee_output"
location = "Fassala-Mbera"
sim_header = f"{location} sim"
data_header= f"{location} data"

sim_results = []

for i in range(1,11):

    # Plotting
    df = pd.read_csv(f"{outdir}/{i}/out.csv")

    # plot all waveforms
    plt.figure(1)
    plt.plot(df['Day'], df[sim_header],'k-', alpha=0.1)
    sim_results.append(df[sim_header])

    if i == 10:
        plt.plot(df['Day'], df[data_header],'b-')

plt.xlabel('Day')
plt.ylabel('Number')
print(sim_results)

#plot mean against quartile range for uncertainty
plt.show()
