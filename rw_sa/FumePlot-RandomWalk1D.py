# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:31:02 2026

@author: 3059176
"""
import json
import sys

import numpy as np

# Defaults match the original hardcoded values. Pass a JSON file as the
# first command-line argument to override any of them for a campaign run,
# e.g.: python3 FumePlot-RandomWalk1D.py params.json
# Plotting is skipped automatically when a params file is given, since a
# campaign run has no display and shouldn't block on a GUI window.
params = {
    "numParticles": 500,
    "numTrials": 3,
    "numSteps": 100,
    "outputRate": 10,
    "flowVelocity": 0,
    "diffSpread": 2,
    "fileBase": "RW1D",
}

plotting = len(sys.argv) == 1
if not plotting:
    with open(sys.argv[1]) as f:
        params.update(json.load(f))

numParticles = params["numParticles"]
numTrials = params["numTrials"]
numSteps = params["numSteps"]
outputRate = params["outputRate"]
flowVelocity = params["flowVelocity"]
diffSpread = params["diffSpread"]
fileBase = params["fileBase"]

if plotting:
    import matplotlib.pyplot as plt

# Output file written in XYZ format for trial T and step S, viz: https://www.ks.uiuc.edu/Research/vmd/plugins/molfile/xyzplugin.html
# and https://epcced.github.io/archer2-intro-to-lammps/08-creating-topology/index.html
# First row is number of particles
# subsequent rows are: particleID x y z
# Here y and z are 0.0 as it is a 1D simulation 

for t in range(1,numTrials+1):
    position = np.zeros(numParticles)
        
    for i in range(1,numSteps+1):
        
        position = position + flowVelocity + diffSpread*(2*np.random.rand(numParticles)-1)

        if i % outputRate == 0:
            filename = fileBase + '-T'+str(t)+'-S'+str(i)+'.txt' #T=trial number, S=step number
            f = open(filename,'w')
            contents = str(numParticles) + '\n\n'
            for p in range(numParticles):
                contents += str(p+1) + ' ' + str(position[p]) + ' 0.0 0.0\n'

            f.write(contents)
            f.close()

    if plotting:
        plt.figure(1)
        plt.plot(position,'.')
        plt.figure(2)
        plt.hist(position, bins=20)

if plotting:
    plt.show()
