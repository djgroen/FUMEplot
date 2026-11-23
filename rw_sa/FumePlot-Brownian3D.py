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
# e.g.: python3 FumePlot-Brownian3D.py params.json
# Plotting is skipped automatically when a params file is given, since a
# campaign run has no display and shouldn't block on a GUI window.
params = {
    "numParticles": 100,
    "massRange": 4,  # integer, particle masses distributed at integer values in [1, massRange); if 1, all particles are mass=1.
    "numTrials": 3,
    "numSteps": 100,
    "outputRate": 10,
    "flowVelocityX": 0,
    "flowVelocityY": 1,
    "flowVelocityZ": 0,
    "brownianForce": 0.3,
    "fileBase": "Brown3D",
}

plotting = len(sys.argv) == 1
if not plotting:
    with open(sys.argv[1]) as f:
        params.update(json.load(f))

numParticles = params["numParticles"]
massRange = params["massRange"]
numTrials = params["numTrials"]
numSteps = params["numSteps"]
outputRate = params["outputRate"]
flowVelocityX = params["flowVelocityX"]
flowVelocityY = params["flowVelocityY"]
flowVelocityZ = params["flowVelocityZ"]
brownianForce = params["brownianForce"]
fileBase = params["fileBase"]

if plotting:
    import matplotlib.pyplot as plt

# Output file written in format of a LAMMPS trajectory file for trial T and step S, viz: https://epcced.github.io/archer2-intro-to-lammps/05-advanced-input-and-output-commands/index.html
# and https://docs.lammps.org/dump.html
# Base information on timestep, number of atoms and then for each particle: particleID, type, x, y, z, vx, vy, vz
# In this case all particles are type=1
# This would correspond to a LAMMPS dump command of:
#    dump            1 all custom 100 filename.lammpstrj id type x y z vx vy vz
#    dump_modify     1 sort id

for t in range(1,numTrials+1):
    positionX = np.zeros(numParticles)
    positionY = np.zeros(numParticles)
    positionZ = np.zeros(numParticles)
    
    velocityX = flowVelocityX*np.ones(numParticles)
    velocityY = flowVelocityY*np.ones(numParticles)
    velocityZ = flowVelocityZ*np.ones(numParticles)
    
    if massRange == 1:
        mass = np.ones(numParticles)
    else:
        mass = np.random.randint(low=1,high=massRange,size=numParticles)
        
    for i in range(1,numSteps+1):
        accelX = brownianForce*(2*np.random.rand(numParticles)-1)/mass
        accelY = brownianForce*(2*np.random.rand(numParticles)-1)/mass
        accelZ = brownianForce*(2*np.random.rand(numParticles)-1)/mass
        
        velocityX += accelX
        velocityY += accelY
        velocityZ += accelZ 
        
        positionX += velocityX - 0.5*accelX
        positionY += velocityY - 0.5*accelY
        positionZ += velocityZ - 0.5*accelZ

        if i % outputRate == 0:
            filename = fileBase + '-T'+str(t)+'-S'+str(i)+'.txt' #T=trial number, S=step number
            f = open(filename,'w')
            contents = 'ITEM: TIMESTEP\n' + str(i) + '\n'
            contents += 'ITEM: NUMBER OF ATOMS\n' + str(numParticles) + '\n'
            contents += 'ITEM: BOX BOUNDS ss ss ss\n' + str(min(positionX)) + ' ' + str(max(positionX)) + '\n'
            contents += str(min(positionY)) + ' ' + str(max(positionY)) + '\n' + str(min(positionZ)) + ' ' + str(max(positionZ)) + '\n'
            contents += 'ITEM: ATOMS id type x y z vx vy vz\n'
            for p in range(numParticles):
                contents += str(p+1) + ' 1 ' + str(positionX[p]) + ' ' + str(positionY[p]) + ' ' + str(positionZ[p]) + ' '  \
                + str(velocityX[p]) + ' ' + str(velocityY[p]) + ' ' + str(velocityZ[p]) +'\n'

            f.write(contents)
            f.close()

    if plotting:
        plt.figure(1)        
        plt.plot(positionX,positionY,'.')
        plt.axis('equal')
        
        plt.figure(2)        
        plt.plot(positionX,positionZ,'.')
        plt.axis('equal')
        
        plt.figure(3)        
        plt.plot(positionY,positionZ,'.')
        plt.axis('equal')
        
        distance = np.sqrt(positionX**2 + positionY**2 + positionZ**2)
        plt.figure(4)
        plt.hist(distance, bins=20)

if plotting:
    plt.show()
