# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:31:02 2026

@author: 3059176

Config-driven version (same pattern as rw_sa/ PR #18): optionally takes one
CLI arg, a path to a JSON params file. Any key present there overrides the
matching hardcoded constant below; keys not present keep their default.
With no CLI arg, behavior is unchanged from Jon's original script, byte for
byte (including the plots). With a params file given, matplotlib is never
imported and no plots are produced, since FabSim3 runs have no display.
"""
import sys
import json

import numpy as np

numParticles = 100
massRange = 4  # integer, Particle masses distributed at integer values between [1,massRange), if set to 1 then all particles are mass=1.

numTrials = 3
numSteps = 100
outputRate = 10

flowVelocityX = 0
flowVelocityY = 1
flowVelocityZ = 0

fluidDrag = 0.01

fileBase = 'Brown3D-Drag'

_PARAMS_FILE = sys.argv[1] if len(sys.argv) > 1 else None

if _PARAMS_FILE:
    with open(_PARAMS_FILE) as _f:
        _overrides = json.load(_f)
    for _key, _value in _overrides.items():
        globals()[_key] = _value

# brownianForce is derived from fluidDrag, so it must be recomputed AFTER
# any override, not hardcoded alongside it.
brownianForce = np.sqrt(2.0 * fluidDrag)  # viz https://en.wikipedia.org/wiki/Brownian_dynamics

if not _PARAMS_FILE:
    import matplotlib.pyplot as plt

# Output file written in format of a LAMMPS trajectory file for trial T and step S, viz: https://epcced.github.io/archer2-intro-to-lammps/05-advanced-input-and-output-commands/index.html
# and https://docs.lammps.org/dump.html
# Base information on timestep, number of atoms and then for each particle: particleID, type, x, y, z, vx, vy, vz
# In this case all particles are type=1
# This would correspond to a LAMMPS dump command of:
#    dump            1 all custom 100 filename.lammpstrj id type x y z vx vy vz
#    dump_modify     1 sort id

for t in range(1, numTrials + 1):
    positionX = np.zeros(numParticles)
    positionY = np.zeros(numParticles)
    positionZ = np.zeros(numParticles)

    velocityX = flowVelocityX * np.ones(numParticles)
    velocityY = flowVelocityY * np.ones(numParticles)
    velocityZ = flowVelocityZ * np.ones(numParticles)

    if massRange == 1:
        mass = np.ones(numParticles)
    else:
        mass = np.random.randint(low=1, high=massRange, size=numParticles)

    for i in range(1, numSteps + 1):

        accelX = (np.random.normal(0.0, brownianForce, numParticles) - fluidDrag * (velocityX - flowVelocityX)) / mass
        accelY = (np.random.normal(0.0, brownianForce, numParticles) - fluidDrag * (velocityY - flowVelocityY)) / mass
        accelZ = (np.random.normal(0.0, brownianForce, numParticles) - fluidDrag * (velocityZ - flowVelocityZ)) / mass

        velocityX += accelX
        velocityY += accelY
        velocityZ += accelZ

        positionX += velocityX - 0.5 * accelX
        positionY += velocityY - 0.5 * accelY
        positionZ += velocityZ - 0.5 * accelZ

        if i % outputRate == 0:
            filename = fileBase + '-T' + str(t) + '-S' + str(i) + '.txt'  # T=trial number, S=step number
            f = open(filename, 'w')
            contents = 'ITEM: TIMESTEP\n' + str(i) + '\n'
            contents += 'ITEM: NUMBER OF ATOMS\n' + str(numParticles) + '\n'
            contents += 'ITEM: BOX BOUNDS ss ss ss\n' + str(min(positionX)) + ' ' + str(max(positionX)) + '\n'
            contents += str(min(positionY)) + ' ' + str(max(positionY)) + '\n' + str(min(positionZ)) + ' ' + str(max(positionZ)) + '\n'
            contents += 'ITEM: ATOMS id type x y z vx vy vz\n'
            for p in range(numParticles):
                contents += str(p + 1) + ' 1 ' + str(positionX[p]) + ' ' + str(positionY[p]) + ' ' + str(positionZ[p]) + ' '  \
                    + str(velocityX[p]) + ' ' + str(velocityY[p]) + ' ' + str(velocityZ[p]) + '\n'

            f.write(contents)
            f.close()

    if not _PARAMS_FILE:
        plt.figure(1)
        plt.plot(positionX, positionY, '.')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')

        plt.figure(2)
        plt.plot(positionX, positionZ, '.')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.axis('equal')

        plt.figure(3)
        plt.plot(positionY, positionZ, '.')
        plt.xlabel('Y')
        plt.ylabel('Z')
        plt.axis('equal')

        distance = np.sqrt(positionX ** 2 + positionY ** 2 + positionZ ** 2)
        plt.figure(4)
        plt.hist(distance, bins=20)
        plt.xlabel('Distance')
        plt.ylabel('Count')

if not _PARAMS_FILE:
    plt.show()
