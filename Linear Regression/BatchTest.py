# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 02:01:13 2022

@author: ryanj
"""

import sys
import pandas as pd
import numpy as np
import GD
import matplotlib.pyplot as plt

trainFile = sys.argv[1]
testFile = sys.argv[2]
t = int(sys.argv[3])

trainData = pd.read_csv(trainFile, sep=',', header = None).to_numpy()
testData = pd.read_csv(testFile, sep=',', header=None).to_numpy()
lr = 0.00005
gd = GD.GD()
prev = np.zeros((trainData.shape[1]-1,1))
gd.train(trainData,lr,1)
curr = gd.getWeights()

#print(np.linalg.norm(curr-prev,ord=2))

for i in range(2,t+1):
    gd.step(trainData,lr)
    prev = curr
    curr = gd.getWeights()
    #print(np.linalg.norm(curr-prev,ord=2))
    
cost = gd.getCosts()

plt.figure("Cost vs Iteration")
plt.plot(range(1,t+1),cost)
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.show()

print(gd.getWeights())
print(gd.test(testData))