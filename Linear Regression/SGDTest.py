# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 02:01:13 2022

@author: ryanj
"""

import sys
import pandas as pd
import numpy as np
import SGD
import matplotlib.pyplot as plt

trainFile = sys.argv[1]
testFile = sys.argv[2]
t = int(sys.argv[3])

trainData = pd.read_csv(trainFile, sep=',', header = None).to_numpy()
testData = pd.read_csv(testFile, sep=',', header=None).to_numpy()
lr = 0.01
sgd = SGD.SGD()
prev = np.zeros((trainData.shape[1]-1,1))
sgd.train(trainData,lr,1)
curr = sgd.getWeights()

#print(np.linalg.norm(curr-prev,ord=2))

for i in range(2,t+1):
    sgd.step(trainData,lr)
    prev = curr
    curr = sgd.getWeights()
    #print(np.linalg.norm(curr-prev,ord=2))
    
cost = sgd.getCosts()

plt.figure("Cost vs Iteration")
plt.plot(range(1,t+1),cost)
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.show()

print(sgd.getWeights())
print(sgd.test(testData))

x = trainData[:,0:trainData.shape[1]-1].T
y = trainData[:,-1:]

w = np.dot(np.dot(np.linalg.inv(np.dot(x,x.T)),x),y)

print(w)