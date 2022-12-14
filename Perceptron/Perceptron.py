# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:31:39 2022

@author: ryanj
"""

import sys
import pandas as pd
import numpy as np
import Standard
import Voted
import Averaged

trainFile = sys.argv[1]
testFile = sys.argv[2]
mode = sys.argv[3]

trainData = pd.read_csv(trainFile, sep=',', header = None).to_numpy()
testData = pd.read_csv(testFile, sep=',', header = None).to_numpy()
xTest = testData[:,0:testData.shape[1]-1]
error = []

if mode == "S":
    model = Standard.Perceptron()
    
elif mode == "V":
    model = Voted.Perceptron()
    
elif mode == "A":
    model = Averaged.Perceptron()

else:
    exit(0)

for i in range(100):
    model.reset()
    model.train(trainData,10,0.01)

    results = model.test(xTest)

    acc = (testData[:,-1] == results)
    error.append(1 - sum(acc)/len(acc))

print("Learned Weight Vector:")
print(model.getWeights())
if mode == "V":
    print(model.getCounts())
print("Average Error:")
print(np.average(error))