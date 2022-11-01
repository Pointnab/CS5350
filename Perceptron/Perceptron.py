# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:31:39 2022

@author: ryanj
"""

import sys
import pandas as pd
import numpy as np
import Standard

trainFile = sys.argv[1]
testFile = sys.argv[2]
mode = sys.argv[3]

trainData = pd.read_csv(trainFile, sep=',', header = 0).to_numpy()
testData = pd.read_csv(testFile, sep=',', header = 0).to_numpy()
xTest = testData[:,0:testData.shape[1]-1]

if mode == "S":
    model = Standard.Perceptron()

model.train(trainData,10,0.01)

results = model.test(xTest)

acc = (testData[:,-1] == results)

print("Learned Weight Vector:")
print(model.getWeights())
print("Accuracy:")
print(str(sum(acc)/len(acc)))