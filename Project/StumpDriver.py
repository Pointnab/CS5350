# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:55:54 2022

@author: ryanj
"""

import sys
import pandas as pd
import LinearStumps as ls

trainFile = sys.argv[1]
testFile = sys.argv[2]

trainData = pd.read_csv(trainFile, sep=',', header = 0)
testData = pd.read_csv(testFile, sep=',', header = 0)
IDnum = testData[["ID"]]

model = ls.LinStump()
print("Building Trees")
model.learnTrees(trainData)
print("Training")
model.train(trainData,3,1)
print("Testing")
results = model.test(testData)

results = pd.concat([testData[testData.columns[0]], pd.Series(results)],axis=1)
results.columns = ['ID','Prediction']

results.to_csv("predictions2.csv",index=False)