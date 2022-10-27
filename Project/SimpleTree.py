# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:33:43 2022

@author: Ryan Lam
"""

import sys
import pandas as pd
import InformationGain

trainFile = sys.argv[1]
testFile = sys.argv[2]

trainData = pd.read_csv(trainFile, sep=',', header = 0)
testData = pd.read_csv(testFile, sep=',', header = 0)

IDnum = testData[["ID"]]

tree = InformationGain.InfoGain()
print("Training")
tree.train(trainData, 14, b=True)

print("Validation")
val = tree.test(trainData, b= True)
acc = (trainData[trainData.columns[len(trainData.columns)-1]] == val)
print(str(sum(acc)/len(acc)))

print("Testing")
results = tree.test(testData, b = True)

results = pd.concat([testData[testData.columns[0]], pd.Series(results)],axis=1)
results.columns = ['ID','Prediction']

results.to_csv("predictions.csv",index=False)