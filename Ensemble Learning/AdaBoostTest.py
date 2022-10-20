# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:31:57 2022

@author: ryanj
"""

import sys
import pandas as pd
import AdaBoost
import matplotlib.pyplot as plt

trainFile = sys.argv[1]
testFile = sys.argv[2]
t = int(sys.argv[3])

trainData = pd.read_csv(trainFile, sep=',', header = None)
testData = pd.read_csv(testFile, sep=',', header=None)
trainErr = []
testErr = []
stumpTestErr = {}

print("iteration 1")
model = AdaBoost.AdaBoost()
model.train(trainData, 1)
result = model.test(trainData)
acc = (trainData[trainData.columns[len(trainData.columns)-1]] == result)
acc = sum(acc)/len(acc)
err = 1-acc
trainErr.append(err)

tr = model.test(trainData)
ta = (testData[testData.columns[len(testData.columns)-1]] == result)
ta = sum(ta)/len(ta)
terr = 1-ta
testErr.append(terr)
stumpTestErr.update({1:model.getTreeError()})
for i in range(2,t+1):
    print("iteration "+str(i))
    model.step(trainData)
    
    result = model.test(trainData)
    acc = (trainData[trainData.columns[len(trainData.columns)-1]] == result)
    acc = sum(acc)/len(acc)
    err = 1-acc
    trainErr.append(err)

    tr = model.test(trainData)
    ta = (testData[testData.columns[len(testData.columns)-1]] == result)
    ta = sum(ta)/len(ta)
    terr = 1-ta
    testErr.append(terr)
    stumpTestErr.update({i:model.getTreeError()})
    
plt.figure("AdaBoost Train and Test Errors")
plt.plot(range(1,t+1),trainErr)
plt.plot(range(1,t+1),testErr)
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.show()

i=1
plt.figure("Stump Test Errors")
for error in stumpTestErr.values():
    plt.scatter([i]*len(error),error)
    i+=1
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.show()