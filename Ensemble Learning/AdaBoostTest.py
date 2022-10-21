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

print("iteration 1")
model = AdaBoost.AdaBoost()
print("train tree")
model.train(trainData, 1)
print("test model")
result = model.test(trainData)
acc = (trainData[trainData.columns[len(trainData.columns)-1]] == result)
acc = sum(acc)/len(acc)
err = 1-acc
print("training error: "+str(err))
trainErr.append(err)

tr = model.test(trainData)
ta = (testData[testData.columns[len(testData.columns)-1]] == result)
ta = sum(ta)/len(ta)
terr = 1-ta
print("test error: "+str(terr))
testErr.append(terr)
for i in range(2,t+1):
    print("iteration "+str(i))
    print("train tree")
    model.step(trainData)
    print("test model")
    if i == t:
        result = model.test(trainData,True)
        stumpTrain = model.getTreeError().copy()
    else:
        result = model.test(trainData)
    acc = (trainData[trainData.columns[len(trainData.columns)-1]] == result)
    acc = sum(acc)/len(acc)
    err = 1-acc
    print("training error: "+str(err))
    trainErr.append(err)

    if i == t:
        tr = model.test(testData,True)
    else:
        tr = model.test(testData)
    ta = (testData[testData.columns[len(testData.columns)-1]] == result)
    ta = sum(ta)/len(ta)
    terr = 1-ta
    print("test error: "+str(terr))
    testErr.append(terr)

    
plt.figure("AdaBoost Train and Test Errors")
plt.plot(range(1,t+1),trainErr,label="Training Error")
plt.plot(range(1,t+1),testErr,label = "Testing Error")
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.show()

stumpTest = model.getTreeError()
plt.figure("Stump Test Errors")
plt.scatter(range(1,t+1),stumpTrain,label="Training Error")
plt.scatter(range(1,t+1),stumpTest,label="Testing Error")
plt.ylabel("Error")
plt.xlabel("Tree #")
plt.show()