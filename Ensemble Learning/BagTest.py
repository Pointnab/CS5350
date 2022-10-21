# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:17:41 2022

@author: ryanj
"""

import sys
import pandas as pd
import Bagging
import matplotlib.pyplot as plt

trainFile = sys.argv[1]
testFile = sys.argv[2]
t = int(sys.argv[3])
m = int(sys.argv[4])

trainData = pd.read_csv(trainFile, sep=',', header = None)
testData = pd.read_csv(testFile, sep=',', header=None)
trainErr = []
testErr = []
models = []

print("Trees 1")
model = Bagging.Bagging()
print("train model")
model.train(trainData, 1, m)
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
    print("Trees "+str(i))
    print("train model")
    model.step(trainData, m)
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
    
plt.figure("Bagging Train and Test Errors")
plt.plot(range(1,t+1),trainErr)
plt.plot(range(1,t+1),testErr)
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.show()