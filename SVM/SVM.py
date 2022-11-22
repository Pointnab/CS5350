# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:31:39 2022

@author: ryanj
"""

import sys
import pandas as pd
import numpy as np
import Primal
import Dual
import Kernel

trainFile = sys.argv[1]
testFile = sys.argv[2]
mode = sys.argv[3]
a = 0
if mode == 'P':
    a = int(sys.argv[4])
elif mode == 'K':
    g = float(sys.argv[4])

trainData = pd.read_csv(trainFile, sep=',', header = None).to_numpy()
testData = pd.read_csv(testFile, sep=',', header = None).to_numpy()
xTrain = trainData[:,0:trainData.shape[1]-1]
xTest = testData[:,0:testData.shape[1]-1]

if mode == 'P':
    model = Primal.SVM()
elif mode == 'D':
    model = Dual.SVM()
elif mode == 'K':
    model = Kernel.SVM()
else:
    exit(0)

for C in {100/873, 500/873, 700/873}:
    print("C: ",C)
    model.reset()
    if mode == 'P':
        model.train(trainData,100,0.01,C,a)
    elif mode == 'D':
        model.train(trainData,C)
    elif mode == 'K':
        model.train(trainData,C,g)

    train_res = model.test(xTrain)
    train_acc = (trainData[:,-1] == train_res)
    print("Training Error: ", 1 - sum(train_acc)/len(train_acc))
    
    results = model.test(xTest)
    acc = (testData[:,-1] == results)
    print("Test Error: ", 1 - sum(acc)/len(acc))
    
    print("Weights: ",model.getWeights())
    if mode == 'D':
        print("Bias: ",model.getBias())