# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:05:20 2022

@author: ryanj
"""

import sys
import pandas as pd
import torch
import NN

trainFile = sys.argv[1]
testFile = sys.argv[2]
gauss = int(sys.argv[3])

trainData = pd.read_csv(trainFile, sep=',', header = None).to_numpy()
testData = pd.read_csv(testFile, sep=',', header = None).to_numpy()
xTrain = torch.tensor(trainData[:,0:trainData.shape[1]-1])
xTest = torch.tensor(testData[:,0:testData.shape[1]-1])

for w in [5,10,25,50,100]:
    print("Layer Size: ",w)
    model = NN.ThreeLayer(4, w, w, 1)
    
    model.train(trainData,1e-3,100,5,gauss)
    
    train_res, _ = model.forward(xTrain)
    train_res = (train_res>0).numpy().reshape((trainData.shape[0]))
    train_acc = (trainData[:,-1] == train_res)
    print("Training Error: ", 1 - sum(train_acc)/len(train_acc))
    
    results, _ = model.forward(xTest)
    results = (results > 0).numpy().reshape((testData.shape[0]))
    acc = (testData[:,-1] == results)
    print("Test Error: ", 1 - sum(acc)/len(acc))