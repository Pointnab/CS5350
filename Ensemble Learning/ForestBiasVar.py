# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 21:37:34 2022

@author: ryanj
"""
import sys
import pandas as pd
import Forest
import numpy as np

trainFile = sys.argv[1]
testFile = sys.argv[2]

trainData = pd.read_csv(trainFile, sep=',', header = None)
testData = pd.read_csv(testFile, sep=',', header=None)
models = []

for i in range(100):
    print("Generating model "+str(i))
    data = trainData.sample(1000,axis=0)
    mod = Forest.Forest()
    mod.train(data, 500, 100, 6)
    
    models.append(mod)
    
    
treeResult = pd.DataFrame()
modelResult = pd.DataFrame()
gt = testData.iloc[:,-1:]
gt[gt=='no']=-1.0
gt[gt=='yes']=1.0

i=0
for m in models:
    print("predicting with model " +str(i))
    tree = m.getTrees()[0]
    
    tRes = tree.test(testData)
    tRes = pd.Series(tRes)
    tRes[tRes=='no']=-1.0
    tRes[tRes=='yes']=1.0
    treeResult = pd.concat([treeResult, tRes],axis=1,ignore_index=True)
    mRes = m.test(testData)
    mRes = pd.Series(mRes)
    mRes[mRes=='no']=-1.0
    mRes[mRes=='yes']=1.0
    modelResult = pd.concat([modelResult, mRes],axis=1,ignore_index=True)
    i+=1

tr = treeResult.to_numpy()
mr = modelResult.to_numpy()
truth = gt.to_numpy()
tBias = np.square(tr.sum(axis=1,keepdims=True)/100.0 - truth)
mBias = np.square(mr.sum(axis=1,keepdims=True)/100.0 - truth)
print("tree bias = "+str(np.mean(tBias)))
print("model bias = "+str(np.mean(mBias)))
tVar = np.sum(np.square(tr - tr.sum(axis=1,keepdims=True)/100.0))/99
mVar = np.sum(np.square(mr - mr.sum(axis=1,keepdims=True)/100.0))/99
print("tree variance = "+str(np.mean(tVar)))
print("model variance = "+str(np.mean(mVar)))
print("tree error = "+str(np.mean(tBias)+np.mean(tVar)))
print("model error = "+str(np.mean(mBias)+np.mean(mVar)))