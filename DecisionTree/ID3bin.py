# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:33:43 2022

@author: Ryan Lam
"""

import sys
import pandas as pd
import InformationGain
import MajorityError
import GiniIndex

trainFile = sys.argv[1]
testFile = sys.argv[2]
method = sys.argv[3]
depth = int(sys.argv[4])
unk = bool(sys.argv[5])

trainData = pd.read_csv(trainFile, sep=',', header = None)
testData = pd.read_csv(testFile, sep=',', header=None)

if method == "H":
    tree = InformationGain.InfoGain()
    
elif method == "ME":
    tree = MajorityError.ME()

elif method == "GI":
    tree = GiniIndex.GI()
    
else:
    print("Invalid method")
    exit(0)

tree.train(trainData, depth, b=True, u=unk)
results = tree.test(testData, b=True, u=unk)

acc = (testData[testData.columns[len(testData.columns)-1]] == results)

print("Method=" + method + " Depth=" + str(depth) + " Accuracy=" + str(sum(acc)/len(acc)))