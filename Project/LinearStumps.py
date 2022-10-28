# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:56:59 2022

@author: ryanj
"""

import pandas as pd
import numpy as np
import InformationGain

class LinStump(object):
    
    def __init__(self):
        self.weights = None
        self.trees = []
        self.rand = None
    
    def learnTrees(self, data):
        for i in range(0,len(data.columns)-1):
            tree = InformationGain.InfoGain()
            dat = data.iloc[:,[i,len(data.columns)-1]]
            tree.train(dat,depth=1,b=True)
            self.trees.append(tree)
        
    def train(self, data, t, lr):
        if self.rand is None:
            self.rand = np.random
            
        if self.weights is None:
            self.weights = np.zeros((len(self.trees),1))
            
        x = None
        y = data.iloc[:,-1].to_numpy()
        y = np.reshape(np.sign(y-0.5),(len(y),1))
        i = 1
        
        for tree in self.trees:
            print("tree: "+str(i))
            i+=1
            if x is None:
                x = np.array(tree.test(data,b=True,p=True))
                x = np.reshape(x,(len(x),1))
            else:
                n = np.array(tree.test(data,b=True,p=True))
                n = np.reshape(n,(len(n),1))
                x = np.append(x,n,axis = 1)
              
        w = np.zeros((len(self.trees),1))
        for i in range(t):
            print("epoch: "+str(i))
            for j in range(len(x)):
                if j % 1000 == 0:
                    print("iteration: "+str(j))
                r = np.dot(x[j,:],w)
                if np.sign(r)!=y[j]:
                    w = w + np.reshape(lr * y[j] * x[j,:],(w.shape))
            self.weights += w
    
    def test(self, data):        
        x = None
            
        for tree in self.trees:
            if x is None:
                x = np.array(tree.test(data,b=True,p=True))
                x = np.reshape(x,(len(x),1))
            else:
                n = np.array(tree.test(data,b=True,p=True))
                n = np.reshape(n,(len(n),1))
                x = np.append(x,n,axis = 1)
        
        r = np.dot(x,self.weights)
        r = r.reshape((data.shape[0],))
        r = r>0
        r = r.astype(int)
        return r
    
    def getWeights(self):
        return self.weights