# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:58:13 2022

@author: ryanj
"""

import numpy as np

class Perceptron(object):
    
    def __init__(self):
        self.weights = None
        self.counts = None
    
    def train(self,data,t,lr):
        if self.weights is None:
            self.weights = np.zeros((1,data.shape[1]-1))
        
        if self.counts is None:
            self.counts = np.zeros((1,1))
            
        m=0
        for i in range(t):
            np.random.shuffle(data)
            print("epoch: "+str(i))
            x = data[:,0:data.shape[1]-1]
            y = data[:,-1:]
            y = np.reshape(np.sign(y-0.5),(len(y),1))
            for j in range(data.shape[0]):
                r = np.dot(x[j,:],self.weights[m,:].T)
                if np.sign(r)!=y[j]:
                    self.weights = np.append(self.weights,np.reshape(lr * y[j] * x[j,:],(1,self.weights.shape[1])),axis=0)
                    m+=1
                    self.counts= np.append(self.counts,np.ones((1,1)),axis=0)
                else:
                    self.counts[m]+=1

    def test(self,data):
        print(self.weights.shape)
        r = np.dot(np.sign(np.dot(data,self.weights.T)),self.counts)
        r = r.reshape((data.shape[0],))
        r = r>0
        r = r.astype(int)
        return r
    
    def getWeights(self):
        return self.weights
    
    def getCounts(self):
        return self.counts
    
    def reset(self):
        self.weights = None
        self.counts = None