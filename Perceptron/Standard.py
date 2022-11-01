# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:29:02 2022

@author: ryanj
"""

import numpy as np

class Perceptron(object):
    
    def __init__(self):
        self.weights = None
    
    def train(self,data,t,lr):
        if self.weights is None:
            self.weights = np.zeros((data.shape[1]-1,1))
        
        for i in range(t):
            np.random.shuffle(data)
            print("epoch: "+str(i))
            x = data[:,0:data.shape[1]-1]
            y = data[:,-1:]
            y = np.reshape(np.sign(y-0.5),(len(y),1))
            for j in range(data.shape[0]):
                r = np.dot(x[j,:],self.weights)
                if np.sign(r)!=y[j]:
                    self.weights += np.reshape(lr * y[j] * x[j,:],(self.weights.shape))

    def test(self,data):
        r = np.dot(data,self.weights)
        r = r.reshape((data.shape[0],))
        r = r>0
        r = r.astype(int)
        return r
    
    def getWeights(self):
        return self.weights
    
    def reset(self):
        self.weights = None