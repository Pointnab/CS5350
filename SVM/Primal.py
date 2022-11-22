# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:29:02 2022

@author: ryanj
"""

import numpy as np

class SVM(object):
    
    def __init__(self):
        self.weights = None
        self.learning_rate = 0
    
    def train(self,data,epochs,lr,C,a = 0):
        if self.weights is None:
            self.weights = np.zeros((data.shape[1]-1,1))
        
        self.learning_rate = lr
        
        N = data.shape[0]
        t = 0
        
        for i in range(epochs):
            np.random.shuffle(data)
            x = data[:,0:data.shape[1]-1]
            y = data[:,-1:]
            y = np.reshape(np.sign(y-0.5),(len(y),1))
            for j in range(data.shape[0]):
                r = np.dot(x[j,:],self.weights)
                if r*y[j]<=1:
                    self.weights -= self.learning_rate * self.weights - np.reshape(self.learning_rate * y[j] * C * N * x[j,:],(self.weights.shape))
                else:
                    self.weights -= self.learning_rate * self.weights
                
                t += 1
                if a:
                    self.learning_rate = lr/(1+lr*t/a)
                else:
                    self.learning_rate = lr/(1+t)
                    
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