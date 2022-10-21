# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 01:58:05 2022

@author: ryanj
"""
import numpy as np

class SGD():
    
    def __init__(self):
        self.weights = None
        self.costs = []
        self.rand = None
        
    def train(self, data, lr, t):
        
        if self.rand is None:
            self.rand = np.random
        
        xc = data[:,0:data.shape[1]-1]
        yc = data[:,-1:]
        
        idx = int(self.rand.rand()*data.shape[0])
        x = data[idx,0:data.shape[1]-1]
        y = data[idx,-1:]
        
        if self.weights is None:
            self.weights = np.zeros((len(x),1))
        
        for i in range(t):
            dw = np.zeros(self.weights.shape)
            
            pred = np.dot(xc,self.weights)            
            cost = 0.5 * np.sum((yc-pred)**2)
            self.costs.append(cost)
            
            dw = -((y-np.dot(x,self.weights))*x)
            dw = dw.reshape((len(x),1))
            self.weights -= lr * dw
    
    def step(self, data, lr):
        
        xc = data[:,0:data.shape[1]-1]
        yc = data[:,-1:]
        
        idx = int(self.rand.rand()*data.shape[0])
        x = data[idx,0:data.shape[1]-1]
        y = data[idx,-1:]
        
        dw = np.zeros(self.weights.shape)
        
        pred = np.dot(xc,self.weights)            
        cost = 0.5 * np.sum((yc-pred)**2)
        self.costs.append(cost)
        
        dw = -((y-np.dot(x,self.weights))*x)
        dw = dw.reshape((len(x),1))
        self.weights -= lr * dw
        
    def test(self, data):
        x = data[:,0:data.shape[1]-1]
        y = data[:,-1:]
        
        pred = np.dot(x,self.weights)            
        cost = 0.5 * np.sum((y-pred)**2)
        
        return cost
    
    def getWeights(self):
        return self.weights
    
    def getCosts(self):
        return self.costs