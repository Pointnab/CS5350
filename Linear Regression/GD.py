# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 01:58:05 2022

@author: ryanj
"""
import numpy as np

class GD():
    
    def __init__(self):
        self.weights = None
        self.costs = []
        
    def train(self, data, lr, t):
        
        x = data[:,0:data.shape[1]-1]
        y = data[:,-1:]
        
        if self.weights is None:
            self.weights = np.zeros((x.shape[1],1))
        
        for i in range(t):
            dw = np.zeros(self.weights.shape)
            
            pred = np.dot(x,self.weights)
            cost = 0.5 * np.sum((y-pred)**2)
            self.costs.append(cost)
            
            dw = -np.sum(((y-pred)*x),axis=0,keepdims=True)
            self.weights -= lr * dw.T
    
    def step(self, data, lr):
        
        x = data[:,0:data.shape[1]-1]
        y = data[:,-1:]
        
        dw = np.zeros(self.weights.shape)
        
        pred = np.dot(x,self.weights)
        cost = 0.5 * np.sum((y-pred)**2)
        self.costs.append(cost)
        
        dw = -np.sum(((y-pred)*x),axis=0,keepdims=True)
        self.weights -= lr * dw.T
        
    def test(self, data):
        x = data[:,0:data.shape[1]-1]
        y = data[:,-1:]
        
        pred = np.dot(x,self.weights)            
        cost = 0.5 * np.sum((y-pred)**2,axis=0,keepdims=True)
        
        return cost
    
    def getWeights(self):
        return self.weights
    
    def getCosts(self):
        return self.costs