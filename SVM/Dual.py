# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:29:02 2022

@author: ryanj
"""

import numpy as np
from scipy import optimize

class SVM(object):
    
    def __init__(self):
        self.weights = None
        self.bias = 0
    
    def train(self,data,C):
        if self.weights is None:
            self.weights = np.zeros((data.shape[1]-1,1))
        
        x = data[:,0:data.shape[1]-1]
        y = data[:,-1:]
        y = np.reshape(np.sign(y-0.5),(len(y),1))
        
        fun = lambda a: 0.5 * np.matmul(np.matmul(a.T,np.matmul(y,y.T)*np.matmul(x,x.T)),a)-np.sum(a)
        cons = ({'type':'eq','fun': lambda a: np.sum(a*y)})
        start = 0.1*np.ones((data.shape[0],1))
        bnds = optimize.Bounds(0,C)
        
        res = optimize.minimize(fun,start, method = 'SLSQP', bounds = bnds, constraints=cons,options={'disp':True})
        
        a = np.reshape(res.x.copy(),y.shape)
        self.weights = np.sum(a*y*x,axis=0)
        mask = (a>0)
        self.bias = np.average(mask*(y-np.reshape(np.matmul(x,self.weights),y.shape)))
                    
    def test(self,data):
        r = np.dot(data,self.weights)+self.bias
        r = r.reshape((data.shape[0],))
        r = r>0
        r = r.astype(int)
        return r
    
    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias
    
    def reset(self):
        self.weights = None
        self.bias = 0