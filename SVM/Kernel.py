# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:29:02 2022

@author: ryanj
"""

import numpy as np
import scipy as sp

class SVM(object):
    
    def __init__(self):
        self.weights = None
        self.bias = 0
        self.g = 0
        self.x = None
        self.y = None
        self.a = None
    
    def train(self,data,C,gamma):
        if self.weights is None:
            self.weights = np.zeros((data.shape[1]-1,1))
        
        self.g = gamma
        x = data[:,0:data.shape[1]-1]
        self.x = x.copy()
        y = data[:,-1:]
        y = np.reshape(np.sign(y-0.5),(len(y),1))
        self.y = y.copy()
        
        fun = lambda a: 0.5 * np.matmul(np.matmul(a.T,np.matmul(y,y.T)*self.kernel(x,x)),a)-np.sum(a)
        cons = ({'type':'eq','fun': lambda a: np.sum(a*y)})
        start = 0.1*np.ones((data.shape[0],1))
        bnds = sp.optimize.Bounds(0,C)
        
        res = sp.optimize.minimize(fun,start, method = 'SLSQP', bounds = bnds, constraints=cons,options={'disp':True})
        
        a = np.reshape(res.x.copy(),y.shape)
        self.a = a.copy()
        self.weights = np.sum(a*y*x,axis=0)
        mask = (a>0)
        self.bias = np.average(mask*(y-np.reshape(np.matmul(x,self.weights),y.shape)))
        
    def kernel(self,x,z):
        dist = np.broadcast_to(np.sum((z**2),axis=1,keepdims=True),(z.shape[0],x.shape[0])) - 2 * np.dot(z,x.T)+ np.broadcast_to(np.sum((x**2),1,keepdims=True),(x.shape[0],z.shape[0])).T
        return np.exp(-(dist/self.g))
                    
    def test(self,data):
        r = np.sum(self.a*self.y*self.kernel(data,self.x),axis=0)+self.bias
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
        self.g = 0
        self.x = None
        self.y = None
        self.a = None