# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 19:01:02 2022

@author: ryanj
"""

import torch
import numpy as np

class ThreeLayer(object):
    
    def __init__(self,in_size,l1_size,l2_size,out_size):
        self.weights1 = torch.zeros((in_size+1,l1_size)).to(dtype=torch.double)
        self.weights2 = torch.zeros((l1_size+1,l2_size)).to(dtype=torch.double)
        self.weights3 = torch.zeros((l2_size+1,out_size)).to(dtype=torch.double)
        self.learning_rate = 0.001
        
    def forward(self, x):
        N = x.shape[0]
        xa = torch.cat((torch.ones((N,1)),x),1)
        
        s1 = torch.matmul(xa,self.weights1)
        z1 = torch.sigmoid(s1)
        l1 = torch.cat((torch.ones((N,1)),z1),1)
        
        s2 = torch.matmul(l1,self.weights2)
        z2 = torch.sigmoid(s2)
        l2 = torch.cat((torch.ones((N,1)),z2),1)
        
        y = torch.matmul(l2,self.weights3)
        
        cache = (xa,s1,z1,l1,s2,z2,l2)
        
        return y, cache
    
    def train(self, data, lr, epochs, d, gauss):
        
        if gauss:
            self.weights1 += torch.normal(0,1,self.weights1.shape)
            self.weights2 += torch.normal(0,1,self.weights2.shape)
            self.weights3 += torch.normal(0,1,self.weights3.shape)
        
        self.learning_rate = lr
                
        N = data.shape[0]
        t = 0
        
        for i in range(epochs):
            np.random.shuffle(data)
            x = torch.tensor(data[:,0:data.shape[1]-1])
            y = data[:,-1:]
            y = np.reshape(np.sign(y-0.5),(len(y),1))
            for j in range(N):
                row = x[j,:]
                r,c = self.forward(row[None,:])
                dL = r - y[j]
                dw3, dw2, dw1 = self.backprop(dL, c)
                
                self.weights3 -= self.learning_rate * dw3
                self.weights2 -= self.learning_rate * dw2
                self.weights1 -= self.learning_rate * dw1
                
                t += 1
                self.learning_rate = lr/(1+lr*t/d)
        
    def backprop(self, dL, cache):
    
        x,s1,z1,l1,s2,z2,l2 = cache
        
        dw3 = dL * l2.T
        dz2 = (dL * self.weights3[1:,:]).T
        
        ds2 = dz2 * z2*(1-z2)
        
        dw2 = torch.matmul(l1.T,ds2)
        dz1 = torch.matmul(ds2,self.weights2.T)[:,1:]
        
        ds1 = dz1 * z1*(1-z1)
        
        dw1 = torch.matmul(x.T,ds1)
        
        return dw3, dw2, dw1