# -*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np
import InformationGain

class AdaBoost(object):
    
    def __init__(self):
        self.weights = None
        self.alph = []
        self.trees = []
        self.error = []
    
    def train(self, data, t):
        self.weights = pd.Series(1/data.shape[0], index=range(data.shape[0]))
        
        for i in range(t):
            dat = pd.concat([data,self.weights],axis=1,ignore_index=True)
            tree = InformationGain.InfoGain()
            tree.train(dat,depth=1,b=True,w=True)
            
            results = tree.test(data,b=True,w=True)

            accuracy = (data[data.columns[len(data.columns)-1]] == results)
            acc = sum(dat[dat.columns[len(dat.columns)-1]]*accuracy)
            err = 1 - acc
            a = 0.5 * math.log((acc/err))
            self.alph.append(a)
            accuracy[accuracy==0]=-1
            ac = -a * accuracy
            w = self.weights*np.exp(ac.astype(float))
            w = w/sum(w)
            self.weights = w
            self.trees.append(tree)
    
    def test(self, data):
        self.error.clear()
        i=0
        result = pd.Series(0.0,index=range(data.shape[0]))
        r = pd.Series(index=range(data.shape[0]),dtype=str)
        for tree in self.trees:
            res = tree.test(data,b=True,w=True)
            acc = (data[data.columns[len(data.columns)-1]] == res)
            acc = sum(acc)/len(acc)
            err = 1 - acc
            self.error.append(err)
            res = pd.Series(res)
            res[res=='no']=-1.0
            res[res=='yes']=1.0
            result = result.add(res.mul(self.alph[i]))
            i+=1
        
        r.loc[result<0] = 'no'
        r.loc[result>=0] = 'yes'
        
        return r.tolist()
    
    def getTreeError(self):
        return self.error
    
    def step(self,data):
        dat = pd.concat([data,self.weights],axis=1,ignore_index=True)
        tree = InformationGain.InfoGain()
        tree.train(dat,depth=1,b=True,w=True)
        
        results = tree.test(data,b=True,w=True)

        accuracy = (data[data.columns[len(data.columns)-1]] == results)
        acc = sum(dat[dat.columns[len(dat.columns)-1]]*accuracy)
        err = 1 - acc
        a = 0.5 * math.log((acc/err))
        self.alph.append(a)
        accuracy[accuracy==0]=-1
        ac = -a * accuracy
        w = self.weights*np.exp(ac.astype(float))
        w = w/sum(w)
        self.weights = w
        self.trees.append(tree)