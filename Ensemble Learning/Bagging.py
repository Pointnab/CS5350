# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:09:19 2022

@author: ryanj
"""

# -*- coding: utf-8 -*-

import pandas as pd
import InformationGain

class Bagging(object):
    
    def __init__(self):
        self.trees = []
    
    def train(self, data, t, m):        
        for i in range(t):
            tree = InformationGain.InfoGain()
            dat = data.sample(m,axis=0,replace=True)
            tree.train(dat,depth=len(dat.columns)-1,b=True)
            self.trees.append(tree)
    
    def test(self, data):
        result = pd.Series(0.0,index=range(data.shape[0]))
        r = pd.Series(index=range(data.shape[0]),dtype=str)
        for tree in self.trees:
            res = tree.test(data,b=True)
            res = pd.Series(res)
            res[res=='no']=-1.0
            res[res=='yes']=1.0
            result = result.add(res.astype(float))
        
        r.loc[result<0] = 'no'
        r.loc[result>=0] = 'yes'
        
        return r.tolist()
    
    def step(self,data,m):
        tree = InformationGain.InfoGain()
        dat = data.sample(m,axis=0,replace=True)
        tree.train(dat,depth=len(dat.columns)-1,b=True)
        self.trees.append(tree)
        
    def getTrees(self):
        return self.trees