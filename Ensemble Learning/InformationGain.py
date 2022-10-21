# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:56:19 2022

@author: Ryan Lam
"""

import Node
import pandas as pd
import math

class InfoGain(object):
    
    def __init__(self):
        self.tree = None
        self.thresholds = {}
        self.majority = {}
        
    def train(self, data, depth = 6, b=False, u=False, w=False, f=-1):
        if self.tree is None:
            self.tree = Node.Node()
            
        for attribute in data.columns:
            if b:    
                if data[attribute].dtype == "int64":
                    thresh = data[attribute].median()
                    self.thresholds.update({attribute:thresh})
                    data.loc[data[attribute]>=thresh,attribute] = thresh
                    data.loc[data[attribute]<thresh,attribute] = thresh-1
        
            if u:
                vals = data.value_counts([attribute])
                maj = vals.index.levels[0][0]
                if maj == "unknown":
                    maj = vals.index.levels[0][1]
                self.majority.update({attribute:maj})
                data.loc[data[attribute]=="unknown",attribute] = maj
        
        self.tree.addMembers(data)
        
        self.splitMembers(self.tree, depth, w, f)
        
        #self.tree.display()
        
        return
    
    def splitMembers(self, node, depth, w, f):
        members = node.getMembers()     
        
        if w:
            lab = members.columns[len(members.columns)-2]
        else:
            lab = members.columns[len(members.columns)-1]
            
        if node.depth >= depth:
            node.setLeaf()
            if w:
                labels = members.groupby([lab],sort=False)[lab+1].sum()
                labels = labels.sort_values(ascending=False)
                name = labels.index[0]
            else:
                labels = members.value_counts([lab])
                name = labels.index[0][0]
            node.setName(name)
            return
        
        if len(members[lab].unique()) == 1:
            node.setLeaf()
            node.setName(members[lab][0])
            return
        
        if w:
            base = members.groupby(lab)[lab+1].sum()
        else:
            base = members.value_counts([lab],normalize=True)
        
        initialEntropy = 0
        for n in base:
            initialEntropy -= n*math.log(n)
        
        gain = {}
        
        if f>0 and f<len(members.columns)-1:
            mem = members.iloc[:,0:len(members.columns)-1]
            mem = mem.sample(f,axis=1)
            members = pd.concat([mem , members.iloc[:,-1:]],axis=1,ignore_index=False)
            
        for attribute in members.columns:
            if attribute >= lab:
                continue
            
            if w:
                vals = members.groupby([attribute,lab])[lab+1].sum()
            else:
                vals = members.value_counts([attribute,lab])
            
            ent = 0
            for label in vals.index.levels[0]:
                cat = vals.loc[label,:]
                cat2 = cat/sum(cat)
                h = 0
                for x in cat2:
                    h -= x*math.log(x)
                ent += h * sum(cat)/sum(vals)
            g = initialEntropy - ent
            gain.update({attribute:g})
        
        best = max(gain, key=gain.get)
        
        if f>0:
            members = node.getMembers()
        
        for l in members[best].unique():
            mem = members[members[best] == l]
            mem = mem.drop(best, axis=1)
            node.setAttribute(best)
            node.addChild(l,mem,node.depth+1)
        
        for child in node.children:
            self.splitMembers(child,depth,w,f)
            
    def test(self, data, b=False, u=False, w=False):
        pred = []
        
        if b:
            for attribute in self.thresholds:
                thresh = self.thresholds[attribute]
                data.loc[data[attribute]>=thresh,attribute] = thresh
                data.loc[data[attribute]<thresh,attribute] = thresh-1
        
        for idx, point in data.iterrows():
            
            if u:
                for attribute in self.majority:
                    maj = self.majority[attribute]
                    if point[attribute]=="unknown":
                        point[attribute] = maj
            
            pred.append(self.classify(point, w))
        
        return pred
    
    def classify(self, data, w):
        current = self.tree
        
        while True:
            end = False
            members = current.getMembers()
            if w:
                lab = members.columns[len(members.columns)-2]
            else:
                lab = members.columns[len(members.columns)-1]
            
            if current.isLeaf:
                if len(members[lab].unique()) == 1:
                    if w:
                        return members.iloc[0,members.shape[1]-2]
                    else:
                        return members.iloc[0,members.shape[1]-1]
                
                if w:
                    return current.getName()
                else:
                    labels = members.value_counts([lab])
                    return labels.index[0][0]
            
            attribute = current.getAttribute()
            for child in current.getChildren():
                if child.label == data[attribute]:
                    current = child
                    end = True
                    break
            
            if not end:
                if w:
                    return current.getName()
                else:
                    labels = members.value_counts([lab])
                    return labels.index[0][0]