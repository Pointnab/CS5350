# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:56:19 2022

@author: Ryan Lam
"""

import Node
import pandas as pd
import math

class ME(object):
    
    def __init__(self):
        self.tree = None
        self.thresholds = {}
        self.majority = {}
        
    def train(self, data, depth = 6, b=False, u=False):
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
        
        self.splitMembers(self.tree, depth)
        
        #self.tree.display()
        
        return
    
    def splitMembers(self, node, depth):
        if node.depth >= depth:
            node.setLeaf()
            return
        
        members = node.getMembers()
        lab = members.columns[len(members.columns)-1]
        
        if len(members[lab].unique()) == 1:
            node.setLeaf()
            return
        
        base = members.value_counts([lab],normalize=True)
        initialEntropy = 1-base[0]
        
        gain = {}
        
        for attribute in members.columns:
            if attribute == lab:
                continue
            
            vals = members.value_counts([attribute,lab])
            ent = 0
            
            for label in vals.index.levels[0]:
                cat = vals.loc[label,:]
                h = 1-cat[0]/sum(cat)
                ent += h * sum(cat)/sum(vals)
            g = initialEntropy - ent
            gain.update({attribute:g})
        
        best = max(gain, key=gain.get)
        
        for l in members[best].unique():
            mem = members[members[best] == l]
            mem = mem.drop(best, axis=1)
            node.setAttribute(best)
            node.addChild(l,mem,node.depth+1)
        
        for child in node.children:
            self.splitMembers(child,depth)
            
    def test(self, data, b=False, u=False):
        pred = []
        
        for idx, point in data.iterrows():
            if b:
                for attribute in self.thresholds:
                    thresh = self.thresholds[attribute]
                    if point[attribute]>=thresh:
                        point[attribute] = thresh
                    else:
                        point[attribute] = thresh-1
                        
            if u:
                for attribute in self.majority:
                    maj = self.majority[attribute]
                    if point[attribute]=="unknown":
                        point[attribute] = maj
            
            pred.append(self.classify(point))
        
        return pred
    
    def classify(self, data):
        current = self.tree
        
        while True:
            end = False
            members = current.getMembers()
            lab = members.columns[len(members.columns)-1]
            
            if current.isLeaf:
                if len(members[lab].unique()) == 1:
                    return members.iloc[0,members.shape[1]-1]
            
                labels = members.value_counts([lab])
                return labels.index[0][0]
            
            attribute = current.getAttribute()
            for child in current.getChildren():
                if child.label == data[attribute]:
                    current = child
                    end = True
                    break
            
            if not end:
                labels = members.value_counts([lab])
                return labels.index[0][0]