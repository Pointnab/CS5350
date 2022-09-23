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
        
    def train(self, data, depth = 6):
        if self.tree is None:
            self.tree = Node.Node()
        
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
        initialEntropy = 0
        for n in base:
            initialEntropy -= n*math.log(n)
        
        gain = {}
        
        for attribute in members.columns:
            if attribute == lab:
                continue
            
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
        
        for l in members[best].unique():
            mem = members[members[best] == l]
            mem = mem.drop(best, axis=1)
            node.setAttribute(best)
            node.addChild(l,mem,node.depth+1)
        
        for child in node.children:
            self.splitMembers(child,depth)
            
    def test(self, data):
        pred = []
        
        for idx, point in data.iterrows():
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