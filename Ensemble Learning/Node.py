# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 19:15:02 2022

@author: Ryan Lam
"""

import pandas as pd

class Node(object):
    
    def __init__(self):
        self.label = ""
        self.attribute = ""
        self.children = []
        self.members = pd.DataFrame()
        self.depth = 0
        self.isLeaf = False
        
    def setLabel(self, name):
        self.label = name
        
    def getLabel(self):
        return self.label
        
    def setAttribute(self, name):
        self.attribute = name
        
    def getAttribute(self):
        return self.attribute
    
    def addMembers(self, data):
        self.members = pd.concat([self.members,data])
        
    def getMembers(self):
        return self.members
    
    def setDepth(self, value):
        self.depth = value
        
    def getDepth(self):
        return self.depth
        
    def addChild(self,label,members,depth):
        node = Node()
        node.addMembers(members)
        node.setLabel(label)
        node.setDepth(depth)
        self.children.append(node)
        
    def getChildren(self):
        return self.children
    
    def display(self):
        print(self.depth)
        print(self.members)
        for child in self.children:
            child.display()
            
    def setLeaf(self):
        self.isLeaf = True