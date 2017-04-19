# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:10:04 2017

@author: duqinyuan
"""
import trees
import treeplotter

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
treeplotter.createPlot(lensesTree)
