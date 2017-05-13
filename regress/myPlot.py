# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:23:46 2017

@author: duqinyuan
"""
import regression
import matplotlib.pyplot as plt
from numpy import *


xArr, yArr = regression.loadDataSet('ex0.txt')

'''
ws = regression.standRegres(xArr, yArr)

xMat = mat(xArr)
yMat = mat(yArr)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy * ws
ax.plot(xCopy[:, 1], yHat)
plt.show()
'''




yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)

xMat = mat(xArr)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1].flatten().A[0], yHat[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()


