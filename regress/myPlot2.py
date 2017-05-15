# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:45:04 2017

@author: 燃烧杯
"""

import regression
import matplotlib.pyplot as plt
from numpy import *

abX, abY = regression.loadDataSet("abalone.txt")
ridgeWeights = regression.ridgeTest(abX, abY)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(arange(-10, 20) ,ridgeWeights)
plt.xlabel("log(lambda)")
plt.show()