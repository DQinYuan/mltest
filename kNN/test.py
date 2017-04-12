import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

# group,labels = kNN.createDataSet()
# result = kNN.classify0([0, 0], group, labels, 3)
# print(result)

datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()