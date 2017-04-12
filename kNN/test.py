import kNN

group,labels = kNN.createDataSet()
result = kNN.classify0([0, 0], group, labels, 3)
print(result)