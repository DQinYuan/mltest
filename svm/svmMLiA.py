"""
Created on Fri Apr 28 11:36:19 2017

@author: Administrator
"""
from numpy import *
from os import listdir

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j = i
    while ( j == i ):
        j = int(random.uniform(0, m))
    return j

#调整aj的值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

#简化版SMO算法
#C 松弛系数  toler 容错率  maxIter  alphas不变的最大迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iterCurr = 0
    while ( iterCurr < maxIter ):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ( (labelMat[i] * Ei < -toler) and (alphas[i] < C) or \
                 (labelMat[i] * Ei > toler) and (alphas[i] > 0) ):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if ( labelMat[i] != labelMat[j] ):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L=H");continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T -\
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0: print("eta>=0");continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if ( abs(alphas[j] - alphaJold) < 0.00001 ):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if ( alphas[i] > 0 ) and ( alphas[i] < C ): b = b1
                elif ( alphas[j] > 0 ) and ( alphas[j] < C ): b = b2
                else: b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iterCurr, i, alphaPairsChanged))
        if ( alphaPairsChanged == 0 ): iterCurr += 1
        else: iterCurr = 0
        print("iteration number: %d" % iterCurr)
    return b, alphas

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
        

def calcEk(oS, k):
#    fXk = float(multiply(oS.alphas, oS.labelMat).T *\
#                (oS.X * oS.X[k, :].T)) + oS.b)
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)     #使用核函数
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i : continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            #选择具有最大步长的j
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej
    
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C) or \
       (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0):
       j, Ej = selectJ(i, oS, Ei)
       alphaIold = oS.alphas[i].copy()
       alphaJold = oS.alphas[j].copy()
       if oS.labelMat[i] != oS.labelMat[j]:
           L = max(0, oS.alphas[j] - oS.alphas[i])
           H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
       else:
           L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
           H = min(oS.C, oS.alphas[j] + oS.alphas[i])
       if L == H: print("L==H");return 0
#       eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
#             oS.X[j, :] * oS.X[j, :].T
       eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]            #使用高斯径向基核函数
       if eta >= 0: print("eta>=0"); return 0
       oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
       oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
       updateEk(oS, j)
       if abs(oS.alphas[j] - alphaJold) < 0.00001:
           print("j not moving enough"); return 0
       oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * \
                       (alphaJold - oS.alphas[j])
       updateEk(oS, i)
#       b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T -\
#            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
#       b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T -\
#            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
       b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] -\
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
       b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] -\
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]                             #使用核函数
       if (oS.alphas[i] > 0) and (oS.alphas[i] < oS.C): oS.b = b1
       elif (oS.alphas[j] > 0) and (oS.alphas[j] < oS.C): oS.b = b2
       else: oS.b = (b1 + b2) / 2.0
       return 1
    else:
       return 0

#Platt SMO算法
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iterCurr = 0
    entireSet = True
    alphaPairsChanged = 0
    #达到到最大迭代次数或者遍历整个集合都未对alpha进行修改时退出循环
    while (iterCurr < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            #完整遍历
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print("fullSet, iter: %d i: %d, pairs changed %d" % (iterCurr, i, alphaPairsChanged))
            iterCurr += 1
        else:
            #非边界值循环
            nonBoundsIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundsIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i: %d, pairs changed: %d" % (iterCurr, i, alphaPairsChanged))
            iterCurr += 1
        #在非边界值循环与完整循环之间切换
        if entireSet: entireSet = False
        elif alphaPairsChanged == 0 : entireSet = True
        print("iteration number: %d" % iterCurr)
    return oS.b, oS.alphas

#得到返回结果后根据data * np.mat(w) + b大于0还是小于0来判断属于哪个类别
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def kernelTrans(X, A, kTup):
    m,n = shape(X)
    K = mat(zeros((m, 1)))
    #线性核函数，即普通的内积
    if kTup[0] == 'lin': K = X * A.T
    #径向基核函数
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

#k1为径向基核函数的到达率
def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]                  #构建支持向量矩阵
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    #使用核函数的情况下进行预测的方法
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (errorCount / m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    #使用核函数情况下进行预测的方法
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (errorCount / m))
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
            
def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (errorCount / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (errorCount / m))
    
    