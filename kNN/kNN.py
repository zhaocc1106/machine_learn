from numpy import *
import operator

def createDateSet():
    group = array([[1.0], [1.1], [0.1], [0.0]])
    labels = array(['A', 'A', 'B', 'B'])
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)
    distance = sqDistance ** 0.5
    sortedDistanceIndicies = distance.argsort()
    classCount = {}
    for i in range (k):
        voteIlabel = labels[sortedDistanceIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

# group, labels = createDateSet()
# print(classify0([0.5, 0.8], group, labels, 3))

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVals, (m, 1))
    normalDataSet = normalDataSet / tile(ranges, (m, 1))
    return normalDataSet, ranges, minVals

def img2Vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        # print(str(lineStr))
        for j in range(32):
            returnVect[0, i*32+j] = int(lineStr[j])
    return returnVect

