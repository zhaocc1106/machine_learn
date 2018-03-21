from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    #calculate the sum for every kind begin.
    for featVect in dataSet:
        currentLabel = featVect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #calculate the sum for every kind end.

    for key in labelCounts:
        prop = float(labelCounts[key]) / numEntries #calculate the prop for every kind
        shannonEnt -= prop * log(prop, 2)           #calculate the shannonEnt
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']
               ]
    featLabels = ['no surfacing', 'flippers']
    return dataSet, featLabels

def splitDataSet(dataSet, feat, val):
    retDataSet = []
    for featVec in dataSet:
        if featVec[feat] == val:
            reducedFeatVec = featVec[:feat]
            reducedFeatVec.extend(featVec[feat+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet