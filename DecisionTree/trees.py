from math import log
import operator

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

# choose the best feature to split data set.
def chooseBestFeature(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        featureSet = set(featureList)           #create unique feature list
        newEntropy = 0.0
        for featureValue in featureSet:
            subDataSet = splitDataSet(dataSet, i, featureValue)     #calculate shannonEnt for every subDataSet
            prop = len(subDataSet) / float(len(dataSet))
            #print("\nsubDataSet splited by feature %d value %d:\n" %(i, featureValue))
            #print(subDataSet)
            newEntropy += prop * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #calculate info gain.
        #print("\nthe info gain splited by feature %d is %f" %(i, infoGain))
        if (bestInfoGain < infoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def mayjorityCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, featLabels):
    labels = featLabels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(dataSet):       #stop split tree when all classes is the same.
        return classList[0]
    if len(dataSet[0]) == 1:
        return mayjorityCount(classList)                    #use the mayjority to represent the class.
    bestFeat = chooseBestFeature(dataSet)
    bestLabel = labels[bestFeat]
    myTrees = {bestLabel:{}}
    del(labels[bestFeat])
    featList = [example[bestFeat] for example in dataSet]
    featSet = set(featList)
    for value in featSet:
        subLabels = labels[:]
        myTrees[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)    #create the child tree
    # print("\nthe sub trees is")
    print(myTrees)
    return myTrees

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":                #this is dict, need continue classify.
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]                             #this is a leaf, the value is the classLabel
    return classLabel