from numpy import *


# the function used to calculate probability vector of every word in every class,
#  and calculate the probability of every class.
# arg1: the train Matrix for every set of words.
# arg2: the train category for every set of words.
def trainNB0(trainMatrix, trainCategory):
    numTrainMatrix = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / numTrainMatrix  # calculate the probability of class1.
    # p0Num = zeros(numWords)
    # p1Num = zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainMatrix):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # calulate the sum of every word.
            p1Denom += sum(trainMatrix[i])  # calculate the sum of all words.
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num / p1Denom  # calculate the probability of every word for class1
    # p0Vect = p0Num / p0Denom  # calculate the probability of every word for class0

    # use log in order to avoid memory overflow.
    p1Vect = log(p1Num / p1Denom)  # calculate the probability of every word for class1
    p0Vect = log(p0Num / p0Denom)  # calculate the probability of every word for class0
    return p1Vect, p0Vect, pAbusive

# load some data set, and this class of this data set.
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'workless', 'garbage'],
                   ['my', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'workless', 'dog', 'food', 'stupid']]
    classVect = [0, 1, 0, 1, 0, 1]
    return postingList, classVect

# create the vocabulary list for data set.
def createVocabList(dataSet):
    vocabList = set()
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)

# create the wordVect in stand of whether one word is in inputset.
def setOfWords2Vec(vocablist, inputSet):
    numberWords = len(vocablist)
    wordVect = zeros(numberWords)
    for i in range(numberWords):
        if vocablist[i] in inputSet:
            wordVect[i] = 1
        #else:
            #print ("this word is not in vocablist:%s" %vocablist[i])
    return wordVect

# classify by probability vector of every word in every class and probability of every class.
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    #print("p1:", p1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    #print("p0:", p0)
    if p1 > p0:
        return 1
    else:
        return 0

# test function
def testingNB():
    listPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)

    trainMatrix = []
    for postinDoc in listPosts:
        trainMatrix.append(setOfWords2Vec(myVocabList, postinDoc))

    p1V, p0V, pAb = trainNB0(trainMatrix, listClasses)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #print("thisDoc:", thisDoc)
    print(testEntry, " classified as ", classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    #print("thisDoc:", thisDoc)
    print(testEntry, " classified as ", classifyNB(thisDoc, p0V, p1V, pAb))

if __name__ == '__main__':
     testingNB()






