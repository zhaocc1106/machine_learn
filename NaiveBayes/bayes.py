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

# create bag of words model vector.
def bagOfWords2Vec(vocablist, inputSet):
    numberWords = len(vocablist)
    wordVect = zeros(numberWords)
    for i in range(numberWords):
        if vocablist[i] in inputSet:
            wordVect[i] += 1
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

# text parse
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 中文句子解析
def chineseParse(documents):
    import jieba
    word_cut = jieba.cut(documents)
    word_list = list(word_cut)
    return word_list

# filter spam
def spamTest():
    docList = []; classList = []; fullText = []

    # 导入并解析邮件
    for i in range(1, 26):
        wordList = textParse(open("email/spam/%d.txt" % i).read())
        #print("wordList:\n" + str(wordList))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open("email/ham/%d.txt" % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocbList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []

    # 创建随机训练集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # 创建训练集词汇向量表
    trainMat = []; trainClass = []
    for i in trainingSet:
        trainMat.append(setOfWords2Vec(vocbList, docList[i]))
        trainClass.append(classList[i])

    # 计算朴素贝叶斯概率
    p1V, p0V, pSpam = trainNB0(array(trainMat), array(trainClass))
    #print("p0V:" + str(p0V) + "\np1V:" + str(p1V) + "\npSpam:" + str(pSpam))

    # 使用上述计算出的概率计算测试集合，并统计错误率
    errorCount = 0
    for docIndex in testSet:
        wordVect = setOfWords2Vec(vocbList, docList[docIndex])
        if classifyNB(wordVect, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is %f" % float(errorCount / len(testSet)))


# 根据出现频率排序,找出出现频率最高的30个单词
def calcMostReq(vocbList, fullText):
    import operator
    freqDict = {}
    for token in vocbList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key = operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

# 当地词汇分类测试
def localWords(feed1, feed0):
    import feedparser

    # 导入并解析RSS
    docList = []; classList = []; fullText = []
    minLen = min(len(feed0['entries']), len(feed1['entries']))
    for i in range(minLen):
        wordList = chineseParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = chineseParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    print("docList:", str(docList))
    vocbList = createVocabList(docList)

    # 移除30个高频词
    top30Words = calcMostReq(vocbList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocbList:
            vocbList.remove(pairW[0])

    # 创建随机训练词汇集
    trainingSet = list(range(2*minLen)); testSet = []
    print("len of trainningSet:", len(trainingSet))
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        #print("randIndex:", randIndex)
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # 生成训练词汇向量集
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocbList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 计算朴素贝叶斯概率
    p1V, p0V, p1Class = trainNB0(trainMat, trainClasses)

    # 计算分类错误概率
    errorCount = 0
    for docIndex in testSet:
        if classifyNB(bagOfWords2Vec(vocbList, docList[docIndex]), p0V, p1V, p1Class) != \
            classList[docIndex]:
            errorCount += 1
    print("the error rate: %f" % (float(errorCount) / len(testSet)))
    return vocbList, p0V, p1V

def localWordsTest():
    import feedparser
    ny = feedparser.parse('http://sports.qq.com/isocce/rss_isocce.xml') # 体育新闻
    sf = feedparser.parse('http://news.qq.com/milite/rss_milit.xml') # 国际新闻
    vocblist, pSF, pNY = localWords(ny, sf)

if __name__ == '__main__':
     #testingNB()
     #spamTest()
     localWordsTest()
