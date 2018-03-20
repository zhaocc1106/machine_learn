from kNN import *
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

def datingClassTest():
    hoRatio = 0.04
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normalDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normalDataSet.shape[0]
    errorCount = 0.0
    numTestVecs = int(m * hoRatio)
    for i in range(numTestVecs):
        classifierResult = classify0(normalDataSet[i, :], normalDataSet[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with :%d, the real answer is :%d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f" % (errorCount / (float(numTestVecs))))

def handWritingTest():
    hwLabels = [];
    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, : ] = img2Vector("trainingDigits/%s" % (fileNameStr))
    testFileList = listdir("testDigits")
    errorCount = 0.0
    m = len(testFileList)
    for i in range(m):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectUnderTest = img2Vector("testDigits/%s" % fileNameStr)
        classifierResult = classify0(vectUnderTest, trainingMat, hwLabels, 3)
        if (classifierResult != classNumStr):
            print("\nerror file", str(fileNameStr))
            print("the classifier result is %d, the real answeris %d" % (classifierResult, classNumStr))
            errorCount += 1.0
    print("\nthe total number of errors is %d" % errorCount)
    print("\nthe error rate is %f" % (errorCount/float(m)))

# datingClassTest()
    #print("normalDataSet: \n" + str(normalDataSet), "\nranges: \n" + str(ranges), "\nminVals: \n" + str(minVals))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(normalDataSet[:, 1], normalDataSet[:, 2], 1.5*array(datingLabels), 1.5*array(datingLabels))
    # plt.show()

handWritingTest()

