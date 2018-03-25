import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrowArgs = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.axl.annotate(nodeText, xy=parentPt, \
    xycoords='axes fraction', \
    xytext=centerPt, textcoords='axes fraction', \
    va='center', ha='center', bbox=nodeType, arrowprops=arrowArgs)

# def createPlot():
#     fig = plt.figure(1, facecolor="white")
#     fig.clf()
#     createPlot.axl = plt.subplot(111, frameon = False)
#     plotNode("decision node", (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode("leaf node", (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()

#get the tree leafs counts
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

#get the tree depth
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if maxDepth < thisDepth:
            maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2 + cntrPt[0];
    yMid = (parentPt[1] - cntrPt[1]) / 2 + cntrPt[1];
    createPlot.axl.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeText):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2 / plotTree.totalW, plotTree.yOff)     #cntrPt = (plotTree.xOff + 1.0 / plotTree.totalW / 2 + float(numLeafs)) / plotTree.totalW / 2, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeText)             #plot the mid text.
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #plot itself.
    secondDict = myTree[firstStr]
    plotTree.yOff -= 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key)) #plot chid tree
        else:
            plotTree.xOff += 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode) #plot leaf node
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff += 1.0 / plotTree.totalD              #exit this recursive, yOff need add (1.0 / plotTree.totalD)

def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks = [], yticks = [])
    createPlot.axl = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
