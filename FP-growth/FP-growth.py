#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
"""
FP-Growth树算法的实现

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/7/6 10:15
"""
import datetime
import twitter
from time import sleep
import re

class TreeNode(object):
    """树节点类型.

    FP-Growth树的节点类型

    Attributes:
        name: 节点的名字
        count: 该节点元素在所有数据集中出现的次数
        parent: 该节点的父节点
        nodeLink: 用于连接相似的元素项
        children: 该节点的子节点
    """

    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.parent = parentNode
        self.nodeLink = None
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        """将树的数据展示出来

        将树的数据展示出来，并且通过空格来分层次

        Args:
            ind: 代表节点的层级，同事代表有几个空格在最开始
        """
        print("  " * ind, self.name, " ", self.count)
        for child in self.children.values():
            child.disp(ind + 1)               # 子节点层级加一

def loadDataFromFile(fileName):
    """
    从文件中获取需要的数据
    :param fileName: 数据文件名
    :return dataArr: 数据
    """
    dataArr = []
    file = open(fileName)
    for line in file.readlines():
        lineArr = line.strip().split(' ')
        dataArr.append(lineArr)
    return dataArr


def loadSimpDat():
    simpleDat = [['r', 'z', 'h', 'j', 'p'],
                 ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                 ['z'],
                 ['r', 'x', 'n', 'o', 's'],
                 ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                 ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpleDat


def createInitSet(dataSet):
    retDict = {}
    for i in range(len(dataSet)):
        retDict[frozenset(dataSet[i])] = 1
    return retDict


def updateHeader(nodeToLink, newNode):
    """将node添加到nodeToLink的链表最后

    Args:
        nodeToLink: 节点链表
        newNode: 新节点
    """
    while nodeToLink.nodeLink != None:
        nodeToLink = nodeToLink.nodeLink
    nodeToLink.nodeLink = newNode


def updateTree(orderedItems, retTree, headerTable, count):
    """使用排序好的事务项集，更新FP树

    将某次事务项集排序好后，添加到retTree树中去，该函数是一个递归函数，每次函数调用只添加一个元素到树中，所以树像是逐渐生长一样。

    Args:
        orderedItems: 排好序的某次事务项集
        retTree: 当前FP-Growth树
        headerTable: header table，如createTree函数中解释
        count: 该事务项集在整个数据集中出现的次数
    """
    if orderedItems[0] in retTree.children:
        retTree.children[orderedItems[0]].inc(count)                        # 如果该元素已经存在，则增加该节点的计数
    else:
        retTree.children[orderedItems[0]] = TreeNode(\
            orderedItems[0], count, parentNode=retTree)                     # 如果该元素不存在，则生成新的节点添加到当前树中
        if headerTable[orderedItems[0]][1] == None:
            headerTable[orderedItems[0]][1] =\
                retTree.children[orderedItems[0]]                           # 如果该元素对应的header table中的元素尚且没有连接任何节点，则认为该节点是生长出来的第一个节点，则连接此节点
        else:
            updateHeader(headerTable[orderedItems[0]][1],\
                         retTree.children[orderedItems[0]])                 # 如果该元素对应的header table中的元素已经连接节点，则将节点连接到header table中的节点链表的最后
    if len(orderedItems) > 1:
        updateTree(orderedItems[1:],\
                   retTree.children[orderedItems[0]], headerTable, count)   # 递归生长下一个节点到当前树的子树中


def createTree(dataSet, minSup=1):
    """构建FP-Growth树

    根据数据集合以及最小支持度来构建FP-Growth树

    Args:
        dataSet: 数据集字典，形式如下：{frozenset(['a', 'b', 'c']): 1, frozenset(['b', 'c', d]): 1, ...}
        minSup: 最小支持度，小于该最小支持度的项集应该被舍去

    Returns:
        retTree: FP-Growth树
        headerTable: 头指针表，header table中存放了每个元素，以及该元素在数据集合中出现的次数，并且连接该元素的在FP-Growth树
        中的第一个该元素节点，形如：headerTable: {'r': [3, None], 'p': [2, None], 'z': [5, None], 'y': [3, None]}
    """
    #--Begin: 第一次循环数据集合生成header table
    headerTable = {}
    iter = 0
    for trans in dataSet:                                                   # 循环计算所有元素出现的次数，并记录到headerTable中
        iter += 1
        print("[First iteration %d] trans:%s" % (iter, str(trans)))
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    for k in list(headerTable.keys()):
        if headerTable.get(k, 0) < minSup:
            del(headerTable[k])                                             # 如果headerTable中某元素出现的次数小于最小支持度，则删除该元素

    if len(headerTable) == 0:
        return None, None                                                   # 如果header table不存在元素，则返回None

    for k in headerTable:
        headerTable[k] = [headerTable[k], None]                             # headerTable连接了该元素的在FP-Growth树中的第一个该元素节点
    print("headerTable:", str(headerTable))
    #--End: 第一次循环数据集合生成header table

    #--Begin: 第二次循环数据集合，构建FP-Growth树
    freqItemSets = set(headerTable.keys())
    retTree = TreeNode('Null set', 1, None)                                 # 创建根节点
    iter = 0
    for tranSet, count in dataSet.items():
        iter += 1
        print("[Second iteration %d] trans %s" % (iter, str(tranSet)))
        localD = {}
        for item in tranSet:
            if item in freqItemSets:
                localD[item] = headerTable[item][0]                         # 记录当前事务集合中每个元素出现的次数，从header table中获取
        print("[Second iteration %d] localD: %s" % (iter, str(localD)))
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),\
                                 key=lambda p: p[1], reverse=True)]         # 根据每个元素出现的次数进行排序，次数越多越靠前
            # print("orderedItems:", str(orderedItems))
            updateTree(orderedItems, retTree, headerTable, count)           # 将本次排好序的项集，构建到FP-Growth树中去
    return retTree, headerTable

    #--End: 第二次循环数据集合，构建FP-Growth树


def ascendTree(leafNode, prefixPath):
    """获取叶子节点前缀

    通过从叶子节点回溯树，从而获得该叶子节点的前缀

    Args:
       leafNode: 叶子节点
       prefixPath: 保存该叶子节点的前缀
    """
    if leafNode.parent.name != 'Null set':
        prefixPath.append(leafNode.parent.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(treeNode):
    """寻找节点的前缀

    寻找树的某个节点所代表的元素的所有的前缀集合，即所有的路径

    Args:
        treeNode: 树的某个节点

    Returns:
        节点所代表的项元素所有的路径
    """
    paths = {}                                              # 每个路径都有该路径的计数，所有需要用字典来记录路径及他的路径计数
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 0:
            paths[frozenset(prefixPath)] = treeNode.count
        treeNode = treeNode.nodeLink                        # 通过nodeLink连接的所有该元素的节点，来获取该元素所有路径
    return paths


def findFreqItems(inTree, headerTable, minSupport, preFix, freqItemsList):
    """寻找频繁项集

    通过递归构建条件FP树，找到所有的频繁项集

    Args:
        inTree: 初始的FP-Growth树
        headerTable: 头指针表，如createTree中的解释
        minSupport: 最小支持度，用来剔除不满足最小支持度的项集
        preFix: 前缀，在构建条件FP树时的前缀（条件模式基），初始为空
        freqItemsList: 用于保存频繁项集的列表
    """
    headers = [v[0] for v in sorted(headerTable.items(),\
                                    key=lambda p: p[1][0])]         # 对headerTable进行排序
    for header in headers:
        newPreFix = preFix.copy()
        newPreFix.add(header)
        freqItemsList.append(newPreFix)
        condPattBases = findPrefixPath(headerTable[header][1])      # 找到条件模式基，即从树的根节点到达该header元素代表的所有节点的路径
        condTree, condHead = createTree(condPattBases, minSupport)  # 使用条件模式基作为FP树的数据集，创建FP条件树
        """
        可以想象出来，当找到某个节点的所有条件模式基（所有路径集合），使用这些数据集进行创建FP树，因为创建FP树时会将不满足minSupport
        条件的元素剔除，则剩下来的元素组成headerTable，则之前节点作为前缀，后生成的headerTable作为后缀组成项集，这些都是频繁项集
        """

        if condHead != None:
            print("newPreFix:", str(newPreFix))
            condTree.disp()
            findFreqItems(condTree, condHead,\
                          minSupport, newPreFix, freqItemsList)     # 当条件树的headerTable不为空的时候，需要递归在该条件树上寻找所有的频繁项集


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)  # 过滤掉url
    listOfTokens = re.split(r'\W*', urlsRemoved)                                            # 过滤掉符号
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]                            # 过滤掉单字符，并小写化


def getLotsOfTweets(searchStr):
    """从推特中获取关键字相关的推文

    Args:
        searchStr: 关键字

    Returns:
        推文结果
    """
    CONSUMER_KEY = '6oAAneQjWtQXZPHWW5JA9lOw9'
    CONSUMER_SECRET = 'eoubjNZaKvXVAnEFSfGrjXpjE1ZkkY0D96NSw6V4wu5lDdBebW'
    ACCESS_TOKEN_KEY = '1015486145713410048-tTL7TQod2cPGClthtHhskEeLIa1kIw'
    ACCESS_TOKEN_KEY_SECRET = 'o989shxl23FRsxT5JxEXj5se2OGrmFJYjHKR10hK1rbxo'
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_KEY_SECRET)
    resultsPages = []                                                           # 保存所有页面的推文
    for i in range(1, 15):
        print("fetching page %d" % i)
        searchRes = api.GetSearch(searchStr, count=15)
        resultsPages.append(searchRes)
        sleep(3)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    """根据twitter搜索到的结果，找到频繁项集

    Args:
        tweetArr: twitter搜索到的结果
        minSup: 最小支持度

    Returns:
        频繁项集
    """
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    findFreqItems(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList


if __name__ == '__main__':
    # rootNode = TreeNode('a', 1, None)
    # rootNode.children['b'] = TreeNode('b', 2, None)
    # rootNode.disp(1)

    # 测试简单的数据
    # dataSet = loadSimpDat()
    # dataDic = createInitSet(dataSet)
    # print("dataDic:", str(dataDic))
    # fpTree, headerTable = createTree(dataDic, 3)
    # print("fpTree:")
    # fpTree.disp()
    # print("headerTable:", str(headerTable))
    # condPattBases = findPrefixPath(headerTable['t'][1])
    # print("condPattBases:", str(condPattBases))
    # prefixPath = set([])
    # freqItems = []
    # findFreqItems(fpTree, headerTable, 3, prefixPath, freqItems)
    # print("freqItems:", str(freqItems))

    # 测试复杂的大量数据
    startTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    dataSet = loadDataFromFile('kosarak.dat')
    dataDic = createInitSet(dataSet)
    fpTree, headerTable = createTree(dataDic, 100000)
    print("headerTable:", str(headerTable))
    print("fpTree:")
    fpTree.disp()
    prefixPath = set([])
    freqItems = []
    findFreqItems(fpTree, headerTable, 100000, prefixPath, freqItems)
    print("freqItems:", str(freqItems))
    stopTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("startTime:", str(startTime))
    print("stopTime:", str(stopTime))

    # dataSet = loadDataFromFile('kosarak.dat')
    # num = 0
    # for tran in dataSet:
    #     has6 = False
    #     has11 = False
    #     for item in tran:
    #         if item == '6':
    #             has6 = True
    #         if item == '1':
    #             has11 = True
    #     if has6 and has11:
    #         num += 1
    # print("num:", num)