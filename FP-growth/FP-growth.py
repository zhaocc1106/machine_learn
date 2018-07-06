#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
"""
FP-Growth树算法的实现

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/7/6 10:15
"""

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
                   headerTable[orderedItems[0]][1], headerTable, count)     # 递归生长下一个节点到当前树的子树中


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
    for trans in dataSet.keys():                                            # 循环计算所有元素出现的次数，并记录到headerTable中
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
    retTree = TreeNode('Null set', 1, None)                                 # 创建根节点
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in set(headerTable.keys()):
                localD[item] = headerTable[item][0]                         # 记录当前事务集合中每个元素出现的次数，从header table中获取
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),\
                                 key=lambda p: p[1], reverse=True)]         # 根据每个元素出现的次数进行排序，次数越多越靠前
            print("orderedItems:", str(orderedItems))
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
                                    key=lambda p: p[0])]            # 对headerTable进行排序
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


if __name__ == '__main__':
    # rootNode = TreeNode('a', 1, None)
    # rootNode.children['b'] = TreeNode('b', 2, None)
    # rootNode.disp(1)

    dataSet = loadSimpDat()
    dataDic = createInitSet(dataSet)
    print("dataDic:", str(dataDic))
    fpTree, headerTable = createTree(dataDic, 3)
    print("fpTree:")
    fpTree.disp()
    print("headerTable:", str(headerTable))
    condPattBases = findPrefixPath(headerTable['t'][1])
    print("condPattBases:", str(condPattBases))
    prefixPath = set([])
    freqItems = []
    findFreqItems(fpTree, headerTable, 3, prefixPath, freqItems)
    print("freqItems:", str(freqItems))