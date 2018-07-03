from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

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

def createC1(dataSet):
    """
    根据数据集合创建C1集合
    :param dataSet: 数据集合
    :return C1: C1集合
    """
    C1 = []
    for row in dataSet:
        for item in row:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport=0.5):
    """
    根据Ck的数据创建Lk，通过在所有数据集D中，扫描和计算Ck的每个元素的支持度，当支持度小于minSupport时就要忽略该元素
    :param D: 所有数据集
    :param Ck: Ck数据
    :param minSupport: 最小支持度
    :return Lk: 返回生成的Lk
    :return supportData: 元素的支持度
    """
    spCount = {}                                    # 记录每个Ck元素的支持度
    for d in D:
        for item in Ck:
            if item.issubset(d):                    # 每当元素在所有数据元素中出现一次，增加一次计数
                if not item in spCount.keys():
                    spCount[item] = 1.0
                else:
                    spCount[item] += 1.0
    numD = float(len(D))
    Lk = []
    supportData = {}
    for key in spCount:
        support = spCount[key] / numD               # 计算每个元素的支持度
        if support >= minSupport:                   # 只有不小于最小支持度的元素才被保留
            Lk.insert(0, key)
        supportData[key] = support
    return Lk, supportData

def aprioriGen(Lk, k):
    """
    Ck, Lk之间的关系如下
                 空集
            0   1   2   3                           Ck代表k行的所有元素, Lk代表k行支持度不小于最小支持度的元素（即频繁项集）
        01  02  03  12  13  23                      C(k+1)是由Lk随机组合成的组合，非频繁项集的超集肯定是非频繁项集
           012 013 023 123
                0123
    根据L(k-1)的数据生成Ck
    :param Lk: L(k-1)数据
    :param k:
    :return Ck: 生成的Ck
    """
    Ck = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:                            # 只有前k-2个项相同时才将两个集合合并，能够避免生成重复的集合
                Ck.append(Lk[i] | Lk[j])
    return Ck

def apriori(dataSet, minSupport=0.5):
    """
    获取dataSet中的频繁项集
    :param dataSet: 数据集合
    :param minSupport: 最小的支持度
    :return L: 频繁项集
    :return supportDat: 频繁项集元素的支持度
    """
    L = []                                          # 用于保存所有的Lk
    D = list(map(set, dataSet))
    C1 = createC1(dataSet)
    L1, supportDat = scanD(dataSet, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supportK = scanD(D, Ck, minSupport)     # 扫描Ck，从Ck得到Lk
        supportDat.update(supportK)
        L.append(Lk)
        k += 1
    return L, supportDat

def calcConf(freqSet, H, supportData, bigRuleList, minConf):
    """
    计算H集合对应的规则可信度，并保存到bigRuleList中。
    :param freqSet: 当前需要生成关联规则的频繁项集
    :param H: 关联规则的后件集合
    :param supportData: 频繁项集的支持度字典
    :param bigRuleList: 用于保存关联规则
    :param minConf: 最小关联规则可信度，小于该可信度的不纳入统计
    :return pruedH: 根据当前H生成的下一个需要统计的H集合
    """
    prunedH = []
    for consq in H:
        conf = supportData[freqSet] / supportData[freqSet-consq]    # 一条规则P ➞ H的可信度定义为support(P | H)/support(P)
        if conf >= minConf:
            print(freqSet-consq, "--->", consq, " conf:", conf)
            bigRuleList.append((freqSet-consq, consq, conf))        # 将生成的规则存放到bigRuleList中，规则是一个元组的格式(前件， 后件， 可信度)
            prunedH.append(consq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf):
    """
    根据最初的H，生成所有关联规则。
    :param freqSet: 当前需要生成关联规则的频繁项集
    :param H: 最初的H
    :param supportData: 频繁项集的支持度字典
    :param bigRuleList: 用于保存关联规则
    :param minConf: 最小关联规则可信度，小于该可信度的不纳入统计
    :return:
    """
    m = len(H[0])
    if len(freqSet) > (m + 1):                                        # 如下面的0123集合的关联规则生成过程图所示，该条件能使递归停止在最后一行
        Hnext = aprioriGen(H, m + 1)
        Hnext = calcConf(freqSet, Hnext, supportData, bigRuleList, minConf)
        if len(Hnext) > 0:
            rulesFromConseq(freqSet, Hnext, supportData, bigRuleList, minConf)


def generateRules(L, supportData, minConf=0.7):
    """
    关联规则量化，这种量化指标称为可信度(Confidence)。
    一条规则P ➞ H的可信度定义为support(P | H)/support(P)，P称为前件，H称为后件。
    关联规则生成函数，根据频繁项集的支持度和最小可信度，生成关联规则。
    对于0123集合的关联规则生成的过程应该如下图，从顶部向下:
                            0123->空集
                123->0  023->1  013->2  012->3
        23->01  13->02  12->03  03->12  02->13  01->23
                3->012  2->013  1->023  0->123
    能够看出来H的生成过程同频繁项集Ck的生成过程是一样的，所以可以直接使用aprioriGen来生成H。
    另外，可信度较低规则，所有以它的后件作为后件的规则的可信度都会较低，所以当一个规则的可信度小于最小可信度时，没有必要再继续计算以它后
    件为后件的规则的可信度。
    :param L: 频繁项集
    :param supportData: 支持度
    :param minConf: 最小可信度
    :return bigRuleList: 不小于minConf的关联规则列表
    """
    bigRuleList = []
    for i in range(1, len(L)):                                                  # 从频繁集C2开始，C1因为都是单元素频繁项集，所以构不成规则
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]                        # 生成后件H1集合
            calcConf(freqSet, H1, supportData, bigRuleList, minConf)            # 计算H1集合代表的关联规则
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf) # 根据后件H1集合，逐渐生成所有的后件，并计算每个后件代表的规则可信度
    return bigRuleList

if __name__ == '__main__':
    # dataSet = loadDataSet()
    # print("dataSet:", str(dataSet))
    # C1 = createC1(dataSet)
    # print("C1:", str(C1))
    # L1, supportDat = scanD(dataSet, C1, 0.5)
    # print("L1:", str(L1), " supportDat:", str(supportDat))
    # C2 = aprioriGen(L1, 2)
    # print("C2:", str(C2))
    """
    L, supportDat = apriori(dataSet, 0.5)
    print("L:", str(L), "\nsupportDat:", str(supportDat))
    rules = generateRules(L, supportDat, 0)
    print("rules:", str(rules))
    """

    # 使用蘑菇的特征数据来测试
    dataSet = loadDataFromFile("mushroom.dat")
    print("dataSet:", str(mat(dataSet).T))
    L, supportDat = apriori(dataSet, 0.5)
    print("L:", str(L), "\nsupportDat:", str(supportDat))
    rules = generateRules(L, supportDat, 0)
    print("rules:", str(rules))