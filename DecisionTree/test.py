from trees import *
from treePlotter import *

dataSet, featLabels = createDataSet()
# print("\nfeat:")
# print(featLabels)
# print("\ndataSet:")
# print(dataSet)
# shannonEnt = calcShannonEnt(dataSet)
# print("shannonEnt = %f" % (shannonEnt))
# bestFeature = chooseBestFeature(dataSet)
# print("\nbestFeature is %d" % (bestFeature))
tree = createTree(dataSet, featLabels)
print("\nthe complete trees is")
print(tree)
print("the leaf counts is %d, the tree depth is %d" %(getNumLeafs(tree), getTreeDepth(tree)))
tree['no surfacing'][3] = 'maybe'
createPlot(tree)