from trees import *

dataSet, featLabels = createDataSet()
shannonEnt = calcShannonEnt(dataSet)
print("shannonEnt = %f" % (shannonEnt))
