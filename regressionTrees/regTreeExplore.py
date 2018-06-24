from numpy import *
from tkinter import *
from regTrees import *
import matplotlib

matplotlib.use("TkAgg")  # 设定matplotlib的后端为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def reDraw(tolS, tolN):
    """
    根据tolS，tolN参数重新绘制树
    :param tolS: 可容忍的误差
    :param tolN: 数据分割时，数据集最少的个数
    """
    reDraw.f.clf()                                          # 清屏
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():                                     # checkbox选中代表使用模型树
        if tolN < 2: tolN = 2                               # 对于模型树来说，tolN最小为2
        modelTree = createTree(reDraw.rawDat, modelLeaf, modelErr, ops=[tolS, tolN])    # 创建模型树
        yHat = createForecast(modelTree, reDraw.testDat, modelTreeEval)
    else:
        regTree = createTree(reDraw.rawDat, regLeaf, regErr, ops=[tolS, tolN])          # 创建回归树
        yHat = createForecast(regTree, reDraw.testDat, regTreeEval)
    reDraw.a.scatter(reDraw.rawDat[:, 1].T.A[0], reDraw.rawDat[:, 2].T.A[0], s=5, c='b')              # 绘出所有点
    # print("reDraw.testDat[:, 1].T.A[0]:", str(reDraw.testDat[:, 1].T.A[0]))
    # print("yHat.T.A[0]", str(yHat.T.A[0]))
    reDraw.a.plot(reDraw.testDat[:, 1].T.A[0], yHat.T.A[0], linewidth=2.0)                            # 绘出回归线
    reDraw.canvas.draw()

def getInputs():
    """
    获取输入框中的值，即tolS和tolN的值
    :return: 返回tolS和tolN的值
    """
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolS, tolN

def drawNewTree():
    """
    获取相关绘制参数并绘制新的树
    """
    tolS, tolN = getInputs()
    reDraw(tolS, tolN)


root = Tk()  # 创建Tk GUI的根

# 在Tk的GUI上放置一个画布
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)  # 将画布放置在0行，并且跨3列

# 创建tolN输入框
Label(root, text='tolN').grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')

# 创建tolS输入框
Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')

# 创建重绘按钮
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)

# 创建树类型复选框
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

dataSet = loadDataSet("sine.txt")
m, n = shape(mat(dataSet))
dataMat = mat(ones((m, n + 1)))
dataMat[:, 1:3] = mat(dataSet)
reDraw.rawDat = dataMat                                 # 用于构建树的测试数据

testDat = mat(arange(min(reDraw.rawDat[:, 1]),\
                        max(reDraw.rawDat[:, 1]), 0.01)) # 用于绘制预测回归线的数据
m, n = shape(testDat.T)
# print("testDat.T:", str(testDat.T))
reDraw.testDat = mat(ones((m, n+1)))
reDraw.testDat[:, 1:n+1] = testDat.T
# print("rawDat:", str(reDraw.rawDat))
# print("testDat:", str(reDraw.testDat))

reDraw(1.0, 10)
root.mainloop()