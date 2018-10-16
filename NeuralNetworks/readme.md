#深度学习-神经网络

#目录架构
mnistData----里面有一个手写识别训练数据库打成的pkl包<br>
plots----通过神经网络训练出来的accuracy-epoches变化图<br>
testDigits----测试训练结果使用的简单的手写识别数据矩阵文件<br>
trainingDigits----训练神经网络时使用的简单的手写识别数据矩阵文件，由于训练数据量少，容易过拟合<br>
ErrBackPropgation.py----使用标准错误反向传播算法写的一个简单的全连接、多层次的前馈神经网络。没有加入正则化、先进初始化等优化方法，
只是通过反向传播是一个简单的前馈网络，通过该网络能够解决简单的异或非线性问题，以及一些线性分类问题。<br>
ImprovedFullConnectedNN.py----一个全连接、多层次前馈神经网络，加入了L1、L2、Dropout正则化手段；加入高斯分布、Xavier等初始化权重手段；
加入平方误差和交叉熵等多种cost方法；加入动态学习率优化手段。通过该神经网络，已经能将accuracy训练到98%以上。<br>
KerasCNN.py----通过keras框架写的一个很简单的CNN实例。<br>
TheanoCNN.py----通过Theano实现的CNN，比较复杂，主要是通过构造各个层次（Convolution-Pool layer、Full-connected layer、Softmax layer）的theano版的
数学表达式，theano提供了grad求解梯度函数，构造训练和测试使用的theano版的function，最后将训练数据真正的用到各个function中从而实现训练和测试该神经网络。<br>
