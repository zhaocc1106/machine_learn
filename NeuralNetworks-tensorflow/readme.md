# 深度学习-神经网络-tensorflow实践
# 目录架构
MNIST_data　　　　　　----　　　　手写识别训练和测试数据<br>
alex_net_module　　　　----　　　　AlexNet CNN模块<br>
cifar10_module　　　　----　　　　Cifar10 CNN模块<br>
image_net_origin_files　　　　----　　　　从ImageNet下载下来的几个分类图片的urls文件，每个分类大概有1000个图片<br>
plots　　　　----　　　　训练结果的plots图片<br>
tools　　　　----　　　　工具包，包括ImageNet图片下载器(根据urls下载)，ImageNet原始图片转成tfRecords文件工具，ImageNet的tfRecords文件阅读器<br>
MLP.py　　　　----　　　　简单多层神经网络分类器<br>
auto_encoder.py　　　　----　　　　自动编码器<br>
softmax_regression.py　　　　----　　　　根据softmax 回归的简单分类器<br>
