# 生成网络模型
模型通过学习一些数据，然后生成类似的数据。例如让机器看一些动物图片，然后自己来产生动物的图片，这就是生成。

## Auto-Encoder(自动编码器)
auto_encoder.py

## Variational Auto-encoder(变分自动编码器)
cvae.py
### 参考
[变分自编码器VAE](https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/79675832)<br>
[直观解读KL散度的数学概念](https://www.jianshu.com/p/7b7c0777f74d)<br>
[正态分布](https://blog.csdn.net/hhaowang/article/details/83898881)<br>
[正态分布分位数表](https://blog.csdn.net/lanchunhui/article/details/51754055)

## Generative Adversarial Network模型
GAN.dcgan.py
### 参考
[Generative Adversarial Nets论文](https://arxiv.org/pdf/1406.2661.pdf)<br>
[Deep convolutional generative adversarial network论文](https://arxiv.org/pdf/1511.06434.pdf)<br>
[译文 | 让深度卷积网络对抗：DCGAN——深度卷积生成对抗网络](https://ask.julyedu.com/question/7681)<br>
[生成对抗网络——GAN](https://blog.csdn.net/leviopku/article/details/81292192)<br>
[keras实现dcgan](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb#scrollTo=rF2x3qooyBTI)

## pix2pix模型
GAN.pix2pix.py<br>
GAN.photo2sketch.py
### 训练结果
![photo2sketch](https://github.com/zhaocc1106/machine_learn/blob/master/NeuralNetworks-tensorflow/generation_network_model/GAN/photo2sketch_01.png)<br>
![satellite_map_to_2d_map](https://github.com/zhaocc1106/machine_learn/blob/master/NeuralNetworks-tensorflow/generation_network_model/GAN/pix2pix_satellite_map_to_2d_map.png)
### 参考
[Image-to-Image Translation with Conditional Adversarial Networks论文](https://arxiv.org/abs/1611.07004)<br>
[keras实现](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/pix2pix.ipynb#scrollTo=0FMYgY_mPfTi)<br>
[数据集](https://www.kaggle.com/vikramtiwari/pix2pix-dataset)
