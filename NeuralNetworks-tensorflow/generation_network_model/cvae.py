#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Conditional variational auto-encoder model.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/7/13 下午2:47
"""

# 3rd-part libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow import keras

BATCH_SIZE = 100
ORIGINAL_DIM = 28 * 28  # 原始图片数据的维度
LATENT_DIM = 2  # 隐变量个数
INTERMEDIATE_DIM = 256  # 中间层维度
EPOCHS = 10
NUM_CLASSES = 10


def load_data():
    """Load mnist dataset.

    Returns:
        The dataset.
    """
    # 加载MNIST数据集
    (x_train, y_train_), (x_test, y_test_) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # 将3维图片数据偏平化为1维，即从(28, 28, 1)reshape为(784, )
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_train = keras.utils.to_categorical(y_train_, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test_, NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Build the cvae model.

    Returns:
        The model.
    """
    # 原始图片数据输入
    x = keras.Input(shape=(ORIGINAL_DIM,))

    # 增加一个中间层，用于生成p(Z|X)正态分布的均值和方差
    h = keras.layers.Dense(INTERMEDIATE_DIM, activation='relu')(x)
    # 算p(Z|X)的均值和方差
    z_mean = keras.layers.Dense(LATENT_DIM)(h)
    z_log_var = keras.layers.Dense(LATENT_DIM)(h)

    # 原始图片label输入
    y = keras.Input(shape=(NUM_CLASSES,))
    # 这里就是直接构建每个类别的均值
    yh = keras.layers.Dense(LATENT_DIM)(y)

    # 重参数技巧
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(z_mean))
        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    # 重参数层，相当于给输入加入噪声
    z = keras.layers.Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean,
                                                                   z_log_var])

    # 解码层，也就是生成器部分
    decoder_h = keras.layers.Dense(INTERMEDIATE_DIM, activation='relu')
    decoder_mean = keras.layers.Dense(ORIGINAL_DIM, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # 建立模型
    vae = keras.Model([x, y], [x_decoded_mean, yh])

    # xent_loss是decoder生成的图片与原始图片的重构loss，kl_loss是模拟正态分布与标准正态
    # 分布的KL loss
    xent_loss = keras.backend.sum(
        keras.backend.binary_crossentropy(x, x_decoded_mean), axis=-1)
    # 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
    kl_loss = - 0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean - yh) - keras.backend.exp(
            z_log_var),
        axis=-1)

    # 同时降低encoder和decoder的loss
    vae_loss = keras.backend.mean(xent_loss + kl_loss)

    # add_loss是新增的方法，用于更灵活地添加各种loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    # 构建encoder，然后观察各个数字在隐空间的分布
    encoder = keras.Model(x, z_mean)

    # 构建decoder，即生成器
    decoder_input = keras.Input(shape=(LATENT_DIM,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    decoder = keras.Model(decoder_input, _x_decoded_mean)

    # 输出每个类的均值向量
    class_mean = keras.Model(y, yh)

    models = {"full_vae": vae,  # 完整的vae模型
              "encoder": encoder,  # Encoder模型，输出模拟正态分布的均值和方差，
              # 模拟高斯噪声采样生成decoder的输入数据
              "decoder": decoder,  # Decoder模型，根据输入的噪声，生成模拟图片数据
              "class_mean": class_mean  # 根据输入的label，生成每个类的模拟均值
              }
    return models


def train_and_predict(models, x_train, y_train, x_test, y_test):
    """Train the vae model and use the model to generate images.

    Args:
        models: The vae model.
        x_train: The training image.
        y_train: The training label.
        x_test: The test image.
        y_test: The test label.
    """
    vae, encoder, decoder, class_mean = models.values()
    # Fit the model.
    vae.fit([x_train, y_train],
            shuffle=True,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=([x_test, y_test], None))

    y_test_ = np.argmax(y_test, axis=-1)
    # 生成模拟高斯分布的均值
    x_test_encoded = encoder.predict(x_test, batch_size=BATCH_SIZE)
    plt.figure()
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
    plt.colorbar()
    plt.title("Mean of the simulated normal distribution of different picture "
              "types")
    plt.show()

    # 观察能否通过控制隐变量的均值来输出特定类别的数字

    # 根据输入的label，生成每个类的模拟均值
    mu = class_mean.predict(np.eye(NUM_CLASSES))
    # 15 * 15个图片
    n = 15
    # 图片形状是 28 * 28
    digit_size = 28
    # 初始化生成的图片
    figure = np.zeros((digit_size * n, digit_size * n))
    # 指定输出数字
    output_digit = 9

    # 用正态分布的分位数来构建隐变量对，可以设想使用当前高斯分布对于出现概率越大的隐变量作
    # 为decoder的输入，对于指定类型生成的图片效果应该越好。即生成的图片集中越往中间对于指定
    # 类型生成的图片越好。
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) + mu[output_digit][1]
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) + mu[output_digit][0]

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_data()
    vae = build_model()
    train_and_predict(vae, x_train, y_train, x_test, y_test)
