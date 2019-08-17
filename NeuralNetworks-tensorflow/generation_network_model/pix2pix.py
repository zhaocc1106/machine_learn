#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Image to image translation using conditional GAN's,

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/8/15 上午8:47
"""

# common libs.
import os
import time

# 3rd-part libs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
FACADES_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/' \
              'datasets/facades.tar.gz'
GOOGLE_MAPS_DATA_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/" \
                       "pix2pix/datasets/maps.tar.gz"
LAMBDA = 100
MODEL_PATH = "tmp/pix2pix/"
CHECKPOINT_PATH = MODEL_PATH + "training_checkpoint/ckpt"
EPOCHES = 200


def load_image(image_file, is_train):
    """Argument, resize, normalize image.

    Args
        image_file: The image file path.
        is_training: If is training image.

    Returns:
        input_image and real image.
    """
    image = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, w:, :]
    input_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_train:
        # resizing to 286 x 286 x 3
        input_image = tf.image.resize_images(input_image, [286, 286],
                                             align_corners=True,
                                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize_images(real_image, [286, 286],
                                            align_corners=True,
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # randomly cropping to 256 x 256 x 3
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.random_crop(stacked_image,
                                       size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
        input_image, real_image = cropped_image[0], cropped_image[1]

        if np.random.random() > 0.5:
            # random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)
    else:
        input_image = tf.image.resize_images(input_image,
                                             size=[IMG_HEIGHT, IMG_WIDTH],
                                             align_corners=True, method=2)
        real_image = tf.image.resize_images(real_image,
                                            size=[IMG_HEIGHT, IMG_WIDTH],
                                            align_corners=True, method=2)

    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


def load_data(data_type):
    """Load training and test data.

    Args:
        data_type: The dataset type. "facades", "maps".

    Returns:
       training data and test data.
    """
    if data_type == "facades":
        # facades dataset.
        path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                              origin=FACADES_URL,
                                              extract=True)
        path = os.path.join(os.path.dirname(path_to_zip), 'facades/')
    elif data_type == "maps":
        # facades dataset.
        path_to_zip = tf.keras.utils.get_file('maps.tar.gz',
                                              origin=GOOGLE_MAPS_DATA_URL,
                                              extract=True)
        path = os.path.join(os.path.dirname(path_to_zip), 'maps/')

    training_dataset = tf.data.Dataset.list_files(path + "train/*.jpg")
    training_dataset = training_dataset.shuffle(BUFFER_SIZE)
    training_dataset = training_dataset.map(lambda x: load_image(x, True))
    training_dataset = training_dataset.batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(path + "val/*.jpg")
    test_dataset = test_dataset.map(lambda x: load_image(x, False))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return training_dataset, test_dataset


class Downsample(tf.keras.Model):
    """Use convolution layer to downsample"""

    def __init__(self, filters, size, apply_batchnorm=True):
        """The construct function.

        Args:
            filters: The convolution filters number.
            size: The convolution filter size.
            apply_batchnorm If use batch normalization:
        """
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0, 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters=filters,
                                            kernel_size=(size, size),
                                            strides=(2, 2),
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)

        if self.apply_batchnorm:
            self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        """Calls the model on new inputs.

        Args:
            x: The input.
            training: If training.

        Returns:
            output.
        """
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batch_norm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    """Use convolution layer to upsample."""

    def __init__(self, filters, size, apply_dropout=True):
        """The construct function.

        Args:
            filters: The convolution filters number.
            size: The convolution filter size.
            apply_batchnorm If use batch normalization:
        """
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0, 0.02)

        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                     kernel_size=(size, size),
                                                     strides=(2, 2),
                                                     padding="same",
                                                     kernel_initializer=initializer,
                                                     use_bias=False)
        self.batch_normal = tf.keras.layers.BatchNormalization()

        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        """Calls the model on new inputs.

        Args:
            x: The input.
            training: If training.

        Returns:
            output.
        """
        x = self.conv1(x1)
        x = self.batch_normal(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        x = tf.concat([x, x2], axis=-1)
        return x


class Generator(tf.keras.Model):
    """The architecture of generator is a modified U-Net."""

    def __init__(self):
        """The construct function."""
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, apply_batchnorm=False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)
        self.down4 = Downsample(512, 4)
        self.down5 = Downsample(512, 4)
        self.down6 = Downsample(512, 4)
        self.down7 = Downsample(512, 4)
        self.down8 = Downsample(512, 4)

        self.up1 = Upsample(512, 4, apply_dropout=True)
        self.up2 = Upsample(512, 4, apply_dropout=True)
        self.up3 = Upsample(512, 4, apply_dropout=True)
        self.up4 = Upsample(512, 4)
        self.up5 = Upsample(256, 4)
        self.up6 = Upsample(128, 4)
        self.up7 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS,
                                                    (4, 4),
                                                    strides=2,
                                                    padding='same',
                                                    kernel_initializer=initializer)

    @tf.contrib.eager.defun
    def call(self, x, training=True):
        """Calls the model on new inputs.

        Args:
            x: The origin image before translation.
            training: If training.

        Returns:
            The generated image.
        """

        # x shape == (bs, 256, 256, 3)
        x1 = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training)  # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training)  # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training)  # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training)  # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training)  # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training)  # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training)  # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training)  # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training)  # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training)  # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training)  # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training)  # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training)  # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training)  # (bs, 128, 128, 128)

        x16 = self.last(x15)  # (bs, 256, 256, 3)
        x16 = tf.nn.tanh(x16)

        return x16


class Discriminator(tf.keras.Model):
    """The Discriminator is a PatchGAN."""

    def __init__(self):
        """The construct function."""
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)

        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)

    @tf.contrib.eager.defun
    def call(self, inp, tar, training=True):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)

        x = self.zero_pad1(x)  # (bs, 34, 34, 256)
        x = self.conv(x)  # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, 30, 30, 1)

        return x


def discriminator_loss(disc_real_output, disc_gen_output):
    """Construct the discriminator loss. The discriminator will can
    more correctly discriminate the generated image with smaller loss.

    Args:
        disc_real_output: The discriminator output of real image and target
        image.
        disc_gen_output: The discriminator output of generated image and
        target image.

    Returns:
        The discriminator loss.
    """
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(disc_real_output),
        logits=disc_real_output)
    gen_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(disc_gen_output),
        logits=disc_gen_output)
    total_disc_losses = real_loss + gen_loss
    return total_disc_losses


def generator_loss(disc_gen_output, gen_output, target):
    """Construct the generator loss. The generator will be more deceived with
     smaller loss.

    Args:
        disc_gen_output: The discriminator output of generated image and
        target image.
        gen_output: The generated output.
        target: The target image.

    Returns:
        The generator loss.
    """
    gen_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(disc_gen_output),
        logits=disc_gen_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_losses = gen_loss + LAMBDA * l1_loss
    return total_gen_losses


def generate_image(model, test_input, target):
    """Generate image with test input. Show the input image, target image and
     predicted image.

    Args:
        model: The generator model.
        test_input: The test input.
        target: The target image.
    """
    """
    As paper said:
    "At inference time, we run the generator net in exactly the same manner 
    as during the training phase. This differs from the usual protocol in that
    we apply dropout at test time, and we apply batch normalization using the
    statistics of the test batch, rather than aggregated statistics of the 
    training batch. This approach to batch normalization, when the batch size
    is set to 1, has been termed “instance normalization” and has been
    demonstrated to be effective at image generation tasks."
    So training is True.
    """
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    images = [test_input[0], target[0], prediction[0]]
    title = ["Input image", "Target image", "Predicted image"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        image = convert_to_rgb(images[i])
        plt.imshow(image)
        plt.axis("off")  # Hide the axis.
    plt.show()


def train(training_data, test_data, generator, discriminator):
    """Train the condition dcgan.

    Args:
        training_data: The training dataset.
        test_data: The test dataset.
        generator: The generator model.
        discriminator: The discriminator model.
    """
    # We use minibatch SGD and apply the Adam solver [32], with a learning rate
    # of 0.0002, and momentum parameters β 1 = 0.5, β 2 = 0.999.
    # See the paper.
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)

    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     gen_optimizer=gen_optimizer,
                                     disc_optimizer=disc_optimizer)

    for epoch in range(EPOCHES):
        start_time = time.time()
        for input_image, target_image in training_data:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Generate output.
                gen_output = generator(input_image, training=True)

                # Calc output of discriminator.
                disc_real_output = discriminator(input_image, target_image,
                                                 training=True)
                disc_gen_output = discriminator(input_image, gen_output,
                                                training=True)

                # Calc the loss of discriminator and generator.
                disc_loss = discriminator_loss(disc_real_output,
                                               disc_gen_output)
                gen_loss = generator_loss(disc_gen_output, gen_output,
                                          target_image)

            # Calc the gradients of discriminator and generator.
            disc_gradients = disc_tape.gradient(disc_loss,
                                                discriminator.variables)
            gen_gradients = gen_tape.gradient(gen_loss, generator.variables)

            # Apply the gradients.
            disc_optimizer.apply_gradients(zip(disc_gradients,
                                               discriminator.variables))
            gen_optimizer.apply_gradients(zip(gen_gradients,
                                              generator.variables))

        print("[EPOCH {0}] use {1}s, with disc_loss: {2:.5}, gen_loss: {"
              "3:.5}".format(epoch + 1, time.time() - start_time, disc_loss,
                             gen_loss))

        # Save checkpoint.
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PATH)

        # Show the train result.
        for input, real_image in test_data.take(1):
            generate_image(generator, input, real_image)


def convert_to_rgb(input):
    """Convert tensor input(-127.5~127.5) to rgb image(0~1) np array.

    Args:
        input: The input.

    Returns:
        The rgb image np array.
    """
    input = input * 0.5 + 0.5
    input = np.asarray(input)
    for index in np.argwhere(input > 1):
        input.itemset(tuple(index), 1.0)

    for index in np.argwhere(input < 0):
        input.itemset(tuple(index), 0.0)

    return input


if __name__ == "__main__":
    training_data, test_data = load_data("maps")

    """
    iterator = iter(test_data)
    input_image, real_image = next(iterator)
    input_image = convert_to_rgb(input_image[0])
    real_image = convert_to_rgb(real_image[0])
    plt.figure()
    plt.imshow(input_image)
    plt.figure()
    plt.imshow(real_image)
    plt.show()
    """

    generator = Generator()
    discriminator = Discriminator()
    train(training_data, test_data, generator, discriminator)
