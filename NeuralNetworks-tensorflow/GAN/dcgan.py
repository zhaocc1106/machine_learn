#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Deep Convolutional Generative Adversarial Network模型。
使用fashion mnist dataset训练一个深度卷积生成对抗网络，用于生成更多fashion mnist数据。
需要使用tensorflow 2.0版本：
pip install tensorflow-gpu==2.0.0-beta1

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/6/26 上午9:07
"""
# common libs.
import os
import shutil
import time

# 3rd-part libs.
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
import imageio

# Total count of fashion mnist data.
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCH_SIZE = 80
# The dimension of noise data to generate image.
NOISE_DIM = 100
MODEL_PATH = "/tmp/dcgan/"
CHECK_POINT_PATH = MODEL_PATH + "training_checkpoint.ckpt"

if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
os.makedirs(MODEL_PATH)


def load_fashion_mnist_dataset():
    """Load fashion mnist image dataset.

    Returns:
        The training fashion mnist image dataset.
    """
    # Just use train images data.
    (train_images, _), (_, _) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1) \
        .astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    train_images = tf.data.Dataset.from_tensor_slices(train_images).shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)
    return train_images


def build_dcgan_model():
    """Build the deep convolutional generative adversarial network model.

    Returns:
        The generator and discriminator.
    """

    """
    Generator model.
    Convert random vector with shape (100,) to (28, 28, 1) image.
    """
    generator = keras.models.Sequential(name="generator")
    generator.add(keras.layers.Dense(units=7 * 7 * 256, use_bias=False,
                                     input_shape=(100,)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    generator.add(keras.layers.Reshape(target_shape=(7, 7, 256)))
    assert generator.output_shape == (None, 7, 7, 256)

    generator.add(keras.layers.Conv2DTranspose(filters=128,
                                               kernel_size=(5, 5),
                                               strides=(1, 1),
                                               padding="same",
                                               use_bias=False))
    assert generator.output_shape == (None, 7, 7, 128)
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    generator.add(keras.layers.Conv2DTranspose(filters=64,
                                               kernel_size=(5, 5),
                                               strides=(2, 2),
                                               padding='same',
                                               use_bias=False))
    assert generator.output_shape == (None, 14, 14, 64)
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    generator.add(keras.layers.Conv2DTranspose(filters=1,
                                               kernel_size=(5, 5),
                                               strides=(2, 2),
                                               padding='same',
                                               use_bias=False,
                                               activation='tanh'))
    assert generator.output_shape == (None, 28, 28, 1)

    """
    Discriminator model.
    Convert image with shape(28, 28, 1) to label(true image or fake image). 
    """
    discriminator = tf.keras.Sequential(name="discriminator")
    discriminator.add(keras.layers.Conv2D(filters=64,
                                          kernel_size=(5, 5),
                                          strides=(2, 2),
                                          padding='same',
                                          input_shape=[28, 28, 1]))
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(0.3))

    discriminator.add(keras.layers.Conv2D(filters=128,
                                          kernel_size=(5, 5),
                                          strides=(2, 2),
                                          padding='same'))
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(0.3))

    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(1))

    generator.summary()
    discriminator.summary()

    return generator, discriminator


def define_loss(real_output, fake_output):
    """Define the generator and discriminator loss.

    Args:
        real_output: The output of discriminator with real image input.
        fake_output: The output of discriminator with fake input(generator
        output).

    Returns:
        The generator loss and discriminator loss.
    """
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

    # Let generator classify if the image is true image.
    discriminator_loss = cross_entropy(tf.ones_like(real_output),
                                       real_output) + \
                         cross_entropy(tf.zeros_like(fake_output), fake_output)

    # Generator fake out the discriminator in order to let it believe the image
    # generated from generator is true image.
    generator_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return generator_loss, discriminator_loss


@tf.function
def train_step(generator, discriminator, generator_optimizer,
               discriminator_optimizer, images_batch):
    """One step to train the generator and discriminator model.

    Args:
        generator: The generator model.
        discriminator: The discriminator model.
        generator_optimizer: The generator gradient optimizer.
        discriminator_optimizer: The discriminator gradient optimizer.
        images_batch: Images of one batch.
    """
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_grad, tf.GradientTape() as dis_grad:
        # Generate the image from generator as fake input of discriminator.
        generated_images = generator(noise, training=True)

        # Infer the real output of discriminator with real image input.
        real_output = discriminator(images_batch, training=True)

        # Infer the fake output of discriminator with fake image input.
        fake_output = discriminator(generated_images, training=True)

        # Infer the generator loss and discriminator loss.
        gen_loss, dis_loss = define_loss(real_output, fake_output)

    fake_success = tf.cast(fake_output >= 0.5, tf.float32)
    # Fake success number.
    fake_success_num = tf.reduce_sum(fake_success)
    # Fake success rate.
    fake_success_rate = fake_success_num / fake_output.shape[0]

    # Discriminator success number.
    disc_success_num = tf.reduce_sum(tf.cast(real_output >= 0.5,
                                             tf.float32)) + \
                       (fake_output.shape[0] - fake_success_num)
    # Discriminator accuracy.
    disc_accuracy = disc_success_num / \
                    (real_output.shape[0] + fake_output.shape[0])

    gen_vars = []
    for var in generator.trainable_variables:
        gen_vars.append((var.name, var))

    disc_vars = []
    for var in discriminator.trainable_variables:
        disc_vars.append((var.name, var))

    step_result = {"gen_loss": gen_loss,
                   "dis_loss": dis_loss,
                   "fake_success_rate": fake_success_rate,
                   "disc_accuracy": disc_accuracy,
                   "generator_vars": gen_vars,
                   "disc_vars": disc_vars}

    # Calc the gradients of generator variables.
    grad_of_generator = gen_grad.gradient(target=gen_loss,
                                          sources=generator.trainable_variables)
    # Calc the gradients of discriminator variables.
    grad_of_discriminator = dis_grad.gradient(target=dis_loss,
                                              sources=discriminator.trainable_variables)

    # Apply the gradients.
    generator_optimizer.apply_gradients(zip(grad_of_generator,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(grad_of_discriminator,
                                                discriminator.trainable_variables))
    return step_result


def generate_and_save_images(model, epoch, test_input):
    """Generate the image from generator.

    Args:
        model: The generator model.
        epoch: Current epoch id.
        test_input: The test input.
    """
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(MODEL_PATH + 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train(generator, discriminator, dataset, epochs):
    """Train the dcgan by dataset.

    Args:
        generator: The generator model.
        discriminator: The discriminator model.
        dataset: The training dataset.
        epochs: The epochs.
    """
    # Define the optimizer.
    gen_optimizer = tf.keras.optimizers.Adam(1e-4)
    dis_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Define the checkpoint.
    checkpoint = tf.train.Checkpoint(gen_optimizer=gen_optimizer,
                                     dis_optimizer=dis_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # Define the tensorboard.
    tf_summary_writer = tf.summary.create_file_writer(MODEL_PATH)

    # Generate random seed as test input of generator.
    seed = tf.random.normal(shape=[16, NOISE_DIM])

    for epoch in range(epochs):
        start = time.time()

        for n, image_batch in enumerate(dataset):
            step_result = train_step(generator,
                                     discriminator,
                                     gen_optimizer,
                                     dis_optimizer,
                                     image_batch)

        # Produce images for the GIF as we go
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Show info.
        print("\n\n")
        print("The fake success rate for epoch {0}: {1:.2%}".format(
            epoch, step_result["fake_success_rate"].numpy()))
        print("The discriminator accuracy for epoch {0}: {1:.2%}:".format(
            epoch, step_result["disc_accuracy"].numpy()))
        print("The generator loss for epoch {0}: {1:.5}".format(
            epoch, step_result["gen_loss"].numpy()))
        print("The discriminator loss for epoch {0}: {1:.5}".format(
            epoch, step_result["dis_loss"].numpy()))

        # Tensorboard.
        with tf_summary_writer.as_default():
            # Scalars.
            tf.summary.scalar("fake_success_rate:",
                              step_result["fake_success_rate"],
                              step=epoch)
            tf.summary.scalar("discriminator_accuracy:",
                              step_result["disc_accuracy"],
                              step=epoch)
            tf.summary.scalar("generator_loss",
                              step_result["gen_loss"],
                              step=epoch)
            tf.summary.scalar("discriminator_loss",
                              step_result["dis_loss"],
                              step=epoch)

            # Variables distribution.
            def vars_summary(scope, vars):
                """Save the vars summary."""
                for var in vars:
                    tf.summary.histogram(
                        name=scope + "/" + var[0].numpy().decode("utf-8"),
                        data=var[1].numpy(),
                        step=epoch)
            vars_summary("generator", step_result["generator_vars"])
            vars_summary("discriminator", step_result["disc_vars"])

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=CHECK_POINT_PATH)

        print('Time for epoch {} is {:.2} sec'.format(epoch + 1,
                                                      time.time() - start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             seed)

    tf_summary_writer.close()


def create_gif():
    """Create gif from image of every epoch."""
    anim_file = MODEL_PATH + 'dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(MODEL_PATH + 'image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


if __name__ == "__main__":
    # Load dataset.
    train_images = load_fashion_mnist_dataset()
    # Build dcgan model.
    generator, discriminator = build_dcgan_model()
    # Train the model.
    train(generator, discriminator, train_images, EPOCH_SIZE)
    # Create gif using generated image of every epoch.
    create_gif()
