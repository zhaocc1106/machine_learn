#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Train a pix2pix model for photo and sketch translation.
Run in tf_2.2.0

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/8/15 上午8:47
"""

# common libs.
import os
import time
import random

# 3rd-part libs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

BUFFER_SIZE = 100
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
# ZIP_PATH = "../datas/sketch_photo.zip"
DATA_PATH = "/home/zhaocc/sources/kaggle/photo2sketch/"
LAMBDA = 100
MODEL_PATH = "/tmp/pix2pix/sketch_photo"
CHECKPOINT_DIR = MODEL_PATH + "/training_checkpoint/"
CHECKPOINT_PATH = CHECKPOINT_DIR + "ckpt"
GENERATE_OUT_PATH = MODEL_PATH + "/generate_out/"
EPOCHES = 200


def py_load_image(image_file):
    """Argument, resize, normalize image.

    Args
        image_file: The image file path.
        is_training: If is training image.

    Returns:
        origin image and target image.
    """
    image_name = image_file.numpy().decode("utf-8")
    origin_image_path = os.path.join(DATA_PATH, 'photos_transparent',
                                     image_name)
    # print("py_load_image image_file:", str(origin_image_path))

    # Add random background color into image.
    origin_image = Image.open(origin_image_path)
    rand_back_color = (random.randint(0, 255), random.randint(0, 255),
                       random.randint(0, 255))
    back_image = Image.new('RGB', size=origin_image.size, color=rand_back_color)
    back_image.paste(origin_image, (0, 0, origin_image.size[0],
                                    origin_image.size[1]), origin_image)
    origin_image = np.asarray(back_image)

    # Find the target sketch image path.
    target_image_path = os.path.join(
        DATA_PATH, 'sketches',
        image_name.replace('-trans', '').split(".")[0] + "-sz1.jpg"
    )

    # print("py_load_image target_image_path:", str(target_image_path))
    target_image = cv2.imread(target_image_path)
    target_image = target_image[..., ::-1]  # bgr to rgb

    origin_image = origin_image.astype(np.float32)
    target_image = target_image.astype(np.float32)

    # normalizing the images to [-1, 1]
    origin_image = (origin_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1

    return origin_image, target_image


def resize_image(origin_image, target_image):
    """Resize image.

    Args:
        origin_image: The origin image.
        target_image: The target image.

    Returns:
        resized origin image and target image.
    """
    # resizing to 256 x 256 x 3
    origin_image.set_shape([None, None, None])
    target_image.set_shape([None, None, None])
    origin_image = tf.image.resize(origin_image, [IMG_HEIGHT, IMG_WIDTH],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [IMG_HEIGHT, IMG_WIDTH],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return origin_image, target_image


def augment_image(origin_image, target_image):
    """Argument image.

    Args:
        origin_image: The origin image.
        target_image: The target image.

    Returns:
        augmented origin image and target image.
    """
    # resizing to 286 x 286 x 3
    origin_image.set_shape([None, None, None])
    target_image.set_shape([None, None, None])
    origin_image = tf.image.resize(origin_image, [286, 286],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, [286, 286],
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    stacked_image = tf.stack([origin_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image,
                                         size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    origin_image, target_image = cropped_image[0], cropped_image[1]

    if np.random.random() > 0.5:
        # random mirroring
        origin_image = tf.image.flip_left_right(origin_image)
        target_image = tf.image.flip_left_right(target_image)

    return origin_image, target_image


def load_data():
    """Load training and test data.

    Returns:
       training data and test data.
    """
    # data_utils.extract_archive(ZIP_PATH, DATA_PATH)

    photos = tf.constant(value=os.listdir(DATA_PATH + "photos_transparent"))
    training_dataset = tf.data.Dataset.from_tensor_slices(photos)
    training_dataset = training_dataset.map(
        lambda filename: tuple(
            tf.py_function(func=py_load_image, inp=[filename],
                           Tout=[tf.float32, tf.float32])))
    training_dataset = training_dataset.map(augment_image)
    training_dataset = training_dataset.shuffle(BUFFER_SIZE)
    training_dataset = training_dataset.batch(BATCH_SIZE)

    photos = tf.constant(value=os.listdir(DATA_PATH + "photos_transparent"))
    eval_dataset = tf.data.Dataset.from_tensor_slices(photos)
    eval_dataset = eval_dataset.map(
        lambda filename: tuple(
            tf.py_function(py_load_image, [filename],
                           [tf.float32, tf.float32])))
    eval_dataset = eval_dataset.map(resize_image)
    eval_dataset = eval_dataset.shuffle(BUFFER_SIZE)
    eval_dataset = eval_dataset.batch(BATCH_SIZE)

    return training_dataset, eval_dataset


def downsample(filters, size, apply_batchnorm=True):
    """Use convolution layer to downsample

    Args:
        filters: The filters number.
        size: The kernel size.
        apply_batchnorm: If apply batch normalization.

    Returns:
        The downsample model.
    """
    initializer = tf.random_normal_initializer(0, 0.02)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=(size, size),
                                     strides=(2, 2),
                                     padding='same',
                                     kernel_initializer=initializer,
                                     use_bias=False))
    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    return model


def upsample(filters, size, apply_dropout=True):
    """Use convolution layers to upsample.

    Args:
        filters: The filters number.
        size: The kernel size.
        apply_dropout: If apply dropout.

    Returns:
        The upsample model.
    """
    initializer = tf.random_normal_initializer(0, 0.02)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(filters=filters,
                                              kernel_size=(size, size),
                                              strides=(2, 2),
                                              padding="same",
                                              kernel_initializer=initializer,
                                              use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.ReLU())
    return model


def Generator():
    """The generator model."""
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=True),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model.
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling through the model.
    for (up, skip) in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])  # Skip connect.

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    """Build the discriminator model by PatchGAN."""
    inp = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                name='input_image')
    tar = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                name='target_image')

    x = tf.keras.layers.Concatenate()([inp, tar])  # (bs, 256, 256, 3 * 2)

    initializer = tf.random_normal_initializer(0., 0.02)

    x = downsample(64, 4, apply_batchnorm=True)(x)  # (bs, 128, 128, 64)
    x = downsample(128, 4)(x)  # (bs, 64, 64, 128)
    x = downsample(256, 4)(x)  # (bs, 32, 32, 236)

    x = tf.keras.layers.ZeroPadding2D()(x)  # (bs, 34, 34, 256)
    x = tf.keras.layers.Conv2D(512, (4, 4), strides=1,
                               kernel_initializer=initializer,
                               use_bias=False)(x)  # (bs, 31, 31, 512)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ZeroPadding2D()(x)  # (bs, 33, 33, 512)

    # (bs, 30, 30, 1)
    x = tf.keras.layers.Conv2D(1, (4, 4), strides=1,
                               kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=x)


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
    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        y_true=tf.ones_like(disc_gen_output),
        y_pred=disc_gen_output
    )

    # Mean absolute loss.
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_losses = gen_loss + LAMBDA * l1_loss
    return total_gen_losses, gen_loss, l1_loss


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
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_obj(
        y_true=tf.ones_like(disc_real_output),
        y_pred=disc_real_output
    )
    gen_loss = loss_obj(
        y_true=tf.zeros_like(disc_gen_output),
        y_pred=disc_gen_output
    )
    total_disc_losses = real_loss + gen_loss
    return total_disc_losses


@tf.function
def train_step(generator, discriminator, input_image, target_image,
               gen_optimizer, disc_optimizer):
    """Train one step.

    Args:
        generator: The generator.
        discriminator: The discriminator.
        input_image: The input image.
        target_image: The target image.
        gen_optimizer: The generator optimizer.
        disc_optimizer: The discriminator optimizer.

    Returns:
        The generator loss and discriminator loss.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate output.
        gen_output = generator(input_image, training=True)

        # Calc output of discriminator.
        disc_real_output = discriminator([input_image, target_image],
                                         training=True)
        disc_gen_output = discriminator([input_image, gen_output],
                                        training=True)

        # Calc the loss of discriminator and generator.
        disc_loss = discriminator_loss(disc_real_output,
                                       disc_gen_output)
        total_gen_losses, gen_loss, l1_loss = generator_loss(disc_gen_output,
                                                             gen_output,
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
    return total_gen_losses, gen_loss, l1_loss, disc_loss


def train(training_data, eval_data, generator, discriminator):
    """Train the condition dcgan.

    Args:
        training_data: The training dataset.
        eval_data: The test dataset.
        generator: The generator model.
        discriminator: The discriminator model.
    """
    # We use minibatch SGD and apply the Adam solver [32], with a learning rate
    # of 0.0002, and momentum parameters β 1 = 0.5, β 2 = 0.999.
    # See the paper.
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4,
                                             beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     gen_optimizer=gen_optimizer,
                                     disc_optimizer=disc_optimizer)

    # Load models if exists.
    if os.path.exists(CHECKPOINT_DIR):
        checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))

    # Tensorboard.
    summary_writer = tf.summary.create_file_writer(MODEL_PATH)

    total_step = 0
    with summary_writer.as_default():
        for epoch in range(EPOCHES):
            start_time = time.time()
            for step, (input_image, target_image) in enumerate(training_data):
                total_step += 1

                if total_step % 100 == 0:
                    tf.summary.trace_on(graph=True)

                total_gen_losses, gen_loss, l1_loss, disc_loss = train_step(
                    generator, discriminator, input_image, target_image,
                    gen_optimizer, disc_optimizer
                )

                tf.summary.scalar('total_gen_losses', total_gen_losses,
                                  step=total_step)
                tf.summary.scalar('gen_loss', gen_loss, step=total_step)
                tf.summary.scalar('l1_loss', l1_loss, step=total_step)
                tf.summary.scalar('discriminator_loss', disc_loss,
                                  step=total_step)

                if total_step % 100 == 0:
                    tf.summary.trace_export(name='photo2sketch',
                                            step=total_step)
            print("[EPOCH {0}] use {1}s, with disc_loss: {2:.5}, gen_loss: {"
                  "3:.5}".format(epoch + 1, time.time() - start_time, disc_loss,
                                 total_gen_losses))

            # Show the training gain by training data.
            for input, real_image in eval_data.take(1):
                show_training_gain(generator, input, real_image,
                                   file_path=os.path.join(
                                       GENERATE_OUT_PATH,
                                       'train-epoch-{}.jpg'.format(epoch + 1)))
            # Show the training gain by test data.
            test(generator, os.path.join(DATA_PATH, 'test'),
                 epoch + 1)

            # Save checkpoint.
            if (epoch + 1) % 10 == 0:
                checkpoint.save(file_prefix=CHECKPOINT_PATH)
    summary_writer.close()

    # Save total generator model.
    tf.keras.models.save_model(
        generator,
        filepath=os.path.join(MODEL_PATH, 'model_saver'),
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        options=None
    )


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


def show_training_gain(model, test_input, target=None, file_path=None):
    """Generate image with test input. Show the input image, target image and
     predicted image.

    Args:
        model: The generator model.
        test_input: The test input.
        target: The target image.
        file_path: The file path to save generated img.
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
    prediction = model(test_input, training=False)

    if target is not None:
        images = [test_input[0], target[0], prediction[0]]
        title = ["Input image", "Target image", "Predicted image"]
    else:
        images = [test_input[0], prediction[0]]
        title = ["Input image", "Predicted image"]

    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title[i])
        image = convert_to_rgb(images[i])
        plt.imshow(image)
        plt.axis("off")  # Hide the axis.
    plt.savefig(file_path)
    # plt.show()


def test(generator, eval_dir, epoch):
    """Generate images from test data.

    Args:
        generator: The generator model.
        eval_dir: The test dataset dir.
    """
    files = os.listdir(eval_dir)
    for i, file_name in enumerate(files):
        photo = cv2.imread(os.path.join(eval_dir, file_name))
        photo = photo[..., ::-1]
        photo = photo.astype(np.float32)
        photo = (photo / 127.5) - 1

        photo = tf.constant(photo)
        photo = tf.image.resize(photo,
                                [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        photo = tf.reshape(photo, (1, IMG_HEIGHT, IMG_WIDTH, 3))

        show_training_gain(generator, photo, target=None,
                           file_path=os.path.join(
                               GENERATE_OUT_PATH,
                               'epoch-{}-eval-{}.jpg'.format(epoch, i)))


def generate_sketch(generator, photo_dir, ckpt_dir=None, ckpt_path=None,
                    flag=None):
    """Generate sketch for photos.

    Args:
        generator: The generator.
        photo_dir: The photo dir.
        ckpt_dir: The checkpoint dir. If not none, will choose the latest
        checkpoint.
        ckpt_path: The checkpoint file path. If not none, will choose the
        specific checkpoint.
        flag: The flag into image name.
    """
    # Restore model.
    checkpoint = tf.train.Checkpoint(generator=generator)
    if ckpt_path is not None:
        checkpoint.restore(ckpt_path)
    elif ckpt_dir is not None and os.path.exists(ckpt_dir):
        checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

    files = os.listdir(photo_dir)
    for i, file_name in enumerate(files):
        photo = cv2.imread(os.path.join(photo_dir, file_name))
        photo = photo[..., ::-1]  # bgr ==> rgb
        photo = photo.astype(np.float32)
        photo = (photo / 127.5) - 1
        origin_h = photo.shape[0]
        origin_w = photo.shape[1]
        origin_h_w_rate = float(origin_h) / float(origin_w)

        photo = tf.constant(photo)
        photo = tf.image.resize(photo,
                                [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        photo = tf.reshape(photo, (1, IMG_HEIGHT, IMG_WIDTH, 3))

        prediction = generator(photo, training=False)
        # Resize to origin size.
        prediction = tf.image.resize(prediction,
                                     size=(int(origin_h_w_rate * 300), 300),
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image = convert_to_rgb(prediction[0])
        height = image.shape[0]
        width = image.shape[1]
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=100)
        axes = fig.add_axes([0, 0, 1, 1])
        axes.set_axis_off()
        axes.imshow(image)
        fig.savefig(GENERATE_OUT_PATH + 'predict-sketch-{}-{}.jpg'.format(
            flag, i))
        fig.show()


if __name__ == "__main__":
    training_data, eval_data = load_data()

    """
    iterator = iter(eval_data)
    input_image, real_image = next(iterator)
    input_image = convert_to_rgb(input_image[0])
    real_image = convert_to_rgb(real_image[0])
    plt.figure()
    plt.imshow(input_image)
    plt.figure()
    plt.imshow(real_image)
    plt.show()
    """

    if not os.path.exists(GENERATE_OUT_PATH):
        os.mkdir(GENERATE_OUT_PATH)

    generator = Generator()
    discriminator = Discriminator()
    generator.summary()
    discriminator.summary()
    tf.keras.utils.plot_model(model=generator, show_shapes=True, dpi=64,
                              to_file='genertor.png')
    tf.keras.utils.plot_model(model=discriminator, show_shapes=True, dpi=64,
                              to_file='discriminator.png')

    train(training_data, eval_data, generator, discriminator)
    generate_sketch(generator, os.path.join(DATA_PATH, 'test'),
                    CHECKPOINT_DIR, CHECKPOINT_PATH + '-41', '41')
