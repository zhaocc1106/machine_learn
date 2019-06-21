#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
This is a simple transfer learning model via tf hub(https://tfhub.dev/).
Copy the codes into colab of google to run, otherwise can't access tf hub.

Need install tf-hub as follow:
pip install -U --pre tf-hub-nightly


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/6/19 下午8:32
"""

# Common libs.
import os
import shutil
import math

# 3rd-part libs.
import numpy as np
import tensorflow_hub as tf_hub
import tensorflow.keras as keras
import matplotlib.pyplot as plt

IMAGE_SHAPE = (224, 224)
PRE_TRAINED_MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2" \
                        "/feature_vector/2"
IMAGE_SIZE = 3670 * 0.8  # 80% data use to training.
NUM_CLASS = 5
EPOCH = 10
BATCH_SIZE = 100
MODEL_PATH = "/tmp/simple_transfer_learning/"
WEIGHTS_PATH = "/tmp/simple_transfer_learning/model_weights.h5"


def load_image_data():
    """Load the flower image datas.

    Returns:
        The image class names and image data generator of training and
        validation.
    """
    data_root = keras.utils.get_file(
        'flower_photos',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
        untar=True)

    # Use 20% image to validate.
    datagen_kwargs = dict(rescale=1. / 255, validation_split=.20)
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        **datagen_kwargs)

    # Load training image data set.
    train_image_data = image_generator.flow_from_directory(str(data_root),
                                                           subset="training",
                                                           target_size=IMAGE_SHAPE,
                                                           batch_size=BATCH_SIZE)

    # Load test image data set.
    vali_image_data = image_generator.flow_from_directory(str(data_root),
                                                          subset="validation",
                                                          target_size=IMAGE_SHAPE,
                                                          batch_size=BATCH_SIZE)

    class_names = sorted(train_image_data.class_indices.items(),
                         key=lambda pair: pair[1])
    # {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    class_names = {value: key for key, value in class_names}
    return class_names, train_image_data, vali_image_data


def build_model(pre_trained_model_url):
    """Build the transfer learning network model.

    Args:
        pre_trained_model: The pre-trained model url of tensorflow hub
        (https://tfhub.dev/s?module-type=image-feature-vector).

    Returns:
        The transfer learning network model.
    """
    # Construct the feature extractor layer via tf hub model.
    feature_extractor_layer = tf_hub.KerasLayer(handle=pre_trained_model_url,
                                                trainable=False,
                                                input_shape=[224, 224, 3])
    model = keras.Sequential([
        feature_extractor_layer,
        keras.layers.Dropout(rate=0.2),
        # Add softmax layer.
        keras.layers.Dense(units=NUM_CLASS,
                           activation="softmax",
                           kernel_regularizer=keras.regularizers.l2(0.0001))
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy"])
    model.summary()
    return model


def train(model, train_image_data, vali_image_data, epoch):
    """Train the model.

    Args:
        model: The network model.
        image_data: The image data generator.
        epoch: The epoch number.
    """
    # Define fit callbacks.
    callbacks = [
        # Tensorboard callback.
        keras.callbacks.TensorBoard(log_dir=MODEL_PATH)
    ]

    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    step_per_epoch = math.ceil(IMAGE_SIZE / BATCH_SIZE)
    model.fit_generator(
        generator=train_image_data,
        epochs=epoch,
        steps_per_epoch=step_per_epoch,
        validation_data=vali_image_data,
        validation_steps=1,
        callbacks=callbacks
    )

    model.save_weights(filepath=WEIGHTS_PATH, overwrite=True)


def predict(model, image_data, image_labels, class_names):
    """Predict the classes of image_data.

    Args:
        model: The model.
        image_data: The test image data.
        image_labels: The true image labels.
        class_names: The class names directory.
    """
    # Predict the labels.
    predict_labels = model.predict(image_data)
    predict_labels = np.argmax(predict_labels, axis=1)
    image_labels = np.argmax(image_labels, axis=1)
    print("Predict labels:\n", str(predict_labels))
    print("True labels:\n", str(image_labels))

    # Plot the image classification results.
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_data[n])
        color = "green" if predict_labels[n] == image_labels[n] else "red"
        plt.title(class_names[predict_labels[n]], color=color)
        plt.axis('off')

    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
    plt.show()


if __name__ == "__main__":
    class_names, train_image_data, vali_image_data = load_image_data()
    print("class names:\n", str(class_names))

    # Build network model.
    model = build_model(PRE_TRAINED_MODEL_URL)

    # Train the model.
    train(model, train_image_data, vali_image_data, EPOCH)

    # Test the model
    # Load model
    model_loaded = build_model(PRE_TRAINED_MODEL_URL)
    model_loaded.load_weights(WEIGHTS_PATH)

    # Get test image data.
    test_image_data, test_image_labels = vali_image_data.next()
    test_image_data = test_image_data[: 30]
    test_image_labels = test_image_labels[: 30]

    # Predict.
    predict(model_loaded, test_image_data, test_image_labels, class_names)
