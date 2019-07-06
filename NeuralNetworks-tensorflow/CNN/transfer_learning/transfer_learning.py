#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Train a model to classify dog vs cat dataset(https://www.kaggle.com/c/dogs-vs-cats/
data) via transfer learning(feature-extraction or fine-tuning).

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/7/5 下午8:01
"""

# common libs.
import os
import shutil

# 3rd-part libs.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

DOGS_VS_CATS_DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
IMAGE_SIZE = 160
EPOCH_SIZE = 30
BATCH_SIZE = 100
MODEL_PATH = "/tmp/transfer-learning"
SAVER_PATH = "/tmp/transfer-learning/model_saver.h5"
WEIGHTS_PATH = "/tmp/transfer-learning/weights.h5"


def load_dog_vs_cat_datas():
    """Load the training and validation data.

    Returns:
        The training data generator and validation data generator.
    """

    # Downloading and extract data. Path is default "~/.keras/"
    zip_file = keras.utils.get_file(origin=DOGS_VS_CATS_DATA_URL,
                                    fname="cats_and_dogs_filtered.zip",
                                    extract=True)
    # Get root dir name.
    base_dir, _ = os.path.splitext(zip_file)

    # Training and validation images dir.
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")

    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    print('Total training cat images:', len(os.listdir(train_cats_dir)))

    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    print('Total training dog images:', len(os.listdir(train_dogs_dir)))

    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    print('Total validation cat images:', len(os.listdir(validation_cats_dir)))

    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

    # Train data generator.
    train_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255).flow_from_directory(directory=train_dir,
                                               target_size=(IMAGE_SIZE,
                                                            IMAGE_SIZE),
                                               batch_size=BATCH_SIZE,
                                               class_mode="binary")

    # Validation data generator.
    validation_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255).flow_from_directory(directory=validation_dir,
                                               target_size=(IMAGE_SIZE,
                                                            IMAGE_SIZE),
                                               class_mode="binary")

    class_names = sorted(train_generator.class_indices.items(),
                         key=lambda pair: pair[1])
    # {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    class_names = {value: key for key, value in class_names}

    return class_names, train_generator, validation_generator


def build_model(method):
    """Build the transfer leraning model.

    Args:
        method: "feature-extraction" or "fine-tuning".

    Returns:
        The model.
    """
    # Add base model(pre-trained convolutional network) without top layer.
    base_model = keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE,
                                                             IMAGE_SIZE, 3),
                                                include_top=False,
                                                weights="imagenet",
                                                pooling="avg")
    # base_model.summary()

    learing_rate = 1e-4
    if method == "feature-extraction":
        # Only use base model to extract image feature.
        base_model.trainable = False
    elif method == "fine-tuning":
        # Fine tune some layers of base model.
        base_model.trainable = True
        # Fine tune from 100th layer.
        fine_tune = 100
        for layer in base_model.layers[: fine_tune]:
            layer.trainable = False
        learing_rate = 2e-5
    else:
        raise ValueError("Wrong method.")

    # Add classifier layer.
    model = keras.Sequential([
        base_model,
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.summary()

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learing_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

    return model


def train_and_eval(model, train_generator, validation_generator):
    """Training and evaluate the model.

    Args:
        model: The model.
        train_generator: The training dataset generator.
        validation_generator: The validation dataset generator.

    Returns:
        The history of training and evaluation.
    """
    # Define the callback of fit.
    callbacks = [
        # Model saver callback.
        tf.keras.callbacks.ModelCheckpoint(filepath=SAVER_PATH),
        # Tensorboard callback.
        tf.keras.callbacks.TensorBoard(log_dir=MODEL_PATH,
                                       histogram_freq=1)
    ]
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n // BATCH_SIZE,
        epochs=EPOCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // BATCH_SIZE,
        verbose=1,
        workers=4,
        use_multiprocessing=True,
        callbacks=callbacks
    )

    model.save_weights(filepath=WEIGHTS_PATH)

    return history


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
    predict_labels = np.transpose((predict_labels > 0.5).astype(np.float))[0, :]
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

    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)",
                     y=0.02)
    plt.show()


def plot_history(history):
    """Plot the train and validation accuracy.

    Args:
        history: The fit history.
    """
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.figure()
    plt.subplot(211)
    plt.plot(train_acc, label='training accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('The accuracy of model')

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.subplot(212)
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('The loss of model')

    plt.show()


if __name__ == "__main__":
    class_names, train_generator, validation_generator = load_dog_vs_cat_datas()
    print(class_names)

    # Build transfer learning model by feature-extraction.
    # model = build_model(method="feature-extraction")

    # Build transfer learning model by fine-tuning.
    model = build_model(method="fine-tuning")

    # Train and evaluate
    history = train_and_eval(model, train_generator, validation_generator)

    # Plot the history.
    plot_history(history)

    # Test the model
    # Load model
    model_loaded = build_model(method="fine-tuning")
    model_loaded.load_weights(WEIGHTS_PATH)

    # Get test image data.
    test_image_data, test_image_labels = validation_generator.next()
    test_image_data = test_image_data[: 30]

    # Predict.
    predict(model_loaded, test_image_data, test_image_labels, class_names)
