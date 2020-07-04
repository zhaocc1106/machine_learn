#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
Build univariate time serials prediction model.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2020/6/21 19:32
"""

# common libs.
import os

# 3rd-part libs.
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/' \
           'jena_climate_2009_2016.csv.zip'

TRAINING_SPLIT = 300000
UNIVARIATE_PAST_HISTORY = 20

BUFFER_SIZE = 10000
BATCH_SIZE = 256
RNN_UNITS = 8
RNN_LAYERS = 1
EPOCHS = 20
EVALUATION_INTERVAL = 200

MODEL_PATH = '/tmp/univariate_time_serial/'


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    """Make feature data and label data for univariate time serial.

    Args:
        dataset: The dataset.
        start_index: The start index.
        end_index: The end index.
        history_size: The history data size used to predict.
        target_size: The distance between last feature and target label.

    Returns:
        The feature data and label data.
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = dataset.shape[0] - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], newshape=(history_size, 1)))
        labels.append(dataset[i + target_size])
    return data, labels


def load_univariate_dataset(training_split):
    """Load dataset used to train univariate time serial prediction model.

    Args:
        training_split: The training data number.

    Returns:
        training dataset and evaluation dataset.
    """
    zip_path = tf.keras.utils.get_file(
        origin=DATA_URL,
        fname='jena_climate_2009_2016.csv.zip',
        extract=True
    )
    csv_path = os.path.splitext(zip_path)[0]
    df = pd.read_csv(csv_path)
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    uni_data = uni_data.values
    print('total_len: ', str(len(uni_data)))
    # sub_plot = uni_data.plot(subplots=True)

    # Standardize.
    # Notice: The mean and standard deviation should only be computed by
    # training data.
    uni_train_mean = np.mean(uni_data[:TRAINING_SPLIT])
    uni_train_std = np.std(uni_data[:TRAINING_SPLIT])
    uni_data = (uni_data - uni_train_mean) / uni_train_std

    # Generate training data and label.
    training_data, training_label = univariate_data(uni_data, 0, TRAINING_SPLIT,
                                                    UNIVARIATE_PAST_HISTORY, 0)
    # Generate evaluation data and label.
    eval_data, eval_label = univariate_data(uni_data, TRAINING_SPLIT, None,
                                            UNIVARIATE_PAST_HISTORY, 0)

    # Build tensorflow dataset.
    training_univariate = tf.data.Dataset.from_tensor_slices((training_data,
                                                              training_label))
    training_univariate = training_univariate.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

    eval_univariate = tf.data.Dataset.from_tensor_slices((eval_data,
                                                          eval_label))
    eval_univariate = eval_univariate.cache().batch(
        BATCH_SIZE, drop_remainder=True).repeat()

    return training_univariate, eval_univariate


def uni_model_func(rnn_layers=RNN_LAYERS, rnn_units=RNN_UNITS):
    """Build univariate model by functional api.

    Args:
        rnn_layers: The rnn layers.
        rnn_units: The rnn units.

    Returns:
        The model.
    """
    input = tf.keras.Input(batch_size=None, shape=(20, 1))
    rnn = input
    for i in range(rnn_layers):
        return_seq = True if i < rnn_layers - 1 else False
        rnn = tf.keras.layers.LSTM(units=RNN_UNITS,
                                   return_sequences=return_seq)(rnn)
    dense1 = tf.keras.layers.Dense(units=1)(rnn)
    return tf.keras.Model(input, dense1)


def show_plot(plot_data, delta, title=None):
    """Show plot of features and label.

    Args:
        plot_data: [features_list, true_label_list, predict_label_list]
        delta: The delta between feature and label.
        title: The title.
    """
    labels = ['features', 'true_label', 'predict_label']
    markers = ['.-', 'rx', 'go']

    if title:
        plt.title(title)
    future = delta if delta else 0

    time_steps = list(range(-plot_data[0].shape[0], 0))
    print(time_steps)
    for i, data in enumerate(plot_data):
        if i:
            plt.plot(delta, data, markers[i], markersize=10, \
                     label=labels[i])
        else:
            plt.plot(time_steps, data, markersize=10,
                     label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('time-step')
    plt.show()


def show_history(history, title):
    """Show train history."""
    train_loss = history.history['loss']
    eval_loss = history.history['val_loss']

    epochs = range(len(train_loss))
    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='train_loss')
    plt.plot(epochs, eval_loss, 'g', label='eval_loss')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def train_and_eval(uni_model, training_data, eval_data, epoch=EPOCHS,
                   steps_per_epoch=EVALUATION_INTERVAL):
    """Train and evaluation"""

    # Callbacks.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                MODEL_PATH, 'ckpt/weights.{epoch:02d}-{val_loss:.4f}'),
            save_weights_only=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(MODEL_PATH, 'tensorboard')
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5
        ),
    ]

    # Load latest checkpoint weights.
    try:
        latest_ckpt = tf.train.latest_checkpoint(
            os.path.join(MODEL_PATH, 'ckpt'))
        uni_model.load_weights(filepath=latest_ckpt)
    except Exception as err:
        print(err)

    # Fit.
    history = uni_model.fit(training_data, epochs=epoch,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=eval_data, validation_steps=50,
                            callbacks=callbacks)
    show_history(history, 'epoch-loss')

    # Evaluate.
    metric = uni_model.evaluate(eval_data, steps=50)
    print(metric)


if __name__ == '__main__':
    training_data, eval_data = load_univariate_dataset(TRAINING_SPLIT)
    uni_model = uni_model_func()
    uni_model.summary()
    uni_model.compile(loss=tf.keras.losses.mean_absolute_error,
                      optimizer=tf.keras.optimizers.Adam())
    # Train
    train_and_eval(uni_model, training_data, eval_data)

    # Test.
    (test_data, test_label) = next(iter(eval_data.take(1)))
    features = tf.expand_dims(test_data[0], 0)
    true_label = test_label[0]
    pred_label = uni_model.predict(features)[0]
    print('features: {}, true_label: {}, pred_label: {}'.format(features,
                                                                true_label[0],
                                                                pred_label))
    show_plot(plot_data=[
        np.reshape(features, -1),
        [true_label],
        pred_label],
        delta=0)
