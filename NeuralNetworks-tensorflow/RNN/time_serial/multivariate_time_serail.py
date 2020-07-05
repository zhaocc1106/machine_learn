#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
Build multivariate time serial prediction model.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2020/7/4 下午4:34
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
MULTIVARIATE_PAST_HISTORY = 720  # 5 days features data.
STEP = 6  # 1 hour
FUTURE_TARGET = 72  # 1 day target data.

BUFFER_SIZE = 10000
BATCH_SIZE = 256
RNN_UNITS = [32, 16]
RNN_LAYERS = 2
EPOCHS = 20
EVALUATION_INTERVAL = 200

MODEL_PATH = '/tmp/multivariate_time_serial'


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    """Make feature data and label data for multivariate time serial.

    Args:
        dataset: The features dataset.
        target: The target dataset.
        start_index: The start index.
        end_index: The end index.
        history_size: The history data size used to predict.
        target_size: The distance between last feature and target label.
        step: The feature step.
        single_step: If target is single step.

    Returns:
        The feature data and label data.
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = dataset.shape[0] - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i: i + target_size])
    return np.array(data), np.array(labels)


def load_multivariate_dataset(training_split, single_step=False,
                              multivariate_past_history=MULTIVARIATE_PAST_HISTORY,
                              step=STEP,
                              future_target=FUTURE_TARGET):
    """Load dataset used to train univariate time serial prediction model.

    Args:
        training_split: The training data number.
        single_step: If target is single step.
        multivariate_past_history: The past history data size.
        step: The step of features data.
        future_target: The future target size.

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
    multi_data = df[['p (mbar)', 'T (degC)', 'rho (g/m**3)']]
    multi_data.index = df['Date Time']
    multi_data = multi_data.values
    print('total_len: ', str(len(multi_data)))
    # sub_plot = multi_data.plot(subplots=True)

    # Standardize.
    # Notice: The mean and standard deviation should only be computed by
    # training data.
    uni_train_mean = np.mean(multi_data[:training_split], axis=0)
    uni_train_std = np.std(multi_data[:training_split], axis=0)
    multi_data = (multi_data - uni_train_mean) / uni_train_std

    # Generate training data and label.
    training_data, training_label = multivariate_data(multi_data,
                                                      multi_data[:, 1],
                                                      0,
                                                      training_split,
                                                      multivariate_past_history,
                                                      future_target,
                                                      step=step,
                                                      single_step=single_step)
    print(training_data.shape)
    print(training_label.shape)
    # Generate evaluation data and label.
    eval_data, eval_label = multivariate_data(multi_data,
                                              multi_data[:, 1],
                                              training_split, None,
                                              multivariate_past_history,
                                              future_target,
                                              step=step,
                                              single_step=single_step)
    print(eval_data.shape)
    print(eval_label.shape)

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


def multi_model_func(rnn_layers=RNN_LAYERS, rnn_units=RNN_UNITS,
                     single_step=False, target_size=FUTURE_TARGET):
    """Build multivariate model by functional api.

    Args:
        rnn_layers: The rnn layers.
        rnn_units: The rnn units.
        single_step: If the target is single step.

    Returns:
        The model.
    """
    input = tf.keras.Input(batch_size=None,
                           shape=(int(MULTIVARIATE_PAST_HISTORY / STEP), 3))
    conv1 = tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1,
                                   activation='relu', padding='same')(input)
    conv1 = tf.keras.layers.Dropout(rate=0.5)(conv1)
    rnn = conv1
    for i in range(rnn_layers):
        return_seq = True if i < rnn_layers - 1 else False
        rnn = tf.keras.layers.LSTM(units=rnn_units[i],
                                   return_sequences=return_seq,
                                   activation='relu' if i else 'tanh'
                                   )(rnn)
    dense1 = tf.keras.layers.Dense(units=1 if single_step else target_size)(rnn)
    return tf.keras.Model(input, dense1)


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


def show_plot(plot_data, delta, title=None, single_stpe=False):
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

    feature_time_steps = list(range(-plot_data[0].shape[0], 0))
    print('feature_time_steps: ', str(feature_time_steps))
    target_time_steps = \
        list(np.array(range(delta, delta + plot_data[1].shape[0])) / STEP) \
            if single_stpe else list(np.array(range(0, delta)) / STEP)
    print('target_time_steps: ', str(target_time_steps))
    for i, data in enumerate(plot_data):
        if i:
            plt.plot(target_time_steps, data, markers[i], markersize=5, \
                     label=labels[i])
        else:
            plt.plot(feature_time_steps, data, markersize=5,
                     label=labels[i])
    plt.legend()
    plt.xlabel('time-step')
    plt.show()


def train_and_eval(multi_model, training_data, eval_data, epoch=EPOCHS,
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
        multi_model.load_weights(filepath=latest_ckpt)
    except Exception as err:
        print(err)

    # Fit.
    history = multi_model.fit(training_data, epochs=epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=eval_data, validation_steps=50,
                              callbacks=callbacks)
    show_history(history, 'epoch-loss')

    # Evaluate.
    metric = multi_model.evaluate(eval_data, steps=50)
    print(metric)


if __name__ == '__main__':
    model_path = MODEL_PATH

    """Single step prediction."""
    MODEL_PATH = model_path + '_single_step'
    training_data, eval_data = load_multivariate_dataset(TRAINING_SPLIT,
                                                         single_step=True)
    model = multi_model_func(rnn_layers=1, single_step=True)
    tf.keras.utils.plot_model(model,
                              to_file='multivariate_time_serial_single_step.png',
                              show_shapes=True)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.mean_absolute_error)

    # Train.
    train_and_eval(model, training_data, eval_data, EPOCHS, EVALUATION_INTERVAL)

    # Prediction.
    for (test_data, test_label) in eval_data.take(3):
        features = tf.expand_dims(test_data[0], 0)
        true_label = test_label[0]
        pred_label = model.predict(features)[0]
        print('features: {}, true_label: {}, pred_label: {}'.format(
            features[:, :, 1],
            true_label,
            pred_label))
        show_plot(
            plot_data=[np.reshape(features[:, :, 1], -1),
                       np.array([true_label]),
                       np.array(pred_label)],
            delta=FUTURE_TARGET,
            single_stpe=True)

    """Multi step prediction."""
    MODEL_PATH = model_path + '_multi_step'
    training_data, eval_data = load_multivariate_dataset(TRAINING_SPLIT)
    model = multi_model_func(rnn_layers=2)
    tf.keras.utils.plot_model(model,
                              to_file='multivariate_time_serial_multi_step.png',
                              show_shapes=True)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

    # Train.
    train_and_eval(model, training_data, eval_data, EPOCHS, EVALUATION_INTERVAL)

    # Prediction.
    for (test_data, test_label) in eval_data.take(3):
        features = tf.expand_dims(test_data[0], 0)
        true_label = test_label[0]
        pred_label = model.predict(features)[0]
        print('features: {}, true_label: {}, pred_label: {}'.format(
            features[:, :, 1],
            true_label,
            pred_label))
        show_plot(
            plot_data=[np.reshape(features[:, :, 1], -1),
                       np.array(true_label),
                       np.array(pred_label)],
            delta=FUTURE_TARGET)
