#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""Binary for trianing a RNN-based classifier for the Quick, Draw! data.

python train_model.py \
  --training_data train_data \
  --eval_data eval_data \
  --model_dir /tmp/quickdraw_model/ \
  --cell_type cudnn_lstm

When running on GPUs using --cell_type cudnn_lstm is much faster.

The expected performance is ~75% in 1.5M steps with the default configuration.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/09/03
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import ast
import functools
import sys
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_num_classes():
    classes = []
    with tf.gfile.GFile(FLAGS.classes_file, "r") as f:
        classes = [x for x in f]
    num_classes = len(classes)
    return num_classes


def get_input_fn(mode, tfrecord_pattern, batch_size):
    """Creates an input_fn that stores all the data in memory.

    Args:
     mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
     tfrecord_pattern: path to a TF record file created using create_dataset.py.
     batch_size: the batch size to output.

    Returns:
      A valid input_fn for the model estimator.
    """

    def _parse_tfexample_fn(example_proto, mode):
        """Parse a single record which is expected to be a tensorflow.Example."""
        feature_to_type = {
            "ink": tf.VarLenFeature(dtype=tf.float32),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64)
        }
        """
        if mode != tf.estimator.ModeKeys.PREDICT:
          # The labels won't be available at inference time, so don't add them
          # to the list of feature_columns to be read.
          feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)
        """
        feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

        parsed_features = tf.parse_single_example(example_proto,
                                                  feature_to_type)
        labels = None
        """
        if mode != tf.estimator.ModeKeys.PREDICT:
          labels = parsed_features["class_index"]
        """
        labels = parsed_features["class_index"]
        parsed_features["ink"] = tf.sparse_tensor_to_dense(
            parsed_features["ink"])
        return parsed_features, labels

    def _input_fn():
        """Estimator `input_fn`.

        Returns:
          A tuple of:
          - Dictionary of string feature name to `Tensor`.
          - `Tensor` of target labels.
        """
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.repeat()
        # Preprocesses 10 files concurrently and interleaves records from each file.
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        # Our inputs are variable length, so pad them.
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=dataset.output_shapes)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn


def model_fn(features, labels, mode, params):
    """Model function for RNN classifier.

    This function sets up a neural network which applies convolutional layers (as
    configured with params.num_conv and params.conv_len) to the input.
    The output of the convolutional layers is given to LSTM layers (as configured
    with params.num_layers and params.num_nodes).
    The final state of the all LSTM layers are concatenated and fed to a fully
    connected layer to obtain the final classification scores.

    Args:
      features: dictionary with keys: inks, lengths.
      labels: one hot encoded classes
      mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
      params: a parameter dictionary with the following keys: num_layers,
        num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.

    Returns:
      ModelFnOps for Estimator API.
    """

    def _get_input_tensors(features, labels):
        """Converts the input dict into inks, lengths, and labels tensors."""
        # features[ink] is a sparse tensor that is [8, batch_maxlen, 3]
        # inks will be a dense tensor of [8, maxlen, 3]
        # shapes is [batchsize, 2]
        shapes = features["shape"]
        # lengths will be [batch_size]
        lengths = tf.squeeze(
            tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1]))
        inks = tf.reshape(features["ink"], [params.batch_size, -1, 3])
        if labels is not None:
            labels = tf.squeeze(labels)
        return inks, lengths, labels

    def _add_conv_layers(inks, lengths):
        """Adds convolution layers."""
        convolved = inks
        for i in range(len(params.num_conv)):
            convolved_input = convolved
            if params.batch_norm:
                convolved_input = tf.layers.batch_normalization(
                    convolved_input,
                    training=(mode == tf.estimator.ModeKeys.TRAIN))
            # Add dropout layer if enabled and not first convolution layer.
            if i > 0 and params.dropout:
                convolved_input = tf.layers.dropout(
                    convolved_input,
                    rate=params.dropout,
                    training=(mode == tf.estimator.ModeKeys.TRAIN))
            convolved = tf.layers.conv1d(
                convolved_input,
                filters=params.num_conv[i],
                kernel_size=params.conv_len[i],
                activation=None,
                strides=1,
                padding="same",
                name="conv1d_%d" % i)
        return convolved, lengths

    def _add_regular_rnn_layers(convolved, lengths):
        """Adds RNN layers."""
        if params.cell_type == "lstm":
            cell = tf.nn.rnn_cell.BasicLSTMCell
        elif params.cell_type == "block_lstm":
            cell = tf.contrib.rnn.LSTMBlockCell
        cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
        if params.dropout > 0.0:
            cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in
                        cells_fw]
            cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in
                        cells_bw]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=convolved,
            sequence_length=lengths,
            dtype=tf.float32,
            scope="rnn_classification")
        return outputs

    def _add_cudnn_rnn_layers(convolved):
        """Adds CUDNN LSTM layers."""
        # Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
        convolved = tf.transpose(convolved, [1, 0, 2])
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=params.num_layers,
            num_units=params.num_nodes,
            dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
            direction="bidirectional")
        outputs, _ = lstm(convolved)
        # Convert back from time-major outputs to batch-major outputs.
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs

    def _add_rnn_layers(convolved, lengths):
        """Adds recurrent neural network layers depending on the cell type."""
        if params.cell_type != "cudnn_lstm":
            outputs = _add_regular_rnn_layers(convolved, lengths)
        else:
            outputs = _add_cudnn_rnn_layers(convolved)
        # outputs is [batch_size, L, N] where L is the maximal sequence length and N
        # the number of nodes in the last layer.

        # "lengths", "outputs" -> "mask" used to select the true outputs of
        # every inks with different length.
        # mask shape is [Dimension(8), Dimension(max_len), Dimension(256)]
        mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
            [1, 1, tf.shape(outputs)[2]])
        zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        outputs = tf.reduce_sum(zero_outside, axis=1)
        return outputs

    def _add_fc_layers(final_state):
        """Adds a fully connected layer."""
        return tf.layers.dense(final_state, params.num_classes)

    # Build the model.
    inks, lengths, labels = _get_input_tensors(features, labels)
    convolved, lengths = _add_conv_layers(inks, lengths)
    final_state = _add_rnn_layers(convolved, lengths)
    logits = _add_fc_layers(final_state)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.argmax(logits, axis=1)
        probs = tf.reduce_max(tf.nn.softmax(logits), axis=1)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"logits": logits,
                         "predictions": predictions,
                         "probs": probs,
                         "inks": inks})

    # Add the loss.
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
    # Add the optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer="Adam",
        # some gradient clipping stabilizes training in the beginning.
        clip_gradients=params.gradient_clipping_norm,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
    # Compute current predictions.
    predictions = tf.argmax(logits, axis=1)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"logits": logits, "predictions": predictions},
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)})


def create_estimator_and_specs(run_config):
    """Creates an Experiment configuration based on the estimator and input fn."""
    model_params = tf.contrib.training.HParams(
        num_layers=FLAGS.num_layers,
        num_nodes=FLAGS.num_nodes,
        batch_size=FLAGS.batch_size,
        num_conv=ast.literal_eval(FLAGS.num_conv),
        conv_len=ast.literal_eval(FLAGS.conv_len),
        num_classes=get_num_classes(),
        learning_rate=FLAGS.learning_rate,
        gradient_clipping_norm=FLAGS.gradient_clipping_norm,
        cell_type=FLAGS.cell_type,
        batch_norm=FLAGS.batch_norm,
        dropout=FLAGS.dropout)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params)

    train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        tfrecord_pattern=FLAGS.training_data,
        batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        tfrecord_pattern=FLAGS.eval_data,
        batch_size=FLAGS.batch_size))

    return estimator, train_spec, eval_spec


def plot_quick_draw(inks, cls_name, *sub_plt_place):
    """Plot the quick drawing.

    Args:
        inks: The ink deltas array with shape(ink_num, 3). Every delta is (
        x_delta, y_delta, if_end).
        cls_name: The class name.
        sub_plt_place: The subplot place.
    """
    inks_num = inks.shape[0] + 1  # The total inks number.
    plt_ink = np.zeros((inks_num, 3))

    # Convert deltas to plot inks with start point(0, 0).
    for i in range(1, inks_num):
        plt_ink[i, 0: -1] = plt_ink[i - 1, 0: -1] + inks[i - 1, 0: -1]
        plt_ink[i, -1] = inks[i - 1, -1]

    end_points = np.where(plt_ink[:, -1] == 1)[0]  # Find the end points.

    # Plot.
    print(*sub_plt_place)
    plt.subplot(*sub_plt_place)
    # print(end_points)
    for i, end_pt in enumerate(end_points):
        if i == 0:
            plt.plot(-plt_ink[0: end_pt + 1, 0], -plt_ink[0: end_pt + 1, 1])
        else:
            plt.plot(-plt_ink[end_points[i - 1] + 1: end_pt + 1, 0],
                     -plt_ink[end_points[i - 1] + 1: end_pt + 1, 1])
    plt.axis('off')
    plt.title(str(cls_name))


def predict(estimator, num):
    """Predict.

    Args:
        estimator: The model estimator.
        num: The prediction number.
    """
    preds = estimator.predict(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.PREDICT,
        tfrecord_pattern=FLAGS.pred_data,
        batch_size=FLAGS.batch_size))

    plt_columns = 5
    plt_rows = math.ceil(num / 5)

    with tf.gfile.GFile(FLAGS.classes_file, "r") as f:
        class_names = [x for x in f]

        plt.figure(figsize=(10, 10))
        for i, pred in enumerate(preds):
            plot_quick_draw(pred['inks'],
                            "{0}({1:.2%})".format(
                                str(class_names[pred['predictions']]),
                                pred['probs']),
                            plt_rows,
                            plt_columns,
                            (i + 1))
            if i >= num - 1:
                break
        plt.show()


def main(unused_args):
    estimator, train_spec, eval_spec = create_estimator_and_specs(
        run_config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=300,
            save_summary_steps=100))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print("######################Train end.")
    predict(estimator, 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--training_data",
        type=str,
        default="\\tmp\\quickdraw_data\\training.tfrecord-*",
        help="Path to training data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--eval_data",
        type=str,
        default="\\tmp\\quickdraw_data\\eval.tfrecord-*",
        help="Path to evaluation data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--pred_data",
        type=str,
        default="\\tmp\\quickdraw_data\\eval.tfrecord-*",
        help="Path to prediction data (tf.Example in TFRecord format)")
    parser.add_argument(
        "--classes_file",
        type=str,
        default="\\tmp\\quickdraw_data\\training.tfrecord.classes",
        help="Path to a file with the classes - one class per line")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of recurrent neural network layers.")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=128,
        help="Number of node per recurrent network layer.")
    parser.add_argument(
        "--num_conv",
        type=str,
        default="[48, 64, 96]",
        help="Number of conv layers along with number of filters per layer.")
    parser.add_argument(
        "--conv_len",
        type=str,
        default="[5, 5, 3]",
        help="Length of the convolution filters.")
    parser.add_argument(
        "--cell_type",
        type=str,
        default="cudnn_lstm",
        help="Cell type used for rnn layers: cudnn_lstm, lstm or block_lstm.")
    parser.add_argument(
        "--batch_norm",
        type="bool",
        default="False",
        help="Whether to enable batch normalization or not.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate used for training.")
    parser.add_argument(
        "--gradient_clipping_norm",
        type=float,
        default=9.0,
        help="Gradient clipping norm used during training.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout used for convolutions and bidi lstm layers.")
    parser.add_argument(
        "--steps",
        type=int,
        default=1000000,
        help="Number of training steps.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size to use for training/evaluation.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="\\tmp\\quickdraw_model",
        help="Path for storing the model checkpoints.")
    parser.add_argument(
        "--self_test",
        type="bool",
        default="False",
        help="Whether to enable batch normalization or not.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
