#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""Creates auto-draw training and eval data from Quickdraw NDJSON files.

This tool reads the NDJSON files from https://quickdraw.withgoogle.com/data
and converts them into tensorflow.Example stored in TFRecord files.

The tensorflow example will contain 3 features:
 shape - contains the shape of the sequence [length, dim] where dim=3.
 class_index - the class index of the class for the example.
 ink - a length * dim vector of the ink.

It creates disjoint training and evaluation sets.

It creates disjoint dataset for every class.

python create_dataset_for_classify.py \
  --ndjson_path ${HOME}/ndjson \
  --output_path ${HOME}/tfrecord

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2019/09/03
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_line(ndjson_line):
    """Parse an ndjson line and return ink (as np array) and classname."""
    sample = json.loads(ndjson_line)
    class_name = sample["word"]
    if not class_name:
        print("Empty classname")
        return None, None
    inkarray = sample["drawing"]
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    if not inkarray:
        print("Empty inkarray")
        return None, None
    for stroke in inkarray:
        if len(stroke[0]) != len(stroke[1]):
            print("Inconsistent number of x and y coordinates.")
            return None, None
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1  # stroke_end
    # Preprocessing.
    # 1. Size normalization.
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    # 2. Compute deltas.
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]
    np_ink[:, 0: 2] = -np_ink[:, 0: 2]
    np_ink[:, 0: 2] = np.round(np_ink[:, 0: 2], 2)
    return np_ink, class_name


def plot_quick_draw(inks, cls_name):
    """Plot the quick drawing.

    Args:
        inks: The ink deltas array with shape(ink_num, 3). Every delta is (
        x_delta, y_delta, if_end).
        cls_name: The class name.
    """
    print(inks)
    inks_num = inks.shape[0] + 1  # The total inks number.
    plt_ink = np.zeros((inks_num, 3))

    # Convert deltas to plot inks with start point(0, 0).
    for i in range(1, inks_num):
        plt_ink[i, 0: -1] = plt_ink[i - 1, 0: -1] + inks[i - 1, 0: -1]
        plt_ink[i, -1] = inks[i - 1, -1]

    end_points = np.where(plt_ink[:, -1] == 1)[0]  # Find the end points.

    # Plot.
    plt.figure()
    plt.subplot()
    for i, end_pt in enumerate(end_points):
        if i == 0:
            plt.plot(plt_ink[0: end_pt + 1, 0], plt_ink[0: end_pt + 1, 1])
        else:
            plt.plot(plt_ink[end_points[i - 1] + 1: end_pt + 1, 0],
                     plt_ink[end_points[i - 1] + 1: end_pt + 1, 1])
    plt.title(str(cls_name))
    plt.show()


def convert_data(trainingdata_dir,
                 output_path,
                 offset=0):
    """Convert training data from ndjson files into tf.Example in tf.Record.

    Args:
         trainingdata_dir: path to the directory containin the training data.
           The training data is stored in that directory as ndjson files.
         output_path: path where to write the output.
         offset: the number of items to skip at the beginning of each file.
    """
    files = tf.gfile.ListDirectory(trainingdata_dir)
    random.shuffle(files)
    for i, filename in enumerate(files):
        if not filename.endswith(".ndjson"):
            print("Skipping", filename)
            continue
        else:
            print("Converting {0}-{1}".format(i, filename))

        # Prepare tfRecord file writer.
        writer = tf.python_io.TFRecordWriter(output_path + "-" +
                                             filename.split(".")[0]
                                             .replace(" ", "_"))

        # Prepare ndjson file reader.
        reader = tf.gfile.GFile(os.path.join(trainingdata_dir, filename), "r")

        # Parse every line.
        for line in reader:
            ink = None
            while ink is None:
                ink, class_name = parse_line(line)
                if ink is None:
                    print("Couldn't parse ink from '" + line + "'.")
            features = {}
            features["ink"] = tf.train.Feature(float_list=tf.train.FloatList(
                value=ink.flatten()))
            # print("ink: " + str(ink.flatten()))
            features["shape"] = tf.train.Feature(int64_list=tf.train.Int64List(
                value=ink.shape))
            # print("ink.shape: " + str(ink.shape))
            f = tf.train.Features(feature=features)
            example = tf.train.Example(features=f)
            writer.write(example.SerializeToString())

        reader.close()
        writer.close()

        if i % 10 == 0:
            plot_quick_draw(ink, class_name)


def main(argv):
    del argv
    convert_data(
        FLAGS.ndjson_path,
        os.path.join(FLAGS.output_path, "training.tfrecord"),
        offset=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--ndjson_path",
        type=str,
        default="/tmp/gcloud/rnn_tutorial_data",
        help="Directory where the ndjson files are stored.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tmp/autodraw_data",
        help="Directory where to store the output TFRecord files.")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
