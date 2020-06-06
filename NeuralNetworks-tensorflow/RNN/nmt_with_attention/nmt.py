#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
Build neural machine translation model with attention. Use it to train a
model to translation english and chinese.
Run in tensorflow 2.1.0

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2020/6/6 下午5:03
"""

# common libs.
import os
import time
import re
import io
import unicodedata

# 3rd-part libs.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

print(tf.__version__)

# Download dataset of chinese-english. Download from
# http://www.manythings.org/anki/ directly because can't access google storage.
# path_to_zip = tf.keras.utils.get_file(
#     'cmn-eng.zip',
#     origin='http://storage.googleapis.com/download.tensorflow.org/data/cmn-eng'
#            '.zip',
#     extract=True)
path_to_file = '/home/zhaocc/.keras/datasets/cmn-eng/cmn.txt'


def unicode_to_ascii(s):
    """Converts from unicode file to ascii."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if
                   unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    """Preprocess sentence."""
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,？。！，、])", r" \1 ", w)
    w = re.sub(r'[" "“”]+', " ", w)  # replace quotation marks to space.

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",",
    # "。", "？", "！", "，", "、", "所有汉字")
    w = re.sub(r"[^a-zA-Z\u4e00-\u9fa5?.!,？。！，、]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples):
    """Create dataset.

    Args:
        path: The file path.
        num_examples: The total examples number.

    Returns:
        word pairs in the format: [ENGLISH, CHINESE]
    """
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    words_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l
                   in lines[:num_examples]]
    return zip(*words_pairs)


if __name__ == '__main__':
    en_sentence = u'The large crowd roared in approval as Mark Knopfler ' \
                  u'played ' \
                  'the first few bars of "Money for Nothing".'
    ch_sentence = u'就像马克·诺弗勒早期演唱的歌曲《金钱无用》一样，绝大多数的' \
                  u'人依然高呼赞成“金钱无用论”。'
    print(preprocess_sentence(en_sentence))
    print(preprocess_sentence(ch_sentence))

    en, ch = create_dataset(path_to_file, 20000)
    print(en[-1])
    print(ch[-1])
