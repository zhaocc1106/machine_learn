#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""The tools to download and read words data used by RNN training.


Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2018/12/5 21:17
"""
# The common libs.
import urllib.request as req
import os
import zipfile
import collections

# 3rd part libs.
import tensorflow as tf

url = "http://mattmahoney.net/dc/"

def maybe_download(filename):
    """Download the file from url.

    :param filename: The file name.
    """
    dir_path = "../datas/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = dir_path + filename
    if not os.path.exists(file_path):
        filename, _ = req.urlretrieve(url + filename, file_path)
        print("Download complete.")
    else:
        print("File exist.")


def read_data(filename):
    """Read words array from file.

    :param filename: The file name.
    :return:
        The data array.
    """
    file_path = "../datas/" + filename
    with zipfile.ZipFile(file_path) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, vocabulary_size=5000):
    """Build data set from words.

    :param words:
    :param vocabulary_size: The vocabulary size.
    :return:
        count: The dictionary of ```(word, count)```.
        data: The number list of every word.
        dictionary: The dictionary of ```(word, number)```.
        reverse_dictionary: The dictionary of ```(number, word)```.

    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # Number these words based on the number of these words.
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    unk_count = 0
    data = list()
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count # Update the count of unknown words.

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, data, dictionary, reverse_dictionary


if __name__ == "__main__":
    maybe_download("text8.zip")
    words = read_data("text8.zip")
    print("words len:\n", len(words))
    vocabulary_size = 50000
    count, data, dicionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    print("count:\n", str(count[:10]))
    print("data:\n", str(data[:10]))
    print([reverse_dictionary[index] for index in data[:10]])
    # print("dictionary:\n", str(dicionary))
    # print("reverse_dictionary:\n", str(reverse_dictionary))
    del words