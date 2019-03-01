#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Practice the usage of feature_columns and tf.contrib.learn.DNNClassifier.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-2-26 上午8:51
"""

# 3rd-part libs.
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn as learn


def _input_fn():
    """Define the input function.

    Returns:
        The input data of feature columns and labels.
    """
    features = {
        "age": tf.constant([[18], [20], [25]]),

        # [['en', 0],
        #  ['fr', 0],
        #  ['zh', 0]]
        "language": tf.SparseTensor(values=['en', 'fr', 'zh'],
                                    indices=[[0, 0], [1, 0], [2, 0]],
                                    dense_shape=[3, 2])
    }
    return features, tf.constant([[1], [0], [2]], dtype=tf.int32)


def test_parse_feature_columns_from_examples():
    """Construct examples by tf.train.Example.
     Then, parse feature columns from examples.
     Finally, get input from feature columns.

    Returns:
        The input tensor transformed from examples in defined feature columns
         format.
    """
    language_column = layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=20)

    feature_columns = [
        layers.embedding_column(language_column, dimension=3),
        layers.real_valued_column("age", dtype=tf.int64)
    ]
    example1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                "age": tf.train.Feature(int64_list=tf.train.Int64List(value=[
                    18])),
                "language": tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[b"en"]))
            }))
    example2 = tf.train.Example(
        features=tf.train.Features(
            feature={
                "age": tf.train.Feature(int64_list=tf.train.Int64List(value=[
                    20])),
                "language": tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[b"fr"]))
            }))
    example3 = tf.train.Example(
        features=tf.train.Features(
            feature={
                "age": tf.train.Feature(int64_list=tf.train.Int64List(value=[
                    25])),
                "language": tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[b"en"]))
            }))
    examples = [example1.SerializeToString(), example2.SerializeToString(),
                example3.SerializeToString()]
    print(examples)
    # feature_lists = tf.train.FeatureLists(
    #     feature_list={
    #         "age": tf.train.FeatureList(
    #             feature=[
    #                 tf.train.Feature(int64_list=tf.train.Int64List(value=[18])),
    #                 tf.train.Feature(int64_list=tf.train.Int64List(value=[20])),
    #                 tf.train.Feature(int64_list=tf.train.Int64List(value=[25])),
    #             ]
    #         ),
    #         "language": tf.train.FeatureList(
    #             feature=[
    #                 tf.train.Feature(bytes_list=tf.train.BytesList(value=[
    #                     b"en"])),
    #                 tf.train.Feature(bytes_list=tf.train.BytesList(value=[
    #                     b"fr"])),
    #                 tf.train.Feature(bytes_list=tf.train.BytesList(value=[
    #                     b"zh"]))
    #             ]
    #         )
    #     }
    # )
    # print(feature_lists)
    # serialized = feature_lists.SerializeToString()

    columns_to_tensor = layers.parse_feature_columns_from_examples(
        serialized=examples,
        feature_columns=feature_columns)
    input_layer = layers.input_from_feature_columns(
        columns_to_tensors=columns_to_tensor,
        feature_columns=feature_columns)
    print("input_layer:\n", str(input_layer))
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run(session=sess)
    print(input_layer.eval(session=sess))


if __name__ == "__main__":
    language_column = layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=20)

    feature_columns = [
        layers.embedding_column(language_column, dimension=3),
        layers.real_valued_column("age", dtype=tf.int64)
    ]

    classifier = learn.DNNClassifier(
        n_classes = 3,
        feature_columns=feature_columns,
        hidden_units=[100, 100],
        config=learn.RunConfig(tf_random_seed=1)
    )
    classifier.fit(input_fn=_input_fn, steps=10000)
    print("variables_names:\n", str(classifier.get_variable_names()))
    scores = classifier.evaluate(input_fn=_input_fn, steps=1)
    print("scores:\n", str(scores))

    # test_parse_feature_columns_from_examples()