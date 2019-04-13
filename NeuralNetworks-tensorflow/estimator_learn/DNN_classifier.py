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
import tensorflow.contrib.framework as framework
import tensorflow.contrib.metrics as metrics
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec


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
    return features, tf.constant([[1], [0], [2]], dtype=tf.int64)


def parse_feature_columns_from_examples_test():
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


def my_metric_op(predictions, targets):
    """Define my metric operation.

    Args:
        predictions: The predictions of dnn classifier.
        targets: The targets.

    Returns:
        The correct predicts number.
    """
    print(predictions.shape)
    return tf.reduce_sum(tf.multiply(predictions, targets))


def optimizer_exp_decay():
    """Construct the optimizer with learning rate decay every experience.

    Returns:
        The optimizer.
    """
    global_step = framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(learning_rate=0.1,
                                               global_step=global_step,
                                               decay_steps=100,
                                               decay_rate=0.001)
    return tf.train.AdagradOptimizer(learning_rate=learning_rate)


def contrib_learn_classifier_test():
    """Test tf.contrib.learn.DNN_classifier."""
    language_column = layers.sparse_column_with_hash_bucket(
        "language", hash_bucket_size=20)

    feature_columns = [
        layers.embedding_column(language_column, dimension=3),
        layers.real_valued_column("age", dtype=tf.int64)
    ]

    classifier = learn.DNNClassifier(
        n_classes=3,
        feature_columns=feature_columns,
        hidden_units=[100, 100],
        config=learn.RunConfig(tf_random_seed=1,
                               model_dir="../model_saver/estimators/"
                                         "DNN_classifier_01"),
        # optimizer=optimizer_exp_decay
    )
    classifier.fit(input_fn=_input_fn, steps=10000)
    print("variables_names:\n", str(classifier.get_variable_names()))
    # scores = classifier.evaluate(input_fn=_input_fn,
    #                              steps=100)
    # print("scores:\n", str(scores))

    scores = classifier.evaluate(
        input_fn=_input_fn,
        steps=100,
        metrics={
            'my_accuracy': MetricSpec(
                metric_fn=metrics.streaming_accuracy,
                prediction_key="classes"),
            'my_precision': MetricSpec(
                metric_fn=metrics.streaming_precision,
                prediction_key="classes"),
            'my_recall': MetricSpec(
                metric_fn=metrics.streaming_recall,
                prediction_key="classes"),
            'my_metric': MetricSpec(
                metric_fn=my_metric_op,
                prediction_key="classes")
        })
    print("scores:\n", str(scores))

    predictions = classifier.predict(input_fn=_input_fn, outputs=["classes",
                                                                  "probabilities"])
    print("predictions")
    for prediction in predictions:
        print(prediction)


def estimator_classifier_test():
    """Test tf.estimator.DNN_classifier."""
    language_cloumn = tf.feature_column.categorical_column_with_hash_bucket(
        key="language",
        hash_bucket_size=3)
    age_column = tf.feature_column.numeric_column(key="age")

    feature_columns = [tf.feature_column.embedding_column(language_cloumn,
                                                          dimension=3),
                       age_column]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[100, 100],
        n_classes=3,
        config=tf.estimator.RunConfig(
            model_dir="../model_saver/estimators/DNN_classifer_02",
            tf_random_seed=1
        )
    )

    classifier.train(input_fn=_input_fn, steps=10000)

    scores = classifier.evaluate(input_fn=_input_fn, steps=100)
    print("scores:\n", str(scores))

    predictions = classifier.predict(input_fn=_input_fn, predict_keys=[
        "probabilities", "classes"])
    print("predictions:")
    for (i, prediction) in enumerate(predictions):
        if i > 2:
            break
        print(prediction)


if __name__ == "__main__":
    # parse_feature_columns_from_examples_test()

    # contrib_learn_classifier_test()

    estimator_classifier_test()
