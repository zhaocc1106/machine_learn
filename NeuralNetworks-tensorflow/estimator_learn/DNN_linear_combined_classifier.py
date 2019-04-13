#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
Practice the DNNLinearCombinedClassifier.

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    19-3-11 下午8:25
"""

# common libs.
import os
import shutil

# 3rd-part libs.
import tensorflow as tf
import absl.flags as flags
from absl import app as absl_app

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def define_flags():
    """Define the flags."""
    flags.DEFINE_enum(name="model_type",
                      short_name="mt",
                      default="deep_wide",
                      enum_values=['wide', 'deep', 'deep_wide'],
                      help="Select model topology.")

    flags.DEFINE_string(name="data_dir",
                        short_name="dd",
                        default="/tmp/census_data",
                        help="Input data dir.")

    flags.DEFINE_integer(name="train_epochs",
                         default=40,
                         help="Input train epochs")

    flags.DEFINE_integer(name="epochs_between_evals",
                         default=2,
                         help="Input the number of train epochs between "
                              "evaluations.")

    flags.DEFINE_integer(name="batch_size",
                         default=40,
                         help="Input the batch size of training and "
                              "evaluation.")

    flags.DEFINE_multi_integer(name="deep_widths",
                               default=[100, 75, 50, 25],
                               help="Input the layer widths of dnn.")

    flags.DEFINE_string(name="model_dir",
                        short_name="md",
                        default="/tmp/census_model",
                        help="Input model dir.")


def build_features():
    """Build the features.

    Returns:
        wide features and deep features.
    """
    # Continuous columns
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    # Categorical columns
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', [
            'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
            'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
            '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', [
            'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
            'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship', [
            'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
            'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass', [
            'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

    # To show an example of hashing:
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        'occupation', hash_bucket_size=1000)

    # Transformations.
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    base_columns = [
        education, marital_status, relationship, workclass, occupation,
        age_buckets,
    ]

    # Crossed columns.
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column(
            [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        # Dnn only accept dense columns.
        # Transform categorical columns to indicator columns.
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        # Transform categorical columns to embedding columns.
        tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    return wide_columns, deep_columns


def build_estimator():
    """Build the estimator according to flags.

    Returns:
        return the estimator.
    """
    # Build features.
    wide_columns, deep_columns = build_features()

    flags_obj = flags.FLAGS
    model_type = flags_obj.model_type
    model_dir = flags_obj.model_dir
    layer_widths = flags_obj.deep_widths
    print("build_estimator model_type: {0} model_dir: {1} layer_widths: {2}"
          "".format(model_type, model_dir, layer_widths))

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    if model_type == "wide":
        return tf.estimator.LinearClassifier(
            feature_columns=wide_columns,
            model_dir=model_dir,
            n_classes=2,
            config=run_config
        )
    elif model_type == "deep":
        return tf.estimator.DNNClassifier(
            feature_columns=deep_columns,
            model_dir=model_dir,
            n_classes=2,
            config=run_config,
            hidden_units=layer_widths
        )
    elif model_type == "deep_wide":
        return tf.estimator.DNNLinearCombinedClassifier(
            dnn_feature_columns=deep_columns,
            linear_feature_columns=wide_columns,
            model_dir=model_dir,
            n_classes=2,
            config=run_config,
            dnn_hidden_units=layer_widths
        )


def input_function(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator.

    Args:
        data_file: The data file path.
        num_epochs: The number of train epochs.
        shuffle: If shuffle the data.
        batch_size: The batch size.

    Returns:
        The dataset of features and label columns.
    """
    assert tf.gfile.Exists(data_file), (
            '%s not found. Please make sure you have run census_data'
            '_download.py and set the --data_dir argument to the correct path.'
            % data_file)

    def parse_csv(line):
        print("Parsing ", data_file)
        # Decode one line.
        columns = tf.decode_csv(records=line,
                                record_defaults=_CSV_COLUMN_DEFAULTS)
        # Build features dict.
        features_dict = dict(zip(_CSV_COLUMNS, columns))
        # Pop the label feature.
        label = features_dict.pop("income_bracket")
        return features_dict, tf.equal(label, '>50K')

    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(map_func=parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
    dataset = dataset.batch(batch_size)
    return dataset


def run(estimator):
    """Run the estimator to train model and evaluate model.

    Args:
        estimator: The estimator.

    Returns:

    """
    flags_obj = flags.FLAGS
    epochs_between_evals = flags_obj.epochs_between_evals
    train_epochs = flags_obj.train_epochs
    data_dir = flags_obj.data_dir
    batch_size = flags_obj.batch_size
    model_dir = flags_obj.model_dir
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    def train_input_fn():
        return input_function(os.path.join(data_dir, "adult.data"),
                              epochs_between_evals, True, batch_size)

    def test_input_fn():
        return input_function(os.path.join(data_dir, "adult.test"),
                              1, False, batch_size)

    """
    # Test data set.
    dataset = train_input_fn()
    # Create iterator.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.InteractiveSession()
    # Run initializer.
    sess.run(iterator.initializer)

    # Iter.
    while True:
        try:
            print("next element:")
            data = sess.run(next_element)
            print(data)
        except tf.errors.OutOfRangeError:
            print("out of range.")
            break
    """

    for i in range(train_epochs // epochs_between_evals):
        # Train the estimator.
        estimator.train(input_fn=train_input_fn)

        # Evaluate the estimator.
        scores = estimator.evaluate(input_fn=test_input_fn)
        print("Epoch({0}), scores:{1}"
              .format((i + 1) * epochs_between_evals, scores))


def main(_):
    estimator = build_estimator()
    run(estimator)
    pass


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_flags()
    absl_app.run(main)
