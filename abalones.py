from __future__ import absolute_import, division, print_function

import sys
import tempfile

from six.moves import urllib

import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def maybe_download():
    train_data = single_download(data=None, url="http://download.tensorflow.org/data/abalone_train.csv")
    test_data = single_download(data=None, url="http://download.tensorflow.org/data/abalone_test.csv")
    predict_data = single_download(data=None, url="http://download.tensorflow.org/data/abalone_predict.csv")
    return train_data, test_data, predict_data


def single_download(data, url):
    if data:
        data_file_name = data
    else:
        data_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve(url, data_file.name)
        data_file_name = data_file.name
        data_file.close()
        print("Data in url {} downloaded to {}".format(url, data_file_name))
    return data_file_name


def model_fn(features, labels, mode, params):
    first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 10, activation=tf.nn.relu)
    output_layer = tf.layers.dense(second_hidden_layer, 1)

    # reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])

    # predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"ages": predictions}
        )

    loss = tf.losses.mean_squared_error(labels, predictions)

    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float64), predictions
        )
    }

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"]
    )
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step()
    )

    # eval and train modes
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


if __name__ == "__main__":
    abalone_train, abalone_test, abalone_predict = maybe_download()
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

    model_params = {"learning_rate": 0.001}
    nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

    # train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True
    )

    nn.train(input_fn=train_input_fn, steps=5000)

    # test
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False
    )
    
    ev = nn.evaluate(input_fn=test_input_fn)
    print("Loss {}".format(ev["loss"]))
    print("Root Mean Squared Error {}".format(ev["rmse"]))

    # predict
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(prediction_set.data)},
        num_epochs=1,
        shuffle=False
    )
    predictions = nn.predict(input_fn=predict_input_fn)
    for i, p in enumerate(predictions):
        print("predictions {} {}".format(i, p["ages"]))
