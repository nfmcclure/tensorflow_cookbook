# -*- coding: utf-8 -*-
# Using TensorFlow Serving (CLIENT)
#----------------------------------
#
# We show how to use "TensorFlow Serving", a model serving api from TensorFlow to serve a model.
#
# Pre-requisites:
#  - Visit https://www.tensorflow.org/serving/setup
#    and follow all the instructions on setting up TensorFlow Serving (including installing Bazel).
#
# The example we will query the TensorFlow-Serving-API we have running on port 9000

import os
import re
import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.flags.DEFINE_string('server', '9000', 'PredictionService host')
tf.flags.DEFINE_string('port', '0.0.0.0', 'PredictionService port')
tf.flags.DEFINE_string('data_dir', 'temp', 'Folder where vocabulary is.')
FLAGS = tf.flags.FLAGS


# Def a functions to process texts into arrays of indices
# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return text_string


# Load vocab processor
def load_vocab():
    vocab_path = os.path.join(FLAGS.data_dir, 'vocab')
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    return vocab_processor


# Process input data:
def process_data(input_data):
    vocab_processor = load_vocab()
    input_data = [clean_text(x) for x in input_data]
    processed_input = np.array(list(vocab_processor.transform(input_data)))
    return processed_input


def get_results(data, server, port):
    channel = grpc.insecure_channel(':'.join([server, port]))
    stub = prediction_service_pb2.PredictionServiceStub(channel)
    processed_data = process_data(data)

    results = []
    for input_x in processed_data:
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'spam_ham'
        request.model_spec.signature_name = 'predict_spam'  # Change to predict spam
        request.inputs['texts'].CopyFrom(tf.contrib.util.make_tensor_proto(input_x, shape=[4, 20]))  # 'texts'
        prediction_future = stub.Predict(request)
        prediction = prediction_future.result().outputs['scores']
        # prediction = np.array(prediction_future.result().outputs['scores'].float_val)
        results.append(prediction)
    return results


def main(data):
    if not FLAGS.server:
        print('please specify server host:port')
        return
    results = get_results(data, FLAGS.server, FLAGS.port)

    for input_text, output_pred in zip(data, results):
        print('Input text: {}, Prediction: {}'.format(input_text, output_pred))


if __name__ == '__main__':
    # Get sample data, here you may feel free to change this to a file, cloud-address, user input, etc...
    test_data = ['Please respond ASAP to claim your prize !',
                 'Hey, are you coming over for dinner tonight?',
                 'Text 444 now to see the top users in your area',
                 'drive safe, and thanks for visiting again!']

    tf.app.run(argv=test_data)
