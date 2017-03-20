from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
import collections
import os
import time


import numpy as np
import tensorflow as tf

import reader
import test

from classification import Model, run_epoch


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "data_path", None, "Where the training/test data in stored."
)
flags.DEFINE_string(
    "save_path", None, "Model output directory."
)
flags.DEFINE_string(
    "model_path", None, "Model restore path."
)
flags.DEFINE_bool(
    "use_fp16", False, "Train using 16-bits floats instead of 32bit floats"
)

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class Config(object):
    init_scale = 0.04
    max_grad_norm = 10
    keep_prob = 0.5
    momentum = 0.94

    learning_rate = 0.01
    lr_decay = 0.96

    batch_size = 1
    num_steps = 4000
    input_size = 75
    output_size = 11
    hidden_size = 100
    num_layers = 3

    #decay_c_epoch = 40
    max_c_epoch = 30
    decay_r_epoch = 60
    max_r_epoch = 120
    max_balance = 15

    regular_balance = 0.001
    baseline = False

def get_config():
    return Config()


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")
    if not FLAGS.model_path:
        raise ValueError("Must set --model_path to model restore directory")


    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 4000

    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True

    with tf.Graph().device('/gpu:0'):

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None):
                m = Model(is_training=True, config=config)
            train_cost = tf.summary.scalar("Train classification Cost", m.cost)
            train_accuracy = tf.summary.scalar("Train Accuracy", m.accuracy)
            train_regression_cost = tf.summary.scalar("Train regression cost", m.cost_regression)
            train_lr = tf.summary.scalar("learning rate", m.lr)
            train_bl = tf.summary.scalar("balance", m.bl)
            #train_c_pred = tf.summary.histogram("trian c pred", m.classification_prediction)
            #train_r_pred = tf.summary.histogram("train r pred", m.regression_prediction)
            train_r_cost = tf.summary.scalar("train r cost", m.r_cost)
            train_merged = tf.summary.merge([train_cost, train_regression_cost, train_accuracy, train_lr, train_bl, train_r_cost])

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True):
                mvalid = Model(is_training=False, config=config)
            valid_cost = tf.summary.scalar("Valid Cost", mvalid.cost)
            valid_accuracy = tf.summary.scalar("Valid Accuracy", mvalid.accuracy)
            valid_r_cost = tf.summary.scalar("Valid R-cost", mvalid.r_cost)
            valid_merged = tf.summary.merge([valid_cost, valid_r_cost, valid_accuracy])

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True):
                mtest = Model(is_training=False, config=config)
            test_cost = tf.summary.scalar("Test Cost", mtest.cost)
            test_accuracy = tf.summary.scalar("Test Accuracy", mtest.accuracy)
            test_r_cost = tf.summary.scalar("Test R-cost", mtest.r_cost)
            test_merged = tf.summary.merge([test_cost, test_r_cost, test_accuracy])

    raw_test_data = reader.skeleton_raw_data(FLAGS.data_path, types="test")
    test_padding = reader.padding(raw_test_data, config.num_steps)
    print("test data read!")

    #rs = tf.train.Saver(max_to_keep=0, write_version=2)
    sv = tf.train.Saver(max_to_keep=0, write_version=2)
    writer = tf.summary.FileWriter(logdir=FLAGS.save_path, graph=tf.get_default_graph())

    with tf.Session(config=gpuconfig) as session:
        sv.restore(session, FLAGS.model_path)

        test_perplexity = run_epoch(session, writer, mtest, regression=True, data_padding=test_padding, sm_op=test_merged, verbose=True)
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()

