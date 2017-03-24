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
    momentum = 0.97

    learning_rate = 0.01
    lr_decay = 0.96

    batch_size = 1
    num_steps = 4000
    input_size = 75
    output_size = 11
    hidden_size = 100
    num_layers = 3

    #decay_c_epoch = 40
    max_c_epoch = 80
    decay_r_epoch = 60
    max_r_epoch = 120
    max_balance = 15

    regular_balance = 0.001
    baseline = False

    is_stacked = True
    convolution = True

def get_config():
    return Config()


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")


    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 4000

    gpuconfig = tf.ConfigProto(log_device_placement=True)
    gpuconfig.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
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
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Model(is_training=False, config=config)
            valid_cost = tf.summary.scalar("Valid Cost", mvalid.cost)
            valid_accuracy = tf.summary.scalar("Valid Accuracy", mvalid.accuracy)
            valid_r_cost = tf.summary.scalar("Valid R-cost", mvalid.r_cost)
            valid_merged = tf.summary.merge([valid_cost, valid_r_cost, valid_accuracy])

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = Model(is_training=False, config=config)
            test_cost = tf.summary.scalar("Test Cost", mtest.cost)
            test_accuracy = tf.summary.scalar("Test Accuracy", mtest.accuracy)
            test_r_cost = tf.summary.scalar("Test R-cost", mtest.r_cost)
            test_merged = tf.summary.merge([test_cost, test_r_cost, test_accuracy])

        raw_train_data = reader.skeleton_raw_data(FLAGS.data_path, types="train")
        train_padding = reader.padding(raw_train_data, config.num_steps)
        print("train data read!")
        raw_valid_data = reader.skeleton_raw_data(FLAGS.data_path, types="valid")
        valid_padding = reader.padding(raw_valid_data, config.num_steps)
        print("valid data read!")
        raw_test_data = reader.skeleton_raw_data(FLAGS.data_path, types="test")
        test_padding = reader.padding(raw_test_data, config.num_steps)
        print("test data read!")

        #rs = tf.train.Saver(max_to_keep=0, write_version=2)
        sv = tf.train.Saver(max_to_keep=0, write_version=2)
        writer = tf.summary.FileWriter(logdir=FLAGS.save_path, graph=tf.get_default_graph())

        logfile = open(FLAGS.save_path+'/best_log.txt', 'w+')
        best_accuracy = 0.0
        with tf.Session(config=gpuconfig) as session:
            init = tf.global_variables_initializer().run()
            # sv.restore(session, "../save/result_222/br_model.ckpt-1")

            for i in range(config.max_c_epoch):
                #lr_decay = config.lr_decay ** max(i + 1 - config.decay_c_epoch, 0.0)
                lr_decay = 1.0
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i+1, session.run(m.lr)))
                train_perplexity, train_accuracy = run_epoch(session, writer, m, regression=False, data_padding=train_padding, eval_op=m.train_op, sm_op=train_merged)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity, valid_accuracy = run_epoch(session, writer, mvalid, regression=False, data_padding=valid_padding, sm_op=valid_merged)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                test_perplexity, test_accuracy = run_epoch(session, writer, mtest, regression=False, data_padding=test_padding, sm_op=test_merged)
                print("Test Perplexity: %.3f" % test_perplexity)

                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    print("Best Model: %d; Train Accuracy: %.5f; Valid Accuracy: %.5f; Test Accuracy: %.5f" % (i+1, train_accuracy, valid_accuracy, test_accuracy), file=logfile)
                    sv.save(session, FLAGS.save_path+'/best_br_model.ckpt', global_step=i)

            sv.save(session, FLAGS.save_path+'/br_model.ckpt', global_step=config.max_c_epoch)

            for i in range(config.max_r_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.decay_r_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                balance_new = (config.max_balance / config.max_r_epoch) * i + 1
                m.assign_balance(session, balance_new)

                best_f1 = 0.0

                print("Epoch: %d Learning rate: %.3f" % (i+1, session.run(m.lr)))
                print("Epoch: %d Learning Balance: %.3f" % (i+1, session.run(m.bl)))
                train_perplexity, train_accuracy, train_f1 = run_epoch(session, writer, m, regression=True, data_padding=train_padding, eval_op=m.train_with_regression_op, sm_op=train_merged)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity, valid_accuracy, valid_f1 = run_epoch(session, writer, mvalid, regression=True, data_padding=valid_padding, sm_op=valid_merged)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
                test_perplexity, test_accuracy, test_f1 = run_epoch(session, writer, mtest, regression=True, data_padding=test_padding, sm_op=test_merged)
                print("Test Perplexity: %.3f" % test_perplexity)

                if valid_f1 > best_f1:
                    best_f1 = valid_f1
                    print("Best Model: %d; Train f1: %.5f; Valid f1: %.5f; Test f1: %.5f" % (i+1+config.max_c_epoch, train_f1, valid_f1, test_f1), file=logfile)
                    sv.save(session, FLAGS.save_path+'/best_ar_model.ckpt', global_step=i+config.max_c_epoch)

                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    print("Best Model: %d; Train Accuracy: %.5f; Valid Accuracy: %.5f; Test Accuracy: %.5f" % (i+1+config.max_c_epoch, train_accuracy, valid_accuracy, test_accuracy), file=logfile)
                    sv.save(session, FLAGS.save_path+'/best_ar_model.ckpt', global_step=i+config.max_c_epoch)


                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    #sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
    tf.app.run()

