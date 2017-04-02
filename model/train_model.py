from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
import collections
import os
import time
import threading

import numpy as np
import tensorflow as tf

import h5_reader
import test

from model import Model, run_epoch, Input, run_epoch_regression


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
flags.DEFINE_bool(
    "convolution", True, "Convolution layer or not"
)
flags.DEFINE_bool(
    "is_stacked", True, "early feature or not"
)

flags.DEFINE_integer(
    "max_c_epoch", 50, "Max classification epoch."
)
flags.DEFINE_integer(
    "max_r_epoch", 80, "Max regression epoch."
)
flags.DEFINE_integer(
    "decay_epoch", 20, "the epoch from this learning rate decay."
)
flags.DEFINE_integer(
    "enqueue_size", 80, "the enqueue_size of the data queue"
)
flags.DEFINE_integer(
    "batch_size", 5, "the batch_size of the training data"
)
flags.DEFINE_integer(
    "filter_size", 3, "the filter_size of the convolution layer"
)
flags.DEFINE_float(
    "learning_rate", 0.01, "init learning rate."
)
flags.DEFINE_float(
    "lr_decay", 0.98, "the learning rate decay."
)
flags.DEFINE_float(
    "keep_prob", 0.6, "the rate of keep prob in dropout"
)
flags.DEFINE_float(
    "momentum", 0.97, 'the strength in momentum'
)
FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class Config(object):
    init_scale = 0.1
    max_grad_norm = 100
    keep_prob = FLAGS.keep_prob
    momentum = FLAGS.momentum

    learning_rate = FLAGS.learning_rate
    lr_decay = FLAGS.lr_decay

    enqueue_size = FLAGS.enqueue_size
    batch_size = FLAGS.batch_size
    num_steps = 8605
    skeleton_size = 75
    output_size = 52
    hidden_size = 100
    num_layers = 3

    #decay_c_epoch = 40
    max_c_epoch = FLAGS.max_c_epoch
    decay_r_epoch = FLAGS.decay_epoch
    max_r_epoch = FLAGS.max_r_epoch
    max_balance = 10

    regular_balance = 0.0001
    baseline = False

    is_stacked = FLAGS.is_stacked
    convolution = FLAGS.convolution

def get_config():
    return Config()

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data directory")
    if not FLAGS.save_path:
        raise ValueError("Must set --save_path to save directory")


    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1

    gpuconfig = tf.ConfigProto(log_device_placement=True)
    gpuconfig.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = Input(config, data=h5_reader.raw_data('train', 'one'))
            with tf.variable_scope("Model", reuse=None):
                m = Model(is_training=True, config=config, inputs=train_input)
            train_cost = tf.summary.scalar("Train classification Cost", m.cost)
            train_accuracy = tf.summary.scalar("Train Accuracy", m.accuracy)
            train_regression_cost = tf.summary.scalar("Train regression cost", m.cost_regression)
            train_lr = tf.summary.scalar("learning rate", m.lr)
            train_bl = tf.summary.scalar("balance", m.bl)
            train_r_cost = tf.summary.scalar("train r cost", m.r_cost)
            train_merged = tf.summary.merge([train_cost, train_regression_cost, train_accuracy, train_lr, train_bl, train_r_cost])

        with tf.name_scope("Valid"):
            valid_input = Input(config, data=h5_reader.raw_data('valid', 'one'))
            with tf.variable_scope("Model", reuse=True):
                mvalid = Model(is_training=False, config=config, inputs=valid_input)
            valid_cost = tf.summary.scalar("Valid Cost", mvalid.cost)
            valid_accuracy = tf.summary.scalar("Valid Accuracy", mvalid.accuracy)
            valid_r_cost = tf.summary.scalar("Valid R-cost", mvalid.r_cost)
            valid_merged = tf.summary.merge([valid_cost, valid_r_cost, valid_accuracy])

        sv = tf.train.Saver(max_to_keep=0, write_version=2)
        writer = tf.summary.FileWriter(logdir=FLAGS.save_path, graph=tf.get_default_graph())

        logfile = open(FLAGS.save_path+'/best_log.txt', 'w+')
        with tf.Session(config=gpuconfig) as session:

            train_coord_enqueue = tf.train.Coordinator()
            train_enqueue_threads = threading.Thread(target=train_input.data_q.enqueue, args=[session, train_coord_enqueue])
            train_enqueue_threads.start()

            valid_coord_enqueue = tf.train.Coordinator()
            valid_enqueue_threads = threading.Thread(target=valid_input.data_q.enqueue, args=[session, valid_coord_enqueue])
            valid_enqueue_threads.start()

            coord_dequeue = tf.train.Coordinator()
            dequeue_threads = tf.train.start_queue_runners(coord=coord_dequeue, sess=session)

            tf.global_variables_initializer().run()
            # sv.restore(session, "../save/result_222/br_model.ckpt-1")

            best_accuracy = 0.0
            for i in range(config.max_c_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.decay_r_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i+1, session.run(m.lr)))
                train_perplexity, train_accuracy = run_epoch(session, writer, m, eval_op=m.train_op, sm_op=train_merged, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity, valid_accuracy = run_epoch(session, writer, mvalid, sm_op=valid_merged)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    print("Best Model: %d; Train Accuracy: %.5f; Valid Accuracy: %.5f;" % (i+1, train_accuracy, valid_accuracy), file=logfile)
                    logfile.flush()
                    sv.save(session, FLAGS.save_path+'/best_br_model.ckpt', global_step=i)

            sv.save(session, FLAGS.save_path+'/br_model.ckpt', global_step=config.max_c_epoch)

            best_f1 = 0.0
            for i in range(config.max_r_epoch):
                lr_decay = config.lr_decay ** max(i+ config.max_c_epoch + 1 - config.decay_r_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                balance_new = (config.max_balance / config.max_r_epoch) * i + 1
                m.assign_balance(session, balance_new)


                print("Epoch: %d Learning rate: %.3f" % (i+1, session.run(m.lr)))
                print("Epoch: %d Learning Balance: %.3f" % (i+1, session.run(m.bl)))
                train_perplexity, train_accuracy, train_f1 = run_epoch_regression(session, writer, m, eval_op=m.train_with_regression_op, sm_op=train_merged, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity, valid_accuracy, valid_f1 = run_epoch_regression(session, writer, mvalid, sm_op=valid_merged)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

                if valid_f1 > best_f1:
                    best_f1 = valid_f1
                    print("Best Model: %d; Train f1: %.5f; Valid f1: %.5f;" % (i+1+config.max_c_epoch, train_f1, valid_f1), file=logfile)
                    print("@ this Model: Train Accuracy: %.5f; Valid Accuracy: %.5f;" % (train_accuracy, valid_accuracy), file=logfile)
                    logfile.flush()
                    sv.save(session, FLAGS.save_path+'/best_ar_model.ckpt', global_step=i+config.max_c_epoch)

                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    print("Best Model: %d; Train Accuracy: %.5f; Valid Accuracy: %.5f;" % (i+1+config.max_c_epoch, train_accuracy, valid_accuracy), file=logfile)
                    sv.save(session, FLAGS.save_path+'/best_ar_model.ckpt', global_step=i+config.max_c_epoch)


                if FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)

            train_coord_enqueue.request_stop()
            session.run(train_input.data_q.queue.close(cancel_pending_enqueues=True))

            valid_coord_enqueue.request_stop()
            valid_coord_enqueue.join([valid_enqueue_threads])
            session.run(valid_input.data_q.queue.close(cancel_pending_enqueues=True))

            coord_dequeue.request_stop()
            coord_dequeue.join(dequeue_threads)

            del(train_input)
            del(valid_input)

if __name__ == "__main__":
    tf.app.run()

