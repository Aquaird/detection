from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import h5py
import threading
import random_reader
import os
from TRP_Actioness import Model, run_epoch, Input


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "save_path", None, "Model output directory."
)
flags.DEFINE_string(
    "data_path", None, "Model input directory."
)
flags.DEFINE_bool(
    "use_fp16", False, "Train using 16-bits floats instead of 32bit floats"
)

flags.DEFINE_integer(
    "max_epoch", 200, "Max classification epoch."
)
flags.DEFINE_integer(
    "decay_epoch", 5, "the epoch from this learning rate decay."
)
flags.DEFINE_integer(
    "enqueue_size", 80, "the enqueue_size of the data queue"
)
flags.DEFINE_integer(
    "batch_size", 100, "the batch_size of the training data"
)
flags.DEFINE_integer(
    "filter_size", 2, "the filter_size of the convolution layer"
)
flags.DEFINE_float(
    "learning_rate", 0.0001, "init learning rate."
)
flags.DEFINE_float(
    "lr_decay", 0.97, "the learning rate decay."
)
flags.DEFINE_float(
    "keep_prob", 0.6, "the rate of keep prob in dropout"
)
flags.DEFINE_float(
    "momentum", 0.95, 'the strength in momentum'
)
flags.DEFINE_integer(
    "segment_number", 10, 'segment_number'
)
flags.DEFINE_integer(
    "snippet_size", 5, 'snippet_size'
)
FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

class Config(object):
    init_scale = 0.1
    max_grad_norm = 10
    keep_prob = FLAGS.keep_prob
    momentum = FLAGS.momentum

    segment_number = FLAGS.segment_number
    GPU_number = 2
    snippet_size = FLAGS.snippet_size
    filter_size = FLAGS.filter_size
    learning_rate = FLAGS.learning_rate
    lr_decay = FLAGS.lr_decay

    enqueue_size = FLAGS.enqueue_size
    batch_size = FLAGS.batch_size
    hidden_size = 150*3
    input_size = 150
    output_size = 2

    max_epoch = FLAGS.max_epoch
    decay_epoch = FLAGS.decay_epoch

def get_config():
    return Config()

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to save directory")

    if not FLAGS.save_path:
        raise ValueError("Must set --save_path to save directory")


    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1

    act_save_path = FLAGS.save_path + '/b_' + str(config.batch_size) + '_e_' +str(config.max_epoch)+ '_d_' +str(config.keep_prob)
    os.mkdir(act_save_path)
    gpuconfig = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    gpuconfig.gpu_options.allow_growth = True

    g = tf.Graph()
    with g.as_default() ,tf.Session(config=gpuconfig, graph=g) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = Input(config, data=random_reader.raw_data('train', 'all', FLAGS.data_path))
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Model(is_training=True, config=config, inputs=train_input)
            train_cost = tf.summary.scalar("Train classification Cost", m.cost)
            train_lr = tf.summary.scalar("learning rate", m.lr)
            train_merged = tf.summary.merge([train_cost, train_lr])

        with tf.name_scope("Valid"):
            valid_input = Input(eval_config, data=random_reader.raw_data('valid', 'all', FLAGS.data_path))
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Model(is_training=False, config=eval_config, inputs=valid_input)
            valid_cost = tf.summary.scalar("Valid Cost", mvalid.cost)
            valid_merged = tf.summary.merge([valid_cost])

        writer = tf.summary.FileWriter(logdir=act_save_path)
        writer.add_graph(g)
        sv = tf.train.Saver(max_to_keep=0, write_version=2)

        logfile = open(act_save_path+'/best_log.txt', 'w+')

        tf.global_variables_initializer().run()

        train_coord_enqueue = tf.train.Coordinator()
        train_enqueue_threads = threading.Thread(target=train_input.data_q.enqueue, args=[session, train_coord_enqueue])
        train_enqueue_threads.start()

        valid_coord_enqueue = tf.train.Coordinator()
        valid_enqueue_threads = threading.Thread(target=valid_input.data_q.enqueue, args=[session, valid_coord_enqueue])
        valid_enqueue_threads.start()

        coord_dequeue = tf.train.Coordinator()
        dequeue_threads = tf.train.start_queue_runners(coord=coord_dequeue, sess=session)


        best_cost = 99999999
        for i in range(config.max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.decay_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i+1, session.run(m.lr)))
            train_perplexity, train_accuracy = run_epoch(session, writer, m, eval_op=m.train_op, sm_op=train_merged, verbose=True)
            print("Epoch: %d Train Cost: %.3f" % (i + 1, train_perplexity))
            print(train_accuracy)
            valid_perplexity, valid_accuracy = run_epoch(session, writer, mvalid, sm_op=valid_merged)
            print("Epoch: %d Valid Cost: %.3f" % (i + 1, valid_perplexity))
            print(valid_accuracy)

            if valid_perplexity < best_cost:
                best_cost = valid_perplexity
                print("Best Model: %d; Best Cost: %.3f" % (i+1, valid_perplexity), file=logfile)
                print(train_accuracy, file=logfile)
                print(valid_accuracy, file=logfile)
                logfile.flush()
                sv.save(session, act_save_path+'/best_model.ckpt', global_step=i)

            if FLAGS.save_path:
                print("Saving model to %s." % act_save_path)

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

