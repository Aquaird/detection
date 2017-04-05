from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import threading
import numpy as np
import tensorflow as tf
import h5py

import trp_reader
import tools
from TRP_Actioness import Model, run_epoch, Input, make_actioness_seq

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "actioness_path", None, "actioness seq store place."
)
flags.DEFINE_string(
    "seq_path", None, "seq path"
)
flags.DEFINE_string(
    "data_path", None, "Model input directory."
)
flags.DEFINE_string(
    "restore_path", None, "Model restore directory."
)
flags.DEFINE_bool(
    "use_fp16", False, "Train using 16-bits floats instead of 32bit floats"
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
    "segment_number", 5, 'segment_number'
)
flags.DEFINE_integer(
    "snippet_size", 10, 'snippet_size'
)
FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32

def insert_data(hdata, append_data):
    hdata.resize(hdata.shape[0]+1, axis=0)
    hdata[-1:] = append_data


def padding(sequence, n_steps, data_type):
    '''
    padding a sequence by the self.n_steps with 0s

    Parameters:
        sequences: the np.array needed to be padded [frame_number, input_size]
        data_type: the data type of the sequence data
    '''
    [frame_number, input_size] = sequence.shape
    new = np.concatenate(
        [sequence, np.zeros([n_steps - frame_number, input_size], dtype=data_type)]
    )

    return new

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
def get_config():
    return Config()

def main(_):
    if not FLAGS.restore_path:
        raise ValueError("Must set --restore_path to save directory")
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to save directory")
    if not FLAGS.actioness_path:
        raise ValueError("Must set --actioness_path to save actioness data")



    config = get_config()

    gpuconfig = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    gpuconfig.gpu_options.allow_growth = True

    g = tf.Graph()
    with g.as_default() ,tf.Session(config=gpuconfig, graph=g) as session:
        initializer = tf.uniform_unit_scaling_initializer(-config.init_scale, config.init_scale)
        with tf.name_scope("Train"):
            train_input = Input(config, data=trp_reader.raw_data('train', 'all', FLAGS.data_path))
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Model(is_training=True, config=config, inputs=train_input)
            train_cost = tf.summary.scalar("Train classification Cost", m.cost)
            train_lr = tf.summary.scalar("learning rate", m.lr)
            train_merged = tf.summary.merge([train_cost, train_lr])

        with tf.name_scope("Valid"):
            valid_input = Input(config, data=trp_reader.raw_data('valid', 'all', FLAGS.data_path))
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Model(is_training=False, config=config, inputs=valid_input)
            valid_cost = tf.summary.scalar("Valid Cost", mvalid.cost)
            valid_merged = tf.summary.merge([valid_cost])

        sv = tf.train.Saver(max_to_keep=0, write_version=2)
        sv.restore(session, FLAGS.restore_path)
        train_coord_enqueue = tf.train.Coordinator()
        train_enqueue_threads = threading.Thread(target=train_input.data_q.enqueue, args=[session, train_coord_enqueue])
        train_enqueue_threads.start()

        valid_coord_enqueue = tf.train.Coordinator()
        valid_enqueue_threads = threading.Thread(target=valid_input.data_q.enqueue, args=[session, valid_coord_enqueue])
        valid_enqueue_threads.start()

        coord_dequeue = tf.train.Coordinator()
        dequeue_threads = tf.train.start_queue_runners(coord=coord_dequeue, sess=session)


        mvalid.run_batch_snippets(config, is_training=False)
        DataFile = h5py.File(FLAGS.seq_path, 'r')

        #actioness_seq_h5 = h5py.File(FLAGS.actioness_path, 'w')
        #actioness_train = actioness_seq_h5.require_group("train")
        #actioness_train_data = actioness_train.create_dataset("all_data",
        #                                                      shape=(0, n_steps, 1),
        #                                                      maxshape=(None, n_steps, 1), dtype='float32')
        #actioness_train_length = actioness_train.create_dataset("all_length",
        #                                                        shape=(0, 1),
        #                                                        maxshape=(None, 1), dtype='int32')
        #actioness_valid = actioness_seq_h5.require_group("valid")
        #actioness_valid_data = actioness_valid.create_dataset("all_data",
        #                                                      shape=(0, n_steps, 1),
        #                                                      maxshape=(None, n_steps, 1), dtype='float32')
        #actioness_valid_length = actioness_valid.create_dataset("all_length",
        #                                                        shape=(0, 1),
        #                                                        maxshape(None, 1), dtype='int32')

        train_seqs = DataFile.get('train').get('all_data')[:, :, :-3]
        train_seqs_length = DataFile.get('train').get('all_length')
        train_seqs_label = DataFile.get('train').get('all_data')[:, :, -1]
        valid_seqs = DataFile.get('valid').get('all_data')[:, :, :-3]
        valid_seqs_length = DataFile.get('valid').get('all_length')

        for i, length in enumerate(train_seqs_length):
            actioness_seq = make_actioness_seq(mvalid, session, train_seqs[i], length[0])
            label_seq = train_seqs_label[i]
            proposal_list = tools.proposal_generator([actioness_seq])
            print("====================================================================")
            print(len(proposal_list[0]))
            tools.self_NMS(proposal_list, [actioness_seq], 0.95)
            print(len(proposal_list[0]))
            gts = tools.ground_truth_locator(label_seq)
            for i, gt in enumerate(gts):
                print(actioness_seq[(gt[0])//FLAGS.snippet_size: (gt[1])//FLAGS.snippet_size])

            print(tools.ground_truth_locator(label_seq))
            print(tools.train_proposals(label_seq, proposal_list[0], FLAGS.snippet_size))
            #insert_data(actioness_train_data, padding(actioness_seq, n_steps, 'float32'))
            #insert_data(actioness_train_length, length)

        for i, length in enumerate(valid_seqs_length):
            actioness_seq = make_actioness_seq(mvalid, session, valid_seqs[i], length[0])
            proposal_list = tools.proposal_generator([actioness_seq])
            print("before: %d" % len(proposal_list[0]))
            tools.self_NMS(proposal_list, [actioness_seq], 0.95)
            print("after: %d" % len(proposal_list[0]))
            #insert_data(actioness_valid_data, padding(actioness_seq, n_steps, 'float32'))
            #insert_data(actioness_valid_length, length)


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

