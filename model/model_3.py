from __future__ import print_function

import os
import threading
import time

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import h5py

import reader

flags =tf.flags
logging = tf.logging

flags.DEFINE_string(
    "data_type", None, "which q is going to be trained"
)
flags.DEFINE_string(
    "save_path", None, "path to save model"
)

FLAGS = flags.FLAGS

class Config(object):
    n_layer = 1
    batch_size = 20
    enqueue_size = 50

    learning_rate = 0.01
    lr_decay = 0.97
    keep_prob = 0.6

    init_std = 0.1 # w initializer
    max_grad_norm = 10 # ???
    decay_epoch = 15 # 
    max_epoch = 20 #
    init_scale = 0.01 #

    memory_size = 50 
    lstm_size = 2048
    feature_size = 2048
    question_size = 30
    question_vob_size = 8400 #9372
    answer_size = 1
    answer_vob_size = 1000


    n_steps = 40

def get_config():
    return Config()

class Input(object):
    """the input data"""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.n_steps = n_steps = config.n_steps
        self.enqueue_size = enqueue_size = config.enqueue_size
        self.data_q = reader.data_queue(data, batch_size, enqueue_size)
        self.epoch_size = data[0].len() // batch_size

        self.feature = self.data_q.feature_batch
        self.question = self.data_q.question_batch
        self.answer = self.data_q.answer_batch
        self.candidate = self.data_q.candidate_batch



class BiRNN(object):
    """the birnn network"""

    def __init__(self, is_training, config, input_data, name):
        with tf.variable_scope(name):

            # input shape [batch_size, n_step, n_hidden]
            print(name+": inputs shape: ")
            print(input_data.get_shape())

            keep_prob = config.keep_prob
            n_hidden = config.lstm_size
            batch_size = config.batch_size
            n_steps = config.n_steps

            lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            if is_training and keep_prob < 1:
                lstm_fw_cell = rnn.DropoutWrapper(
                    lstm_fw_cell, output_keep_prob = keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(
                    lstm_bw_cell, output_keep_prob = keep_prob)

            self.sl = tf.constant(n_steps, dtype=tf.int32, shape=[batch_size])
            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input_data, sequence_length=self.sl, dtype=tf.float32)
            fw_outputs, bw_outputs = outputs[0], outputs[1]

            #fw_outputs = tf.reshape(fw_outputs, [-1, n_hidden])
            #bw_outputs = tf.reshape(bw_outputs, [-1, n_hidden])

            add_outputs = tf.add_n([fw_outputs + bw_outputs])



            print(name+": outputs shape: ")
            print(add_outputs.get_shape())
            self._o = add_outputs
            self._s = states

    @property
    def outputs(self):
        return self._o

    @property
    def states(self):
        return self._s

class Model(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        self.hid = []

        # input config
        batch_size = input_.batch_size
        n_steps = input_.n_steps

        # variable size config
        feature_size = config.feature_size
        question_size = config.question_size
        answer_size = config.answer_size
        lstm_size = config.lstm_size
        memory_size = config.memory_size

        n_layer = config.n_layer

        # BiRNN Network
        bi_rnn = BiRNN(is_training, config, input_.feature, "BiRNN")
        bi_rnn_outputs = bi_rnn.outputs
        # bi_rnn_fw_outputs, bi_rnn_bw_outputs: [batch_size, n_steps, lstm_size]

        # Embedding A
        # embedding_inputs: [batch_size * n_steps, lstm_size]
        embedding_inputs = tf.reshape(bi_rnn_outputs, [-1, lstm_size])
        w_initializer = tf.random_normal_initializer(stddev=config.init_std, dtype=tf.float32)
        with tf.variable_scope("fc_A"):
            w_a = tf.get_variable("w", [lstm_size, memory_size], initializer=w_initializer)
            # m_a: [batch_size * n_steps, memory_size]
            m_a = tf.nn.relu(tf.matmul(embedding_inputs, w_a))
            m_a = tf.reshape(m_a, [-1, n_steps, memory_size])

        # Embedding B
        with tf.variable_scope("fc_B"):
            w_b = tf.get_variable("w", [lstm_size, memory_size], initializer=w_initializer)
            # m_b: [batch_size * n_steps, memory_size]
            m_b = tf.nn.relu(tf.matmul(embedding_inputs, w_b))
            m_b = tf.reshape(m_b, [-1, n_steps, memory_size])

        # Embedding Question
        with tf.variable_scope("embedding_q"):
            # input_.question: [batch_size, question_size]
            w_q = tf.get_variable("w", [config.question_vob_size, memory_size], initializer=w_initializer)
            # bin_q: [batch_size, question_size, memory_size]
            bin_q = tf.nn.embedding_lookup(w_q, input_.question)
            # u: [batch_size, memory_size]
            u = tf.reduce_sum(bin_q, axis=1)
            self.hid.append(u)
            # print(u.get_shape())


        # Build Memory
        with tf.name_scope("Memory"):

            for h in range(0, n_layer):
                # h3d: [batch_size, 1, memory_size]
                h3d = tf.reshape(self.hid[-1], [-1,1,memory_size])
                # softmax_input: [batch_size, 1, n_steps]
                softmax_input = tf.matmul(h3d, m_a, adjoint_b=True)
                softmax_input = tf.reshape(softmax_input, [-1, n_steps])
                # p: [batch_size, n_steps]
                p = tf.nn.softmax(softmax_input)
                # p: [batch_size, 1, n_steps]
                p3d = tf.reshape(p, [-1, 1, n_steps])

                # o: [batch_size, 1, memory_size]
                o = tf.matmul(p3d, m_b)
                # o: [batch_size, memory_size]
                o = tf.reshape(o, [-1, memory_size])
                memory_out = tf.add(o, u)

                self.hid.append(memory_out)

        # Build Final Softmax
        with tf.variable_scope("final_softmax"):
            w_s = tf.get_variable("w", [memory_size, config.answer_vob_size], initializer=w_initializer)
            # final_input : [batch_size, answer_vob_size]
            final_input = tf.matmul(self.hid[-1], w_s)

            # [batch_size, answer_vob_size]
            final_softmax_output = tf.nn.softmax(final_input)
            final_softmax_output = tf.reshape(final_softmax_output, [-1, config.answer_vob_size])

        # Obs
        # self._pred: [batch_size, 1]
        self._pred = pred = tf.reshape(tf.argmax(final_softmax_output, axis=1), [-1,1])
        #print("self._pred:")
        #print(self._pred.get_shape())

        pred = tf.cast(pred, tf.int32)
        correct_pred = tf.equal(pred, tf.reshape(input_.answer, [-1,1]))
        #print("correct_pred:")
        #print(correct_pred.get_shape())
        correct_pred = tf.cast(correct_pred, tf.float32)
        #print(correct_pred.get_shape())

        self._answer_accuracy = tf.reduce_mean(correct_pred)

        # input_.candidate: [batch_size, 4]
        candidate_pred = []
        for i in range(0,batch_size):
            b = tf.gather(final_softmax_output[i], input_.candidate[i])
            candidate_pred.append(b)
        candidate_pred = tf.stack(candidate_pred)
        #print(candidate_pred.get_shape())

        # one_of_four: [batch_size, 1]
        one_of_four = tf.argmax(candidate_pred, axis=1)

        # answer_choose: [batch_size, 1]
        answer_choose = []
        for i in range(0, batch_size):
            b = tf.gather(input_.candidate[i], one_of_four[i])
            answer_choose.append(b)
        answer_choose = tf.reshape(tf.stack(answer_choose), [-1,1])
        #print(answer_choose.get_shape())

        choose_correct_pred = tf.equal(answer_choose, input_.answer)
        choose_correct_pred = tf.cast(choose_correct_pred, tf.float32)
        self._choose_accuracy = tf.reduce_mean(choose_correct_pred)

        # self._loss
        # input_.answer: [batch_size, 1]
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_input, labels=tf.reshape(input_.answer, [-1]))
        self._loss = tf.reduce_mean(loss)
        # train_op
        self._globle_step = tf.Variable(0, name="gs")

        if not is_training:
            return

        tvars = tf.trainable_variables()

        self._lr = tf.Variable(0.0, trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        opt = tf.train.GradientDescentOptimizer(self._lr)

        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_grad_norm)

        self._train_op = opt.apply_gradients(
            zip(grads, tvars),
            global_step = tf.contrib.framework.get_or_create_global_step() )

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def cost(self):
        return self._loss

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def answer_accuracy(self):
        return self._answer_accuracy

    @property
    def choose_accuracy(self):
        return self._choose_accuracy

    @property
    def answer_prediction(self):
        return self._pred



def run_epoch(session, model, eval_op=None, verbose=False):
    """run the model on the given data"""
    start_time = time.time()
    costs = 0.0
    iters = 0
    a_a_sum = []
    c_a_sum = []

    fetches = {
        "cost": model.cost,
        "answer_accuracy": model.answer_accuracy,
        "choose_accuracy": model.choose_accuracy,
        "pred": model.answer_prediction,
    }


    if eval_op is not None:
        fetches["eval_op"] = eval_op



    for step in range(model.input.epoch_size):
        feed_dict = {}

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        a_a = vals["answer_accuracy"]
        c_a = vals["choose_accuracy"]
        pred = vals["pred"]
        a_a_sum.append(a_a)
        c_a_sum.append(c_a)

        costs += cost
        iters += 1

        if verbose and iters%10==0:
            print("%.4f perplexity: %.4f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size,
             np.exp( cost),
             iters * model.input.batch_size / (time.time() - start_time)
            ))
            print("A-Accuracy: %.4f" % a_a)
            print("C-Accuracy: %.4f" % c_a)

    # print("OE.accuracy: %.3f MC.accuracy:%.3f\n" % (np.array(a_a_sum).mean(), np.array(c_a_sum).mean()))    
    return np.exp(costs / iters), np.array(a_a_sum).mean(), np.array(c_a_sum).mean()


def main(_):
    if not FLAGS.data_type:
        raise ValueError("Must set --data_type")
    if not FLAGS.save_path:
        raise ValueError("Must set --save_path")

    raw_data = reader.raw_data(FLAGS.data_type)
    train_data, valid_data, test_data= raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.name_scope("Train"):
            train_input = Input(config=config, data=train_data)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = Model(is_training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)
            tf.summary.scalar("Training A-A", m.answer_accuracy)
            tf.summary.scalar("Training C-A", m.choose_accuracy)

        with tf.name_scope("Valid"):
            valid_input = Input(config=config, data=valid_data)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = Model(is_training=False, config=config, input_=valid_input)
            tf.summary.scalar("Valid Loss", mvalid.cost)
            tf.summary.scalar("Valid A-A", mvalid.answer_accuracy)
            tf.summary.scalar("Valid C-A", mvalid.choose_accuracy)

        with tf.name_scope("Test"):
            test_input = Input(config=config, data=test_data)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = Model(is_training=False, config=config, input_=test_input)
            tf.summary.scalar("Test Loss", mtest.cost)
            tf.summary.scalar("Test A-A", mtest.answer_accuracy)
            tf.summary.scalar("Test C-A", mtest.choose_accuracy)


        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        gpuconfig = tf.ConfigProto()
        gpuconfig.gpu_options.allow_growth = True


        min_loss = 0.0
        with sv.managed_session(config=gpuconfig) as session:
            train_enqueue_thread = threading.Thread(target=train_input.data_q.enqueue, args = [session])
            train_enqueue_thread.isDaemon()
            train_enqueue_thread.start()
            valid_enqueue_thread = threading.Thread(target=valid_input.data_q.enqueue, args = [session])
            valid_enqueue_thread.isDaemon()
            valid_enqueue_thread.start()
            test_enqueue_thread = threading.Thread(target=test_input.data_q.enqueue, args = [session])
            test_enqueue_thread.isDaemon()
            test_enqueue_thread.start()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=session)

            for i in range(config.max_epoch):
                lr_decay = config.lr_decay ** max(i+1-config.decay_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d learning rate: %.3f" % (i+1, session.run(m.lr)))

                train_perplexity, train_mean_a, train_mean_c = run_epoch(session, m, eval_op=m.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f. OE accuracy:%.3f. MC accuracy:%.3f." % (i+1,train_perplexity,train_mean_a, train_mean_c))
                open(os.path.join(FLAGS.save_path,'%s_res_out.txt'%FLAGS.data_type),'a').write(\
                    "Epoch: %d Train Perplexity: %.3f. OE accuracy:%.3f. MC accuracy:%.3f.\n" % (i+1,train_perplexity,train_mean_a, train_mean_c))

                valid_perplexity, valid_mean_a, valid_mean_c= run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f. OE accuracy:%.3f. MC accuracy:%.3f." % (i+1, valid_perplexity,valid_mean_a, valid_mean_c))
                open(os.path.join(FLAGS.save_path,'%s_res_out.txt'%FLAGS.data_type),'a').write(\
                    "Epoch: %d Valid Perplexity: %.3f. OE accuracy:%.3f. MC accuracy:%.3f.\n" % (i+1, valid_perplexity,valid_mean_a, valid_mean_c))

                test_perplexity, test_mean_a, test_mean_c = run_epoch(session, mtest)
                print("Epoch: %d Test Perplexity: %.3f. OE accuracy:%.3f. MC accuracy:%.3f." % (i+1, test_perplexity,test_mean_a, test_mean_c))
                open(os.path.join(FLAGS.save_path,'%s_res_out.txt'%FLAGS.data_type),'a').write(\
                    "Epoch: %d Test Perplexity: %.3f. OE accuracy:%.3f. MC accuracy:%.3f.\n" % (i+1, test_perplexity,test_mean_a, test_mean_c))
                # print("Epoch: %d OE.accuracy: %.3f MC.accuracy:%.3f" % (i+1, mean_a_a, mean_c_a))

                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

                if(i == 0):
                    min_loss = valid_perplexity
                else:
                    if min_loss > valid_perplexity:
                        min_loss = valid_perplexity
                        print("saving to %s" % FLAGS.save_path)
                        sv.saver.save(session, FLAGS.save_path+"/model.ckpt", global_step=sv.global_step)


            session.run(train_input.data_q.queue.close(cancel_pending_enqueues=True))
            session.run(valid_input.data_q.queue.close(cancel_pending_enqueues=True))
            session.run(test_input.data_q.queue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)


        del(train_input)
        del(valid_input)
        del(test_input)



if __name__ == "__main__":
    tf.app.run()




