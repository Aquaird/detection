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

import h5_reader
import test

def data_type():
    return tf.float32

def cost_and_accuracy(prediction, target):
    '''
    Calculate cost and accuracy for prediction & target
    Parameter:
        prediction, target: [batch_size, n_steps, output_size]
    Return:
        cross_entropy loss and accuracy frame-wise
    '''
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.reduce_max(tf.abs(target), reduction_indices=2)
    cross_entropy *= mask

    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)

    correct_pred = tf.equal(tf.argmax(prediction,2), tf.argmax(target, 2))
    correct_pred = tf.cast(correct_pred, tf.float32)
    correct_pred *= mask
    accuracy = tf.reduce_sum(correct_pred, 1) 
    accuracy /= tf.reduce_sum(mask, reduction_indices=1)

    return tf.reduce_mean(cross_entropy), tf.reduce_mean(accuracy)

def l2_cost(prediction, target, label_target):
    '''
    Calculate cost and accuracy for prediction & target
    Parameter:
        prediction, target: [batch_size, n_steps, 2]
    Return:
        l2-loss and accuracy frame-wise
    '''

    differ = prediction - target
    l2differ = differ * differ

    l2differ = tf.reduce_sum(l2differ, reduction_indices=2)
    mask = tf.reduce_max(tf.abs(label_target), reduction_indices=2)
    l2differ *= mask
    l2differ = tf.reduce_sum(l2differ, reduction_indices=1)
    l2differ /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(l2differ)


class Input(object):
    '''the class for input data queue '''

    def __init__(self, config, data):
        self.data_q = h5_reader.data_queue(data, config.batch_size, config.enqueue_size)
        self.epoch_size = data[0].len() // config.batch_size
        self.feature = self.data_q.feature_batch
        self.target = self.data_q.target_batch
        self.length = self.data_q.length_batch

class FullConnectedLayer(object):
    '''the full connected layer'''
    def __init__(self, input_size, output_size, input_data, name):
        '''
        Parameter:
            is_training: whether or not the network is training or not
            batch_size: the batch_size of the full_connected_layer
            input_size: the size of the input data
            output_size: the size of the output value
            input_data: the data to be calculate
        '''
        with tf.variable_scope(name):
            batch_size = int(input_data.get_shape()[0])
            print(name+": inputs shape:")
            print(input_data.get_shape())
            init_value = np.random.randn(input_size, output_size) / np.sqrt(input_size/2)
            w_fc = tf.get_variable("w", [input_size, output_size], initializer=tf.constant_initializer(init_value))
            b_fc = tf.get_variable("b", [output_size], dtype=data_type(), initializer = tf.constant_initializer(0.0))

            input_data = tf.reshape(input_data, [-1, input_size])
            self._out = tf.nn.relu(tf.add(tf.matmul(input_data, w_fc), b_fc))
            self._out = tf.reshape(self._out, [batch_size, -1, output_size])
            self._w_loss = tf.reduce_sum(w_fc * w_fc)
            print(name+": outputs shape:")
            print(self._out.get_shape())

    @property
    def out(self):
        return self._out

    @property
    def w_loss(self):
        return self._w_loss


class ConvolutionalLayer(object):
    '''the convolution layer'''
    def __init__(self, input_size, input_data, filter_size, name):
        '''
        Parameter:
            input_size: the size of the input data
            input_data: the data to be calculate
            filter_size: the size of the kernal filter
        '''

        with tf.variable_scope(name):
            init_value = np.random.randn(input_size, input_size) / np.sqrt(input_size/2)
            conv_w = tf.get_variable("w", [filter_size, input_size, input_size], initializer=tf.constant_initializer(init_value))
            conv_b = tf.get_variable("b", [input_size])
            self._out = tf.nn.relu( tf.nn.conv1d(input_data, conv_w, stride=1, padding='SAME', use_cudnn_on_gpu=True, data_format='NHWC') + conv_b)
            self._w_loss = tf.reduce_sum(conv_w * conv_w)

    @property
    def out(self):
        return self._out

    @property
    def w_loss(self):
        return self._w_loss


class RegressionLayer(object):
    def __init__(self, input_size, input_data, prev_size, prev_data, name):
        '''
        Parameter:
            batch_size: the batch_size of the full_connected_layer
            input_size: the size of the input data
            prev_size: the size of the prev-network
            input_data: the data to be calculate
            prev_data: the prev-data
        '''
        with tf.variable_scope(name):

            batch_size = int(input_data.get_shape()[0])
            init_value = np.random.randn(input_size, prev_size) / np.sqrt(input_size/2)
            w_input = tf.get_variable("w_i",
                                      [input_size, prev_size],
                                      initializer=tf.constant_initializer(init_value))

            init_value = np.random.randn(prev_size, prev_size) / np.sqrt(prev_size)
            w_preb = tf.get_variable("w_p", [prev_size, prev_size], initializer=tf.constant_initializer(init_value))
            b_h = tf.get_variable("b_h", [prev_size])

            input_data = tf.reshape(input_data, [-1, input_size])
            prev_data = tf.reshape(prev_data, [-1, prev_size])
            _h = tf.add(tf.matmul(input_data, w_input), tf.matmul(prev_data, w_preb))
            _h = tf.nn.sigmoid(tf.add(_h, b_h))

            init_value = np.random.randn(prev_size, 2) / np.sqrt(prev_size)
            w_out = tf.get_variable("w_o", [prev_size, 2], initializer=tf.constant_initializer(init_value))
            b_out = tf.get_variable("b_o", [2], initializer=tf.constant_initializer(np.zeros(2)))
            self._o = tf.nn.sigmoid(tf.add(tf.matmul(_h, w_out), b_out))
            self._o = tf.reshape(self._o, [batch_size, -1, 2])

            w_cost = tf.reduce_sum(w_input * w_input)
            w_cost = tf.add(w_cost, tf.reduce_sum(w_preb * w_preb))
            self._w_cost = tf.add(w_cost, tf.reduce_sum(w_out * w_out))


    @property
    def out(self):
        return self._o

    @property
    def w_cost(self):
        return self._w_cost


class LSTMLayer(object):
    '''LSTM Layer with dynamic_rnn'''
    def __init__(self, is_training, keep_prob, LSTM_size, input_data, input_lengths, name):
        with tf.variable_scope(name):

            print(name+": inputs shape:")
            print(input_data.get_shape())


            lstm_cell = tf.contrib.rnn.LSTMCell(LSTM_size, forget_bias=1.0)
            self.is_training = is_training
            self.keep_prob = keep_prob

            if is_training and keep_prob < 1:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell, output_keep_prob = keep_prob)

            if self.is_training and keep_prob < 1:
                self._inputs = tf.nn.dropout(input_data, keep_prob)

            self._outputs, self._states = tf.nn.dynamic_rnn(lstm_cell, input_data, sequence_length=input_lengths, dtype=data_type())


    @property
    def outputs(self):
        return self._outputs

    @property
    def states(self):
        return self._states


class MultiLSTMLayer(object):
    def __init__(self, is_training, config, input_data, input_lengths, name):
        with tf.variable_scope(name):
            self.is_training = is_training
            if config.is_stacked :
                self.input_size = config.skeleton_size * 3
                self.size = config.hidden_size * 3
            else:
                self.size = config.hidden_size
                self.input_size = config.skeleton_size
            self.batch_size = config.batch_size
            self.inputs = input_data
            self.input_lengths = input_lengths

            # fc1 layer
            fc1 = FullConnectedLayer(self.input_size, self.size, self.inputs, name="fc1")
            fc1_output = fc1.out
            # LSTM 1 layer
            lstm1 = LSTMLayer(is_training, config.keep_prob, self.size, fc1_output, input_lengths, name="lstm1")
            lstm1_output = lstm1.outputs
            # fc2 layer
            fc2 = FullConnectedLayer(self.size, self.size, lstm1_output, name='fc2')
            fc2_output = fc2.out
            # LSTM 2 layer
            lstm2 = LSTMLayer(is_training, config.keep_prob, self.size, fc2_output, input_lengths, name="lstm2")
            lstm2_output = lstm2.outputs
            # fc3 layer
            fc3 = FullConnectedLayer(self.size, self.size, lstm2_output, name='fc3')
            fc3_output = fc3.out
            # LSTM 3 layer
            lstm3 = LSTMLayer(is_training, config.keep_prob, self.size, fc3_output, input_lengths, name="lstm3")
            lstm3_output = lstm3.outputs
            # fc4 layer
            fc4 = FullConnectedLayer(self.size, self.size, lstm3_output, name='fc4')
            self._outputs = fc4.out
            self._w_loss = fc1.w_loss + fc2.w_loss + fc3.w_loss + fc4.w_loss

    @property
    def outputs(self):
        return self._outputs

    @property
    def w_loss(self):
        return self._w_loss


class Model(object):
    """The Model"""

    def __init__(self, is_training, config, inputs):

        # inputs is a batch of data
        # input.data: batch_size * sequence * 225 of data
        # input.label: batch_size * sequence(used) * 11 of frame-wised label
        # input.length: batch_size * 1 of the length of each example in this batch

        num_steps = config.num_steps

        size = config.hidden_size
        skeleton_size = config.skeleton_size
        self._input_object = inputs
        self._output_size = output_size = config.output_size

        self._inputs, self._regressions = tf.split(inputs.feature, [skeleton_size*3, 2], axis=2)
        self._targets = inputs.target
        self._lengths = tf.reshape(inputs.length, [-1])

        self._batch_size = config.batch_size

        sm,t1,t2 = tf.split(self._inputs,3,2)

        input_0 = sm
        if(config.convolution == True):
            self._conv1_network = ConvolutionalLayer(skeleton_size, sm, 2, 'conv1')
            input_1 = self._conv1_network.out
            self._conv2_network = ConvolutionalLayer(skeleton_size, input_1, 2, 'conv2')
            input_2 = self._conv2_network.out
        else:
            input_1 = t1
            input_2 = t2

        if config.is_stacked:
            lstm_input = tf.concat([input_0, input_1, input_2], axis=2)
            self._lstm_network = MultiLSTMLayer(is_training, config, lstm_input, self._lengths, name='all')
            self.LSTM_outputs = self._lstm_network.outputs
            LSTM_outputs_size = size * 3

        else:
            #smooth sub-network
            self._sm_network = MultiLSTMLayer(is_training, config, input_0, self._lengths,name="smooth")
            sm_outputs = self._sm_network.outputs

            #t1 sub-network
            self._t1_network = MultiLSTMLayer(is_training, config, input_1, self._lengths, name="t1")
            t1_outputs = self._t1_network.outputs

            #t2 sub-network
            self._t2_network = MultiLSTMLayer(is_training, config, input_2, self._lengths, name="t2")
            t2_outputs = self._t2_network.outputs

            self.LSTM_outputs = tf.concat([sm_outputs, t1_outputs, t2_outputs], axis=2)
            LSTM_outputs_size = size * 3

        with tf.variable_scope("softmax"):

            # prediction
            init_value = np.random.randn(LSTM_outputs_size, output_size) / np.sqrt(LSTM_outputs_size)
            softmax_w = tf.get_variable("softmax_w",
                                        [LSTM_outputs_size, output_size],
                                        initializer=tf.constant_initializer(init_value)
                                        )

            softmax_b = tf.get_variable("softmax_b", [output_size], dtype=data_type(), initializer=tf.constant_initializer(0.1, dtype=data_type()))

            softmax_input = tf.reshape(self.LSTM_outputs, [-1, LSTM_outputs_size])

            self._prediction = tf.nn.softmax(tf.matmul(softmax_input, softmax_w) + softmax_b)
            self._prediction = tf.reshape(self._prediction, [-1, num_steps, output_size])

            self._cost, self._accuracy = cost_and_accuracy(self._prediction, self._targets)

        """ regression layer"""
        self._rl = RegressionLayer(LSTM_outputs_size, self.LSTM_outputs, output_size, self._prediction, "regression")
        self._regression_output = self._rl.out
        self._w_cost = self._rl.w_cost

        self._r_cost = r_cost = l2_cost(self._regression_output, self._regressions, self._targets)

        self._balance = tf.Variable(0.0, trainable=False)
        r_cost = self._balance * r_cost
        r_cost_with_w = tf.add(r_cost, config.regular_balance * self._w_cost)

        self._cost_regression = tf.add(r_cost_with_w, self._cost)

        self._new_balance = tf.placeholder(
            tf.float32, shape=[], name="new_balance")
        self._balance_update = tf.assign(self._balance, self._new_balance)

        self._global_step = tf.Variable(0, trainable=False)
        self._new_global_step = tf.placeholder(tf.int32, shape=[], name="new_global_step")
        self._gs_updata = tf.assign(self._global_step, self._new_global_step)

        if not is_training:
            return


        tvars = tf.trainable_variables()

        self._lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.MomentumOptimizer(self._lr, config.momentum)

        grads_and_vars = optimizer.compute_gradients(self._cost, tvars)
        grads = self.add_noise_to_gradients(grads_and_vars, 0.0005)
        grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

        grads_regression_and_vars = optimizer.compute_gradients(self._cost_regression, tvars)
        grads_regression = self.add_noise_to_gradients(grads_regression_and_vars, 0.0005)
        grads_regression, _ = tf.clip_by_global_norm(grads_regression, config.max_grad_norm)

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=self._global_step)

        self._train_with_regression_op = optimizer.apply_gradients(
            zip(grads_regression, tvars),
            global_step=self._global_step)

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_balance(self, session, balance_value):
        session.run(self._balance_update, feed_dict={self._new_balance: balance_value})

    def assign_global_step(self, session, global_step_value):
        session.run(self._gs_updata, feed_dict={self._new_global_step:global_step_value})

    def add_noise_to_gradients(self, grads_and_vars, gradient_noise_scale):
        gradients, variables = zip(*grads_and_vars)
        noisy_gradients = []

        for gradient in gradients:
            if gradient is None:
                noisy_gradients.append(None)
                continue
            if isinstance(gradients, ops.IndexedSlices):
                gradient_shape = gradient.dense_shape
            else:
                gradient_shape = gradient.get_shape()
            noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
            noisy_gradients.append(gradient + noise)

        return noisy_gradients

    @property
    def inputs(self):
        return self._input_object

    @property
    def feature(self):
        return self._inputs

    @property
    def regression(self):
        return self._regressions

    @property
    def targets(self):
        return self._targets

    @property
    def lengths(self):
        return self._lengths

    @property
    def cost(self):
        return self._cost

    @property
    def cost_regression(self):
        return self._cost_regression

    @property
    def r_cost(self):
        return self._r_cost

    @property
    def w_cost(self):
        return self._w_cost


    @property
    def accuracy(self):
        return self._accuracy

    @property
    def classification_prediction(self):
        return self._prediction

    @property
    def regression_prediction(self):
        return self._regression_output

    def input_data(self):
        return self._inputs

    @property
    def lr(self):
        return self._lr

    @property
    def bl(self):
        return self._balance

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_op(self):
        return self._train_op

    @property
    def train_with_regression_op(self):
        return self._train_with_regression_op

    @property
    def output_size(self):
        return self._output_size

    @property
    def batch_size(self):
        return self._batch_size


def run_epoch(session, writer, model, eval_op=None, sm_op=None, verbose=False):
    """run the model on the given data.
    data is formed as all-data form"""

    costs = 0.0
    iters = 0
    accuracy_mean = []

    fetches = {
        "cost": model.cost,
        "accuracy": model.accuracy,
           }

    if eval_op is not None:
        fetches["eval_op"] = eval_op
    if sm_op is not None:
        fetches["sm_op"] = sm_op
        fetches["global_step"] = model.global_step

    for step in range(model.inputs.epoch_size):

        feed_dict = {}

        network_time = time.time()
        vals = session.run(fetches, feed_dict)
        network_time = time.time() - network_time

        cost = vals["cost"]
        accuracy = vals["accuracy"]
        accuracy_mean.append(accuracy)

        costs += cost
        iters +=  1

        if sm_op is not None:
            writer.add_summary(vals["sm_op"], vals["global_step"])

        if verbose:
            print("%.3f perplexity: %.3f time_cost: %.3f" %
                  (step*1.0 / model.inputs.epoch_size, np.exp(costs / iters), network_time))
            print("accuracy: %.5f" % accuracy)

        if eval_op == None:
            model.assign_global_step(session, vals["global_step"]+1)

    return np.exp(costs / iters), np.mean(accuracy_mean)


def run_epoch_regression(session, writer, model, eval_op=None, sm_op=None, verbose=False):
    """run the model on the given data.
    data is formed as all-data form"""

    costs = 0.0
    regression_costs = 0.0
    r_costs= 0.0
    w_costs= 0.0
    iters = 0
    f1_sets_epoch = np.zeros([model.output_size, 3])
    accuracy_mean = []

    fetches = {}
    fetches["accuracy"] = model.accuracy
    fetches["cost"] = model.cost
    fetches["w_cost"] = model.w_cost
    fetches["r_cost"] = model.r_cost
    fetches["cost_regression"] = model.cost_regression

    fetches["c_pred"] = model.classification_prediction
    fetches["r_pred"] = model.regression_prediction
    fetches["targets"] = model.targets
    fetches["lengths"] = model.lengths


    if eval_op is not None:
        fetches["eval_op"] = eval_op
    if sm_op is not None:
        fetches["sm_op"] = sm_op
        fetches["global_step"] = model.global_step

    for step in range(model.inputs.epoch_size):

        feed_dict = {}
        network_time = time.time()
        vals = session.run(fetches, feed_dict)
        network_time = time.time() - network_time

        accuracy = vals["accuracy"]
        cost = vals["cost"]
        w_cost = vals["w_cost"]
        r_cost = vals["r_cost"]
        regression_cost = vals["cost_regression"]
        c_pred = vals["c_pred"]
        r_pred = vals["r_pred"]
        targets = vals["targets"]
        lengths = vals["lengths"]

        accuracy_mean.append(accuracy)
        regression_costs += regression_cost
        w_costs += w_cost
        r_costs += r_cost
        costs += cost
        iters += 1


        f1_sets = np.zeros([model.output_size, 3])
        for i in range(0, model.batch_size):
            label_pred = np.argmax(c_pred[i], axis=1)
            regression_pred = r_pred[i]
            batch_sets = test.f1(label_pred, regression_pred, np.argmax(targets[i],axis=1), lengths[i])
            f1_sets = f1_sets + batch_sets
            f1_sets_epoch = f1_sets_epoch + batch_sets
        f1_scores = test.calculate_f1(f1_sets)

        if sm_op is not None:
            writer.add_summary(vals["sm_op"], vals["global_step"])

        if verbose:
            print("%.3f perplexity: %.3f NET_time_cost: %.3f" %
                  (step*1.0 / model.inputs.epoch_size, np.exp(costs / iters), network_time))
            print("accuracy: ")
            print(accuracy)

            print("regression perplexity: %.3f at r perplexity: %.3f, w perplexity: %.3f" %
                  (np.exp(regression_costs / iters), np.exp(r_costs / iters), np.exp(w_costs / iters)))
            print("f1 scores:")
            print(f1_scores)

        if eval_op is None:
            model.assign_global_step(session, vals["global_step"]+1)

    f1_scores_epoch = test.calculate_f1(f1_sets_epoch)

    return np.exp(regression_costs / iters), np.mean(accuracy_mean), np.mean(f1_scores_epoch[1:])

