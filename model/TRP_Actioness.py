from __future__ import absolute_import
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.contrib import rnn

import time
import numpy as np
import tensorflow as tf

import trp_reader

def data_type():
    return tf.float32

class Input(object):
    '''the class for input data queue '''
    def __init__(self, config, data):
        self.data_q = trp_reader.data_queue(data, config.batch_size, config.enqueue_size)
        self.epoch_size = data[0].len() // config.batch_size
        self.feature = self.data_q.feature_batch
        self.target = self.data_q.target_batch

def cost_and_accuracy(prediction, target):
    '''
    Calculate cost and accuracy for prediction & target
    Parameter:
        prediction, target: [batch_size, output_size]
    Return:
        cross_entropy loss and accuracy
    '''
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=1)

    correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(target, 1))
    correct_pred = tf.cast(correct_pred, tf.float32)

    return tf.reduce_mean(cross_entropy), tf.reduce_mean(correct_pred)


class FullConnectedLayer(object):
    '''the full connected layer'''
    def __init__(self, output_size, input_data, name):
        '''
        Parameter:
            is_training: whether or not the network is training or not
            batch_size: the batch_size of the full_connected_layer
            input_size: the size of the input data
            output_size: the size of the output value
            input_data: the data to be calculate
        '''
        print(tf.get_variable_scope().name)
        print(input_data.get_shape())
        batch_size = int(input_data.get_shape()[0])
        print(name+": inputs shape:")
        print(input_data.get_shape())
        input_size = int(input_data.get_shape()[-1])
        init_value = np.random.randn(input_size, output_size) / np.sqrt(input_size/2)
        w_fc = tf.get_variable("w", [input_size, output_size], dtype=data_type(), initializer=tf.constant_initializer(init_value))
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
    def __init__(self, input_data, filter_size, name):
        '''
        Parameter:
            input_data: [batch_size, snippet_size, input_size]
            filter_size: the size of the kernal filter
        '''
        with tf.variable_scope(name):
            print(input_data.get_shape())
            print(tf.get_variable_scope().name)
            input_size = int(input_data.get_shape()[2])
            init_value = np.random.randn(input_size, input_size) / np.sqrt(input_size/2)
            conv_w = tf.get_variable("w", [filter_size, input_size, input_size], dtype=data_type(), initializer=tf.constant_initializer(init_value))
            conv_b = tf.get_variable("b", [input_size], dtype=data_type(), initializer=tf.constant_initializer(0.0))
            self._out = tf.nn.relu( tf.nn.conv1d(input_data, conv_w, stride=1, padding='SAME', use_cudnn_on_gpu=True, data_format='NHWC') + conv_b)
            self._w_loss = tf.reduce_sum(conv_w * conv_w)
            tf.get_variable_scope().reuse_variables()

    @property
    def out(self):
        return self._out

    @property
    def w_loss(self):
        return self._w_loss



class BiRNN(object):
    '''bi-direction LSTM'''
    def __init__(self, is_training, keep_prob, input_data, name):
        with tf.variable_scope(name):
            # input_size shape[batch_size, snippet_size, LSTM_size]
            print(tf.get_variable_scope().name)
            print(name+": inputs shape: ")
            print(input_data.get_shape())

            batch_size = int(input_data.get_shape()[0])
            n_steps = int(input_data.get_shape()[1])
            LSTM_size = int(input_data.get_shape()[2])

            lstm_fw_cell = rnn.BasicLSTMCell(LSTM_size, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(LSTM_size, forget_bias=1.0)
            if is_training and keep_prob < 1:
                lstm_fw_cell = rnn.DropoutWrapper(
                    lstm_fw_cell, output_keep_prob = keep_prob)
                lstm_bw_cell = rnn.DropoutWrapper(
                    lstm_bw_cell, output_keep_prob = keep_prob)

            sl = tf.constant(n_steps, dtype=tf.int32, shape=[batch_size])
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell, lstm_bw_cell,
                input_data, sequence_length=sl,
                dtype=tf.float32)
            fw_final_output, bw_final_output = outputs[0][:, -1, :], outputs[1][:, -1, :]
            fw_outputs, bw_outputs = outputs[0], outputs[1]
            self.outputs = tf.add_n([fw_outputs, bw_outputs]) / 2
            add_final_output = (fw_final_output + bw_final_output) / 2

            print(name+": outputs shape: ")
            print(add_final_output.get_shape())
            self._o = add_final_output
            self._s = states
            tf.get_variable_scope().reuse_variables()

    @property
    def out(self):
        return self._o

    @property
    def outs(self):
        return self.outputs
    @property
    def states(self):
        return self._s


class Model(object):
    """The Model"""

    def __init__(self, is_training, config, inputs):

        # inputs is a batch of data
        # input.feature: batch_size * segment_number * snippet_size * input_size
        # input.targets: batch_size  * 2 of snippet-wised label
        self._inputs = inputs.feature
        self._targets = tf.reshape(inputs.target, [-1, 2])


        self._batch_size = config.batch_size
        size = config.hidden_size
        self._input_object = inputs
        self._output_size = output_size = self._targets.get_shape()[-1]
        self.segment_number = config.segment_number
        self.snippet_size = config.snippet_size
        self.input_size = int(self._inputs.get_shape()[-1])


        # input_list: [segment_number, batch_size, snippet_size, input_size]
        input_list = tf.split(self._inputs, self.segment_number, axis=1)
        # out_list: [segment_number, batch_size, 1]
        out_list = []

        with tf.variable_scope("mulit-tower"):
            for i in range(self.segment_number):
                with tf.device('/gpu:%d' % (i%config.GPU_number)):
                    with tf.name_scope('%s_%d' % ("Snippets", i)):
                        input_list[i] = tf.reshape(input_list[i], [self._batch_size, self.snippet_size, -1])
                        conv1_layer = ConvolutionalLayer(input_list[i], config.filter_size, 'conv1')
                        input_1 = conv1_layer.out
                        conv2_layer = ConvolutionalLayer(input_1, config.filter_size, 'conv2')
                        input_2 = conv2_layer.out

                        lstm_input = tf.concat([input_list[i], input_1, input_2], axis=2)
                        bi_lstm_layer_1 = BiRNN(is_training, config.keep_prob, lstm_input, name='biLSTM_1')
                        lstm_1_outs = bi_lstm_layer_1.outs
                        # lstm_output: [batch_size, input_size]
                        bi_lstm_layer_2 = BiRNN(is_training, config.keep_prob, lstm_1_outs, name='biLSTM2')
                        lstm_output = bi_lstm_layer_2.out
                        fc_layer = FullConnectedLayer(1, lstm_output, name='fc')
                        # fc_out: [batch_size, i]
                        fc_out = fc_layer.out
                        out_list.append(fc_out)
                        tf.get_variable_scope().reuse_variables()

        init_value = np.random.randn(self.segment_number, output_size) / np.sqrt(self.segment_number)
        w = tf.get_variable('final_w', [self.segment_number, output_size], dtype=data_type(), initializer=tf.constant_initializer(init_value))
        b = tf.get_variable('final_b', [output_size], dtype=data_type(), initializer=tf.constant_initializer(0.0))
        # final_input: [batch_size, segment_number]
        print(self._targets.get_shape())
        final_input = tf.reshape(tf.concat(out_list, axis=1), [-1, self.segment_number])
        print(final_input.get_shape())
        # self._out: [batch_size, 2]
        self._out = tf.nn.softmax(tf.matmul(final_input, w) + b)
        self._cost, self._accuracy = cost_and_accuracy(self._out, self._targets)

        self._global_step = tf.Variable(0, trainable=False)
        self._new_global_step = tf.placeholder(tf.int32, shape=[], name="new_global_step")
        self._gs_updata = tf.assign(self._global_step, self._new_global_step)

        if not is_training:
            return


        tvars = tf.trainable_variables()

        self._lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.MomentumOptimizer(self._lr, config.momentum)

        grads_and_vars = optimizer.compute_gradients(self._cost, tvars)
        grads = self.add_noise_to_gradients(grads_and_vars, 0.0001)
        grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=self._global_step)

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

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
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def prediction(self):
        return self._out

    @property
    def lr(self):
        return self._lr

    @property
    def global_step(self):
        return self._global_step

    @property
    def train_op(self):
        return self._train_op

    @property
    def output_size(self):
        return self._output_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def accuracy(self):
        return self._accuracy

def run_epoch(session, writer, model, eval_op=None, sm_op=None, verbose=False):
    """run the model on the given data.
    data is formed as all-data form"""

    costs = 0.0
    iters = 0
    accuracy_epoch = []
    fetches = {
        "cost": model.cost,
        "predictions": model.prediction,
        "targets": model.targets,
        "accuracy": model.accuracy
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op
    if sm_op is not None:
        fetches["sm_op"] = sm_op
        fetches["global_step"] = model.global_step

    for step in range(model.inputs.epoch_size):
        accuracy = np.zeros([7, 2])
        feed_dict = {}

        network_time = time.time()
        vals = session.run(fetches, feed_dict)
        network_time = time.time() - network_time

        cost = vals["cost"]
        targets = vals["targets"][:, 1]
        predictions = vals["predictions"][:, 1]
        segment_accuracy = vals["accuracy"]
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i,threshold in enumerate(thresholds):
            tt_count = 0
            tf_count = 0
            ft_count = 0
            for j in zip(predictions, targets):
                if j[0]>=threshold:
                    if j[1]==1:
                        tt_count += 1
                    else:
                        tf_count += 1
                if j[0]<threshold and j[1]==1:
                    ft_count += 1
            if(tt_count+tf_count)!=0:
                precise = tt_count / (tt_count+tf_count)
            else:
                precise = 0
            recall = tt_count / (tt_count + ft_count)
            accuracy[i] = [precise, recall]
            accuracy_epoch.append(accuracy)

        costs += cost
        iters +=  1
        if sm_op is not None:
            writer.add_summary(vals["sm_op"], vals["global_step"])

        if verbose:
            print("%.3f cost: %.3f time_cost: %.3f" %
                  (step*1.0 / model.inputs.epoch_size, costs / iters, network_time))
            print("accuracy: ")
            print(accuracy)
            print("segment accuracy:")
            print(segment_accuracy)

        if eval_op == None:
            model.assign_global_step(session, vals["global_step"]+1)

    return costs / iters, np.mean(accuracy_epoch, axis=0)

def make_actioness_seq(model, session, h5_writer):
    """run the model on the given data.
    data is formed as all-data form"""

    fetches = {
        "predictions": model.prediction
    }


    for step in range(model.inputs.epoch_size):
        feed_dict = {}

        network_time = time.time()
        vals = session.run(fetches, feed_dict)
        network_time = time.time() - network_time

        predictions = vals["predictions"][:, 1]
        for i in predictions:
            h5_writer.resize(h5_writer.shape[0]+1, axis=0)
            h5_writer[-1:] = i




