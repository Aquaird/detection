import os
import h5py
import numpy as np
import threading

import tensorflow as tf

PATH = {
    "data_path": "../pre/result/ICMEW/k_7.hdf5"
}

CV_NUMBER = 7

class DataQueue(object):
    '''
    Class to make CV_NUMBER data_queue
    '''
    def __init__(self, data_path, n_steps, validation_index, enqueue_size=10, batch_size=3):
        '''
        Parameters:
            data_path: the hdf5 file
            validation_index: the list of the index in hdf5 file to be validate
            n_steps: the max frame_number
            batch_size: the batch_size to train the model
        '''

        data_h5 = h5py.File(data_path, 'r')
        self.n_steps = n_steps
        self.train_set = []
        self.valid_set = []
        self.queue_list = [None, None, None, None]
        self.enqueue_list = [None, None, None, None]
        self.batch_data_list = [None, None, None, None]
        self.data = []

        for i in data_h5:
            if int(i) in validation_index:
                self.valid_set.append(data_h5.get(i))
            else:
                self.train_set.append(data_h5.get(i))

        # train_one queue
        self.queue_list[0], self.enqueue_list[0], self.batch_data_list[0] = self.build_queue(228, 10, 2)
        # valid_one queue
        self.queue_list[1], self.enqueue_list[1], self.batch_data_list[1] = self.build_queue(228, 10, 1)
        # train_two queue
        self.queue_list[2], self.enqueue_list[2], self.batch_data_list[2] = self.build_queue(453, 10, 2)
        # valid_two queue
        self.queue_list[3], self.enqueue_list[3], self.batch_data_list[3] = self.build_queue(453, 10, 1)

    def build_queue(self, data_size, enqueue_size, batch_size):
        '''
        function to build both queue for training data and testing data

        Parameters:
            enqueue_size: the number of record to enqueue per-thread operation

        Return:
            queue: the tf.FIFOQueue for this queue
            enqueue_op: the enqueue op for this queue
            batched_data: batched data for the input of model: [feature, label, regression, length]
        '''

        queue_data = tf.placeholder(tf.float32, shape=[enqueue_size, self.n_steps, data_size])
        queue_length = tf.placeholder(tf.int32, shape=[enqueue_size, 1])

        queue = tf.FIFOQueue(
            capacity=32,
            dtypes=[tf.float32, tf.int32],
            shapes=[[self.n_steps, data_size],
                    [1]])
        enqueue_op = queue.enqueue_many([queue_data,
                                         queue_length])
        dequeue_op = queue.dequeue()
        batched_data = tf.train.batch(dequeue_op, batch_size=batch_size, capacity=32)

        return queue, enqueue_op, batched_data

    def enqueue(self, sess, queue, data_list):
        under = 0
        max_now = self.data_list[0].get("")

