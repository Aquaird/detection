import os
import h5py
import numpy as np
import threading

import tensorflow as tf

PATH = {
    "data_path": "../pre/result/ICMEW/renew_10_train_vs_valid_all.hdf5"
}

class DataQueue(object):
    '''
    Class to make CV_NUMBER data_queue
    '''
    def __init__(self, data, n_step, enqueue_size=10, batch_size=3):
        '''
        Parameters:
            data_path: the hdf5 file
            validation_index: the list of the index in hdf5 file to be validate
            n_steps: the max frame_number
            batch_size: the batch_size to train the model
        '''

        self.n_steps = n_steps
        self.data = data
        self.batch_size = batch_size
        self.enqueue_size = enqueue_size

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

    def enqueue(self, sess):
        under = 0
        max_now = self.data_list[0].get("")
        while 

