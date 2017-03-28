import os
import h5py
import numpy as np
import threading
import time
import tensorflow as tf

PATH = {
    "data_path": "../pre/result/ICMEW/renew_10_train_vs_valid_all.hdf5"
}

class data_queue(object):
    def __init__(self, data, batch_size, enqueue_size):
        self.feature = data[0]
        self.length = data[1]
        self.target = []

        self.batch_size = batch_size
        self.enqueue_size = enqueue_size
        n_steps = self.feature.shape[1]
        data_size = self.feature.shape[2] - 1

        self.queue_feature_data = tf.placeholder(tf.float32, shape=[enqueue_size, n_steps, data_size])
        self.queue_length_data = tf.placeholder(tf.int32, shape=[enqueue_size, 1])
        self.queue_target_data = tf.placeholder(tf.float32, shape=[enqueue_size, n_steps, 52])

        self.queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32, tf.float32, tf.int32], shapes=[[n_steps, data_size], [n_steps, 52], [1]])
        self.enqueue_op = self.queue.enqueue_many([self.queue_feature_data, self.queue_target_data, self.queue_length_data])
        self.dequeue_op = self.queue.dequeue()

        self.feature_batch, self.target_batch, self.length_batch = tf.train.batch(self.dequeue_op, batch_size= batch_size, capacity=1000)

    def enqueue(self, sess, coord):
        under = 0
        max_count = self.feature.len()
        while not coord.should_stop():
            #print("starting to write into queue")
            enqueue_start_time = time.time()

            upper = under + self.enqueue_size
            #print("try to enqueue ", under, "to ", upper)
            if upper <= max_count:
                curr_data = self.feature[under:upper].astype('float32')
                curr_feature = curr_data[:, :, :-1]
                curr_label_index = curr_data[:, :, -1]
                curr_target = []
                for i, indexs in enumerate(curr_label_index):
                    curr_target.append(np.identity(52)[indexs.astype("int32")])
                curr_target = np.array(curr_target)
                curr_length = self.length[under:upper].astype('int32')

                under = upper
            else:
                rest = upper - max_count
                curr_data = np.concatenate([self.feature[under:max_count], self.feature[0:rest]]).astype('float32')
                curr_feature = curr_data[:, :, :-1]
                curr_label_index = curr_data[:, :, -1]
                curr_target = []
                for i, indexs in enumerate(curr_label_index):
                    curr_target.append(np.identity(52)[indexs.astype("int32")])
                curr_target = np.array(curr_target)
                curr_length = np.concatenate([self.length[under:max_count], self.length[0:rest]]).astype('int32')

                under = rest

            sess.run(self.enqueue_op, feed_dict={
                self.queue_feature_data : curr_feature,
                self.queue_target_data : curr_target,
                self.queue_length_data: curr_length
            })
            enqueue_time = time.time() - enqueue_start_time
            #print("enqueue time: %s" % enqueue_time)

            #print("added to the queue")
        #print("finished enqueueing")


def raw_data(data_type, person_number):
    f = h5py.File(PATH["data_path"], 'r')
    data_all = f.get(data_type)
    feature_data = data_all.get(person_number+'_data')
    length_data = data_all.get(person_number+'_length')
    return feature_data, length_data

#features, lengths = raw_data('valid', 'one')
#print(features.shape)
#print(lengths.shape)

#dq = data_queue([features, lengths], 5 , 100)
#gpuconfig = tf.ConfigProto()
#gpuconfig.gpu_options.allow_growth = True
#sess = tf.Session(config = gpuconfig)

#coord_enqueue = tf.train.Coordinator()
#enqueue_threads = threading.Thread(target=dq.enqueue, args=[sess, coord_enqueue])
#enqueue_threads.start()

#coord_dequeue = tf.train.Coordinator()
#dequeue_threads = tf.train.start_queue_runners(coord=coord_dequeue, sess=sess)

#for i in range(int(5*features.len() /3)):
#    curr_feature_batch, curr_target_batch, curr_length_batch = sess.run([ dq.feature_batch, d#q.target_batch, dq.length_batch])
#    print("batch: "+str(i))

#coord_enqueue.request_stop()
#coord_enqueue.join([enqueue_threads])
#sess.run(dq.queue.close(cancel_pending_enqueues=True))

#coord_dequeue.request_stop()
#coord_dequeue.join(dequeue_threads)

#sess.close()

#del(dq)

