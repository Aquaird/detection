import h5py
import numpy as np
import tensorflow as tf
import random
import threading

PATH = {
    "data_path": "../pre/result/ICMEW/region_data.hdf5"
}

TARGET_SIZE = 2

def make_snippets(seq_list, length_list, segment_number, snippet_size):
    snippets_list = []
    length_list = np.reshape(length_list, [-1])
    for seq, length in zip(seq_list, length_list):
        if length >= segment_number*snippet_size:
            snippets = sample(seq[:length], segment_number, snippet_size)
        else:
            snippets = convolution(seq[:length], segment_number, snippet_size)
        snippets_list.append(snippets)
    return snippets_list

def sample(seq, segment_number, snippet_size):
    segment_length = len(seq) // (segment_number)
    remains = len(seq) % segment_number

    divide = []
    for i in range(0, segment_number):
        divide.append(segment_length*(i+1))
    divide = np.array(divide)
    for i in range(0, remains):
        append_segment = random.randint(0, segment_number-1)
        divide[append_segment:] += 1

    snippets = []
    for i in range(segment_number):
        if i == 0:
            start = 0
        else:
            start = divide[i-1]
        end = divide[i]

        snippets_start = random.randint(start, end-snippet_size)
        snippets.append(seq[snippets_start:snippets_start+snippet_size])

    return np.array(snippets)

def convolution(seq, segment_number, snippet_size):
    move = len(seq) - snippet_size
    step = move // (segment_number-1)
    remains = move % (segment_number-1)

    divide = []
    for i in range(0, segment_number):
        divide.append(step*i)
    divide = np.array(divide)
    for i in range(0, remains):
        append_step = random.randint(1, segment_number)
        divide[append_step:] += 1

    snippets = []
    for i in range(segment_number):
        snippets.append(seq[divide[i]:divide[i]+snippet_size])

    return np.array(snippets)


class data_queue(object):
    def __init__(self, data, batch_size, enqueue_size, segment_number, snippet_size):
        self.feature = data[0]
        self.length = data[2]
        self.label = data[1]
        self.target = []
        self.segment_number = segment_number
        self.snippet_size = snippet_size
        self.batch_size = batch_size
        self.enqueue_size = enqueue_size
        self.feature_shape = [enqueue_size, segment_number, snippet_size, self.feature.shape[-1]]

        self.queue_feature_data = tf.placeholder(tf.float32, shape=self.feature_shape)
        self.queue_length_data = tf.placeholder(tf.int32, shape=[enqueue_size, 1])
        self.queue_target_data = tf.placeholder(tf.float32, shape=[enqueue_size, TARGET_SIZE])

        self.queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32, tf.float32, tf.int32], shapes=[self.feature_shape[1:], [TARGET_SIZE], [1]])
        self.enqueue_op = self.queue.enqueue_many([self.queue_feature_data, self.queue_target_data, self.queue_length_data])
        self.dequeue_op = self.queue.dequeue()
        self.feature_batch, self.target_batch, self.length_batch = tf.train.batch(self.dequeue_op, batch_size= batch_size, capacity=1000)

    def enqueue(self, sess, coord):
        under = 0
        max_count = self.feature.len()
        while not coord.should_stop():
            #print("starting to write into queue")
            #enqueue_start_time = time.time()

            upper = under + self.enqueue_size
            #print("try to enqueue ", under, "to ", upper)
            if upper <= max_count:
                curr_length = np.reshape(self.length[under:upper].astype('int32'), [-1, 1])
                curr_feature = make_snippets(self.feature[under:upper].astype('float32'), curr_length, self.segment_number, self.snippet_size)
                curr_label_index = self.label[under:upper].astype('int32')
                curr_target = []
                for i, indexs in enumerate(curr_label_index):
                    curr_target.append(np.identity(TARGET_SIZE)[indexs.astype("int32")])
                curr_target = np.reshape(np.array(curr_target), [-1, TARGET_SIZE])

                under = upper
            else:
                rest = upper - max_count
                curr_length = np.reshape(np.concatenate([self.length[under:max_count], self.length[0:rest]]).astype('int32'), [-1, 1])
                curr_feature = make_snippets(np.concatenate([self.feature[under:max_count], self.feature[0:rest]]).astype('float32'), curr_length, self.segment_number, self.snippet_size)
                curr_label_index = np.concatenate([self.label[under:max_count], self.label[0:rest]]).astype('float32')
                curr_target = []
                for i, indexs in enumerate(curr_label_index):
                    curr_target.append(np.identity(TARGET_SIZE)[indexs.astype("int32")])
                curr_target = np.reshape(np.array(curr_target), [-1, TARGET_SIZE])

                under = rest

            sess.run(self.enqueue_op, feed_dict={
                self.queue_feature_data : curr_feature,
                self.queue_target_data : curr_target,
                self.queue_length_data: curr_length
            })
            #enqueue_time = time.time() - enqueue_start_time
            #print(enqueue_time)

            #print("added to the queue")
        print("finished enqueueing")


def raw_data(data_type, person_number, data_path):
    f = h5py.File(data_path, 'r')
    data_all = f.get(data_type)
    feature_data = data_all.get(person_number+'_data')
    length_data = data_all.get(person_number+'_length')
    label_data = data_all.get(person_number+'_label')
    return feature_data, label_data, length_data
'''
features, labels, lengths = raw_data('valid', 'all', PATH["data_path"])
print(features.shape)
print(labels.shape)
print(lengths.shape)

dq = data_queue([features, labels, lengths], 100 , 300, 5, 5)
gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth = True
sess = tf.Session(config = gpuconfig)
sess = tf.Session()
coord_enqueue = tf.train.Coordinator()
enqueue_threads = threading.Thread(target=dq.enqueue, args=[sess, coord_enqueue])
enqueue_threads.start()

coord_dequeue = tf.train.Coordinator()
dequeue_threads = tf.train.start_queue_runners(coord=coord_dequeue, sess=sess)

print(features.len()/100)
for i in range(int(features.len() /100)):
    curr_feature_batch, curr_target_batch, curr_length_batch = sess.run([ dq.feature_batch, dq.target_batch, dq.length_batch])
    print(curr_feature_batch.shape)
    #print(curr_target_batch)
    print("batch: "+str(i))

coord_enqueue.request_stop()
sess.run(dq.queue.close(cancel_pending_enqueues=True))
coord_enqueue.join([enqueue_threads], stop_grace_period_secs=5)

coord_dequeue.request_stop()
coord_dequeue.join(dequeue_threads)

sess.close()

del(dq)
'''
