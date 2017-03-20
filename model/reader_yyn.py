import os
import numpy as np
import h5py
import threading

import tensorflow as tf

PATH = {
    "feature_path": "../data/feature",
    "question_path": "../data/question",
    "attribute_path": "../data/attribute",
    "answer_path": "../data/answer",
    "dict_path": "../data/dict",

    "candidate_path": "../data/candidate"
}


def read_attribute(attribute_path, question_class, set_type):
    filePath = os.path.join(attribute_path,question_class+"_"+set_type+"_att_vec.h5")
    f = h5py.File(filePath, 'r')
    data = f.get("data")
    return data

def read_feature(feature_path, question_class, set_type):
    filePath = os.path.join(feature_path, question_class+"_"+set_type+"_feature.h5")
    f = h5py.File(filePath, 'r')
    data = f.get("data")
    return data

def read_question(question_path, question_class, set_type):
    filePath = os.path.join(question_path, "q_"+question_class+"_"+set_type+".h5")
    f = h5py.File(filePath, 'r')
    data = f.get("data")
    return data

def read_answer(answer_path, question_class, set_type):
    filePath = os.path.join(answer_path, "a_"+question_class+"_"+set_type+".h5")
    f = h5py.File(filePath, 'r')
    data = f.get("data")
    return data

def embedding(data):
    #print(data.shape)
    batch_size  = data.shape[0]
    embeded_data = []
    identity = np.identity(10000, dtype=np.float32)
    for i in range(0, batch_size):
        #print(data[i].shape)
        large_matrix = identity[data[i]]
        compressed_matrix = np.sum(large_matrix, axis=0)
        embeded_data.append(compressed_matrix)
    embeded_data = np.array(embeded_data)
    return embeded_data


def read_candidate(candidate_path, question_class, set_type):
    filePath = os.path.join(candidate_path, question_class+"_"+set_type+"_candidate.h5")
    f = h5py.File(filePath, 'r')
    data = f.get("data")
    return data

class data_iterator:
    def __init__(self, data, batch_size):
        self.i = 0
        self.batch_size = batch_size
        self.n = n_example = data.len()
        self.batch_number = ( n_example // batch_size ) + 1
        self.data = data

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n-1:
            i = self.i
            self.i += 1
            return self.data[i*self.batch_size : (i+1)*self.batch_size]
        elif self.i == self.n-1:
            i = self.i
            self.i += 1
            return self.data[i*self.batch_size : self.n]
        else:
            raise StopIteration()

class data_queue(object):
    def __init__(self, data, batch_size, enqueue_size):
        self.feature = data[0]
        self.question = data[1]
        self.answer = data[2]
        self.candidate = data[3]
        self.attribute = data[4]
        self.batch_size = batch_size
        self.enqueue_size = enqueue_size

        self.queue_feature_data = tf.placeholder(tf.float32, shape = [enqueue_size, 40, 2048])
        self.queue_question_data = tf.placeholder(tf.int32, shape = [enqueue_size, 30])
        self.queue_answer_data = tf.placeholder(tf.int32, shape = [enqueue_size, 1])
        self.queue_candidate_data = tf.placeholder(tf.int32, shape = [enqueue_size, 4])
        self.queue_attribute_data = tf.placeholder(tf.int32, shape = [enqueue_size, 40, 5])

        self.queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.float32, tf.int32, tf.int32, tf.int32, tf.int32], \
            shapes=[[40,2048], [30], [1], [4],[40,5]])
        self.enqueue_op = self.queue.enqueue_many([self.queue_feature_data, self.queue_question_data,\
         self.queue_answer_data, self.queue_candidate_data, self.queue_attribute_data])
        self.dequeue_op = self.queue.dequeue()

        self.feature_batch, self.question_batch, self.answer_batch, self.candidate_batch, self.attribute_batch\
         = tf.train.batch(self.dequeue_op, batch_size= batch_size, capacity=1000)

    def enqueue(self, sess):
        under = 0
        max = self.feature.len()
        while True:
            #print("starting to write into queue")
            upper = under + self.enqueue_size
            #print("try to enqueue ", under, "to ", upper)
            if upper <= max:
                curr_feature = self.feature[under:upper].astype('float32')
                curr_question = self.question[under:upper].astype('int32')
                curr_answer = self.answer[under:upper].astype('int32')  -1
                curr_candidate = self.candidate[under:upper].astype('int32')  - 1
                curr_attribute = self.attribute[under:upper].astype('int32')
                under = upper
            else:
                rest = upper - max
                curr_feature = np.concatenate((self.feature[under:max], self.feature[0:rest])).astype('float32')
                #print(curr_feature.shape)
                curr_question = np.concatenate((self.question[under:max], self.question[0:rest])).astype('int32')
                curr_answer = np.concatenate((self.answer[under:max], self.answer[0:rest])).astype('int32')  - 1
                curr_candidate = np.concatenate((self.candidate[under:max], self.candidate[0:rest])).astype('int32') - 1
                curr_attribute = np.concatenate((self.attribute[under:max], self.attribute[0:rest])).astype('int32')
                under = rest

            sess.run(self.enqueue_op, feed_dict={
                self.queue_feature_data : curr_feature,
                self.queue_question_data : curr_question,
                self.queue_answer_data : curr_answer,
                self.queue_candidate_data : curr_candidate,
                self.queue_attribute_data : curr_attribute
            })

            #print("added to the queue")
        #print("finished enqueueing")


def raw_data(data_type):
    f = read_feature(PATH["feature_path"], data_type, "train")
    a = read_answer(PATH["answer_path"], data_type, "train")
    q = read_question(PATH["question_path"], data_type, "train")
    c = read_candidate(PATH["candidate_path"], data_type, "train")
    att = read_attribute(PATH["attribute_path"], data_type, "train")
    train_data = [f,q,a,c,att]

    f = read_feature(PATH["feature_path"], data_type, "valid")
    a = read_answer(PATH["answer_path"], data_type, "valid")
    q = read_question(PATH["question_path"], data_type, "valid")
    c = read_candidate(PATH["candidate_path"], data_type, "valid")
    att = read_attribute(PATH["attribute_path"], data_type, "valid")
    valid_data = [f,q,a,c,att]

    f = read_feature(PATH["feature_path"], data_type, "test")
    a = read_answer(PATH["answer_path"], data_type, "test")
    q = read_question(PATH["question_path"], data_type, "test")
    c = read_candidate(PATH["candidate_path"], data_type, "test")
    att = read_attribute(PATH["attribute_path"], data_type, "test")
    test_data = [f,q,a,c,att]

    return train_data, valid_data, test_data

#f = read_feature(PATH["feature_path"], "what", "train")
#a = read_answer(PATH["answer_path"], "what", "train")
#q = read_question(PATH["question_path"], "what", "train")
#c = read_candidate(PATH["candidate_path"], "what", "train")



#print(f.len())
#print(a.len())
#print(q.len())
#print(c.len())

#j = 0
#for i in data_iterator(f, 1000):
#    b = i.shape
#    print(b)
#    j += 1
#print(j)

#dq = data_queue([f,q,a,c], 20, 100)
#sess = tf.Session()
#enqueue_thread = threading.Thread(target=dq.enqueue, args=[sess])
#enqueue_thread.isDaemon()
#enqueue_thread.start()

#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(coord=coord, sess=sess)

#for i in range(5):
    # no feed
#    curr_feature_batch, curr_question_batch, curr_answer_batch, curr_candidate_batch = sess.run([ dq.feature_batch, dq.question_batch, dq.answer_batch, dq.candidate_batch ])
#    print(curr_question_batch.shape)


#enqueue_thread.join()

#sess.run(dq.queue.close(cancel_pending_enqueues=True))
#coord.request_stop()
#coord.join(threads)
#sess.close()

#del(dq)
