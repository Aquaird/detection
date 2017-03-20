from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf

import test

def get_data(PATH):
    features = []
    labels = []
    lengths = []
    regressions = []
    frame_indexs = []
    files = os.listdir(PATH)
    files.sort()
    for f in files:
        print(f)
        seq = []
        label = []
        regression = []
        frame_index = []
        with open(os.path.join(PATH,f), 'r') as fh:
            line = fh.readline()
            while line:
                c_label = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                f_line= line.split(',')
                for(i,x) in enumerate(f_line):
                    if x[-1] == '\n':
                        f_line[i] = int(x[:-1])
                    else:
                        f_line[i] = float(x)
                #print(len(f_line))
                seq.append(f_line[:-4])
                regression.append(f_line[-4:-2])
                label_number = int(f_line[-2]) -1
                frame_index.append(f_line[-1])
                c_label[label_number] = 1.0
                label.append(c_label)

                line = fh.readline()


        fh.close()
        features.append(seq)
        labels.append(label)
        lengths.append(len(label))
        regressions.append(regression)
        frame_indexs.append(frame_index)
            #print("file: "+os.path.join(PATH,f)+" completed!")
   # print(features.shape)
    return features, labels, lengths, regressions, frame_indexs

def skeleton_raw_data(data_path=None, types='train'):
    if(types == 'train'):
        path = os.path.join(data_path, "train/all")
    elif(types == "test"):
        path = os.path.join(data_path, "test/all")
    else:
        path = os.path.join(data_path, "valid/all")

    data, label, length, regression, frame_index = get_data(path)
    #4000

    return data, label, length, regression, frame_index

class batch_iterator(object):
    def __init__(self, all_data, mask, batch_size):
        self.i = 0
        self.data = all_data[0]
        self.labels = all_data[1]
        self.lengths = all_data[2]
        self.regressions = all_data[3]
        self.frame_indexs = all_data[4]
        self.mask = np.copy(mask)

        n_example = self.data.shape[0]

        self.n = batch_number = n_example // batch_size
        extra = n_example % batch_size
        if(extra != 0):
            self.batched_mask = np.split(self.mask[:-extra], batch_number)
            self.batched_data = np.split(self.data[:-extra], batch_number)
            self.batched_labels = np.split(self.labels[:-extra], batch_number)
            self.batched_lengths = np.split(self.lengths[:-extra], batch_number)
            self.batched_regressions = np.split(self.regressions[:-extra], batch_number)
            self.batched_frame_indexs = np.split(self.regressions[:-extra], batch_number)
        else:
            self.batched_mask = np.split(self.mask[:], batch_number)
            self.batched_data = np.split(self.data[:], batch_number)
            self.batched_labels = np.split(self.labels[:], batch_number)
            self.batched_lengths = np.split(self.lengths[:], batch_number)
            self.batched_regressions = np.split(self.regressions[:], batch_number)
            self.batched_frame_indexs = np.split(self.frame_indexs[:], batch_number)


    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return self.batched_data[i], self.batched_labels[i], self.batched_mask[i], self.batched_lengths[i], self.batched_regressions[i], self.batched_frame_indexs[i]
        else:
            raise StopIteration()

def padding(all_data, n_steps):
    data = list(all_data)
    n_example = len(data[0])
    mask = []

    #print(n_example)
    for i in range(0, n_example):
        # print(lengths[i])
        mask.append([])
        for j in range(0, data[2][i]):
            mask[i].append(1)
        for j in range(data[2][i], n_steps):
            empty_frame = np.zeros(225,dtype=float)
            empty_label = np.zeros(11, dtype=float)
            empty_regression = np.zeros(2, dtype=float)
            data[0][i].append(empty_frame)
            data[1][i].append(empty_label)
            data[3][i].append(empty_regression)
            data[4][i].append(-1)
            mask[i].append(0)

    data[0] = np.reshape(data[0], (n_example, n_steps, 225))
    data[1] = np.reshape(data[1], (n_example, n_steps, 11))
    data[2] = np.reshape(data[2], (n_example))
    data[3] = np.reshape(data[3], (n_example, n_steps, 2))
    mask = np.reshape(mask, (n_example, n_steps))
    data[4] = np.reshape(data[4], (n_example, n_steps))

    return data, mask

#all_data = skeleton_raw_data("../pre/result/OAD_with_frame_index/", types='valid')
#print(len(all_data))
#a, mask = padding(all_data, 4000)

#for i in range(0,10):
#    i_a = batch_iterator(a, mask, 1)
#    for i,(x,y,m,l,r,f) in enumerate(i_a):
#        #print(r[66][0])
#        #print(r[])
#        for i in range(0,1):
#            f1s = test.f1(y[i], r[i], y[i], f[i], l[i])
#            print(f1s)
