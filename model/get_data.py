from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf

def get_raw_data(fileNameQueues):
    """Load raw data from data directory "data_path".
    Args:
    data_path: string path to the directory where lies all the input data

    Returns:
    tuple: (train_data, valid_data, test_data, data_size)
    where each of the data objects can be passed to DataInerator
    """
    record_defaults = []
    results = []
    for i in range(75):
        record_defaults.append([0.0])

    for i in range(len(fileNameQueues)):
        results.append([])

    for i in range(len(fileNameQueues)):
        reader = tf.TextLineReader()
        key, value = reader.read(fileNameQueues[i])
        # default
        result = tf.decode_csv(value, record_defaults=record_defaults)
        results[i] = tf.pack(result)

    return results

def inputPipeLine(fileNames, batchSize=20, numEpochs=None):
    fileNameQueues = []
    for i in range(len(fileNames)):
        fileNameQueues.append(tf.train.string_input_producer(fileNames[i], num_epochs=numEpochs))
    input_data = get_raw_data(fileNameQueues)
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 12*batchSize
    input_data_Batch = tf.train.shuffle_batch(input_data, batch_size=batchSize, num_threads=12, capacity=capacity, min_after_dequeue=min_after_dequeue)

    return input_data_Batch





data_path = "../pre/result/OAD"
if "train" == type:
    data_path = os.path.join(data_path, "train")
elif "valid" == type:
    data_path = os.path.join(data_path, "valid")
else:
    data_path = os.path.join(data_path, "test")

## smooth data:
feature_types = ["smooth", "t1", "t2"]
fileNames = []
for i in range(len(feature_types)):
    fileNames.append([])

for i in range(len(feature_types)):
    exact_path = os.path.join(data_path, feature_types[i])
    list_dirs = os.walk(exact_path)
    for root,dirs,files in list_dirs:
        ##print(files)
        files.sort(key=lambda x:int(x[1:-3]))
        for f in files:
            fileNames[i].append(os.path.join(exact_path, f))
print(fileNames)

featureBatch = inputPipeLine(fileNames, batchSize=20)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while True:
            temp = sess.run(featureBatch)
            print(np.array(temp).shape)
    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()
        coord.join(threads)
        sess.close()
