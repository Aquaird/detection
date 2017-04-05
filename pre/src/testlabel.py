import os
import h5py
import numpy as np

from ICMEW_Action import Action

NUM_STEPS = 8605

def read_label(label_filename):
    start_end = []
    with open(label_filename, 'r') as hlabelfile:
        reader = hlabelfile.readlines()
        for __row in reader:
            info = __row.split(',')
            start = int(info[1])
            end = int(info[2])
            start_end.append([start, end])
    hlabelfile.close()
    return start_end

if __name__ == '__main__':
    LABEL_ROOT = '../../Data/ICMEW/label/train'

    LABEL_FILE_NAMES = os.listdir(LABEL_ROOT)
    file_count = len(LABEL_FILE_NAMES)

    max_instance = 0
    min_instance = 800
    count = np.zeros([200], dtype='int32')
    for _, file_name in enumerate(LABEL_FILE_NAMES):

        end = 0
        data_path = os.path.join(LABEL_ROOT, file_name)
        label_path = os.path.join(LABEL_ROOT, file_name)

        instances = read_label(label_path)
        base_name = data_path.split('/')[-1].split('.')[0]
        for i in instances:
            instance = i[1] - i[0]
            #instance = i[0] - end
            #if instance <10:
            #    print(base_name, i[0])
            for j in range(200):
                if (instance < (j+1)*5):
                    count[j] += 1
                    break
            if instance > max_instance:
                max_instance = instance
            if instance < min_instance:
                min_instance = instance

            end = i[1]

    total = np.sum(count)
    start_sample = 0
    end_sample = 200
    alpha = 1.0
    while alpha > 0.97:
        if count[start_sample] > count[end_sample-1]:
            end_sample -= 1
        else:
            start_sample += 1
        alpha = np.sum(count[start_sample:end_sample]) / total

    print(max_instance)
    print(min_instance)
    print(count)
    print(total*2)

    print(start_sample, end_sample+1)
