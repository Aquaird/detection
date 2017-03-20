from Action import *

import os
from joblib import Parallel, delayed
import tensorflow as tf

# generalize fuzzy feature for msra3d-like data file
# which means given the number of points in one frame, and the data of each points is separated in one line


train = [1,2,3,4,7,8,9,14,15,16,18,19,20,22,23,24,25,32,33,34,35,37,38,39,49,50,51,54,57,58]
test = [0,10,13,17,21,26,27,28,29,36,40,41,42,43,44,45,52,53,55,56]

def makeFeature(PATH, label_root,out_path, point_number):
    action = Action(PATH, point_number)
    baseName = PATH.split('/')[-1].split('.')[0][1:]
    #print(baseName)
    label_path = os.path.join(label_root, 'L'+baseName+'.txt')
    action.read_label(label_path)
    norm_data = np.array(action.norm_point_seq)
    smooth_data = np.array(action.savitzky_point_seq)
    t1_data = np.array(action.taylor_1_point_seq)
    t2_data = np.array(action.taylor_2_point_seq)
    label_data = np.array(action.label_seq)
    regression_data = np.array(action.regression_seq)
    frame_index_data = np.array(action.frame_seq)
    fileBaseName = PATH.split('/')[-1][:-4]
    #print(fileBaseName)
    labelPathName = os.path.join(label_root, 'L'+fileBaseName[1:]+'.txt')
    #print(labelPathName)
    action.read_label(labelPathName)
    label_data = np.array(action.label_seq)
    #print(label_data)

    if(int(fileBaseName[1:]) in train):
        out_path = os.path.join(out_path, 'train')
    elif(int(fileBaseName[1:]) in test):
        out_path = os.path.join(out_path, 'test')
    else:
        out_path = os.path.join(out_path, 'valid')

# all in one:
    all_path = os.path.join(out_path, 'all')
    with open(os.path.join(all_path,fileBaseName+'.al'), 'w+') as wf:
        for i in range(len(t2_data[0])):
            size = t2_data[:,i].flatten().shape[0]
            count = 0
            for j in range(size):
                wf.write(str(smooth_data[:,i].flatten()[j])+',')
            for j in range(size):
                wf.write(str(t1_data[:,i].flatten()[j])+',')
            for j in range(size):
                wf.write(str(t2_data[:,i].flatten()[j])+',')
            for j in range(2):
                wf.write(str(regression_data[i].flatten()[j])+',')
            wf.write(str(label_data[i])+',')
            wf.write(str(frame_index_data[i]))
            wf.write('\n')

    wf.close()

if __name__ == '__main__':
    rawdata_root = '../../Data/OAD/skeleton'
    label_root = '../../Data/OAD/label'
    feature_root = '../../pre/result/OAD/'
    point_number = 25

    for root,dirs,files in os.walk(rawdata_root):
        Parallel(n_jobs=12)(delayed(makeFeature)(os.path.join(rawdata_root, fileName), label_root, feature_root, point_number) for fileName in sorted(files))
