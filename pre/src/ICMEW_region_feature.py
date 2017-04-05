import os
import h5py
import numpy as np
import random

from ICMEW_Action import Action

NUMBER_STEPS = 8605
REGION_STEPS = 1760

def make_trp_instance(action_object):
    instance = []
    background = []

    all_smooth_data =  action_object.savitzky_seq
    for i in action_object.start_end:
        instance.append(all_smooth_data[i[0]:i[1]])
    for j in action_object.background:
        background.append(all_smooth_data[j[0]:j[1]])

    #print(len(instance))
    #print(len(background))
    instance = np.array(instance)
    background = np.array(background)
    np.random.shuffle(background)
    instance_data = []
    instance_label = []
    instance_length = []
    for i in range(len(instance)):
        instance_data.append(instance[i])
        instance_label.append(1)
        instance_length.append(len(instance[i]))
        instance_data.append(background[i])
        instance_length.append(len(background[i]))
        instance_label.append(0)
        assert len(instance[i]) < REGION_STEPS
        assert len(background[i]) < REGION_STEPS

    data_to_dataset = []
    instance_label = np.reshape(np.array(instance_label), [-1,1])
    instance_length = np.reshape(np.array(instance_length), [-1,1])

    for i in zip(instance_data, instance_label, instance_length):
        data_to_dataset.append(i)

    np.random.shuffle(data_to_dataset)

    #print(data_to_dataset)
    data_to_dataset = np.array(data_to_dataset)
    return data_to_dataset

def insert_data(hdata, append_data):
    hdata.resize(hdata.shape[0]+1, axis=0)
    hdata[-1:] = append_data

def make_feature(action_path, label_path, hgroup, trp_group, n_steps=NUMBER_STEPS):
    '''
    get feature by path and write it into hdf5_data file
    Parameters:
        d_p: data_path
        l_p: label_path
        hdf5_data_writer: h5 files to write
    Return:
        None
    '''
    action_object = Action(data_path, label_path)

    length = len(action_object.savitzky_seq)
    all_smooth_data = padding(action_object.savitzky_seq[:], n_steps, 'float32')
    label_data = action_object.label_seq.astype('float32')
    regression_data = action_object.regression_seq
    all_trp = make_trp_instance(action_object)
    all_trp_data = all_trp[:, 0]
    trp_label = all_trp[:, 1]
    trp_length = all_trp[:, 2]

    print(all_trp_data.shape)
    #print(trp_length)
    #print(trp_label)
    # add to one or two dataset
    if not action_object.TWO:
        smooth_data = all_smooth_data[:, :75]
        hdata = hgroup.get('one_data')
        hlength = hgroup.get('one_length')
        htrp_data = trp_group.get('one_data')
        htrp_length = trp_group.get('one_length')
        htrp_label = trp_group.get('one_label')

        for i,element in enumerate(all_trp_data):
            #print(element.shape)
            if(trp_length[i]<15) or (trp_length[i]>280):
                continue
            else:
                insert_data(htrp_data, padding(element[:, :75], REGION_STEPS, np.float32))
                insert_data(htrp_label, trp_label[i])
                insert_data(htrp_length, trp_length[i])

    else:
        smooth_data = all_smooth_data
        hdata = hgroup.get('two_data')
        hlength = hgroup.get('two_length')
        htrp_data = trp_group.get('two_data')
        htrp_length = trp_group.get('two_length')
        htrp_label = trp_group.get('two_label')

        for i,element in enumerate(all_trp_data):
            if(trp_length[i]<15) or (trp_length[i]>280):
                continue
            else:
                insert_data(htrp_data, padding(element, REGION_STEPS, np.float32))
                insert_data(htrp_label, trp_label[i])
                insert_data(htrp_length, trp_length[i])


    insert_data(hdata, np.concatenate([
        padding(smooth_data, n_steps, 'float32'),
        padding(regression_data, n_steps, 'float32'),
        padding(label_data, n_steps, 'float32')
    ], axis=1))
    insert_data(hlength, length)

    # add to all dataset
    all_hdata = hgroup.get('all_data')
    all_hlength = hgroup.get('all_length')
    insert_data(all_hdata, np.concatenate([
        padding(all_smooth_data, n_steps, 'float32'),
        padding(regression_data, n_steps, 'float32'),
        padding(label_data, n_steps, 'float32')
    ], axis=1))
    insert_data(all_hlength, [length])

    all_htrp_data = trp_group.get('all_data')
    all_htrp_label = trp_group.get('all_label')
    all_htrp_length = trp_group.get('all_length')
    for i,element in enumerate(all_trp_data):
        if(trp_length[i]<15) or (trp_length[i]>250):
            continue
        else:
            insert_data(all_htrp_data, padding(element, REGION_STEPS, np.float32))
            insert_data(all_htrp_label, trp_label[i])
            insert_data(all_htrp_length, trp_length[i])



def padding(sequence, n_steps, data_type):
    '''
    padding a sequence by the self.n_steps with 0s

    Parameters:
        sequences: the np.array needed to be padded [frame_number, input_size]
        data_type: the data type of the sequence data
    '''
    [frame_number, input_size] = sequence.shape
    new = np.concatenate(
        [sequence, np.zeros([n_steps - frame_number, input_size], dtype=data_type)]
    )

    return new


def create_region_h5(feature_root, feature_name):
    h5_writer = h5py.File(os.path.join(feature_root, feature_name), "w")

    train_group = h5_writer.require_group("train")
    valid_group = h5_writer.require_group("valid")

    one_size = 75
    two_size = 150

    train_group.create_dataset('all_label',
                               shape=(0,1),
                               maxshape=(None, 1),
                               dtype='int32')
    train_group.create_dataset('one_label',
                               shape=(0,1),
                               maxshape=(None, 1),
                               dtype='int32')
    train_group.create_dataset('two_label',
                               shape=(0,1),
                               maxshape=(None, 1),
                               dtype='int32')
    valid_group.create_dataset('all_label',
                               shape=(0,1),
                               maxshape=(None, 1),
                               dtype='int32')
    valid_group.create_dataset('one_label',
                               shape=(0,1),
                               maxshape=(None, 1),
                               dtype='int32')
    valid_group.create_dataset('two_label',
                               shape=(0,1),
                               maxshape=(None, 1),
                               dtype='int32')

    train_group.create_dataset('all_data',
                               shape=(0, REGION_STEPS, two_size),
                               maxshape=(None, REGION_STEPS, two_size), dtype='float32')
    train_group.create_dataset('all_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    train_group.create_dataset('one_data',
                               shape=(0, REGION_STEPS, one_size),
                               maxshape=(None, REGION_STEPS, one_size), dtype='float32')
    train_group.create_dataset('one_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    train_group.create_dataset('two_data',
                               shape=(0, REGION_STEPS, two_size),
                               maxshape=(None, REGION_STEPS, two_size), dtype='float32')
    train_group.create_dataset('two_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')

    valid_group.create_dataset('all_data',
                               shape=(0, REGION_STEPS, two_size),
                               maxshape=(None, REGION_STEPS, two_size), dtype='float32')
    valid_group.create_dataset('all_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    valid_group.create_dataset('one_data',
                               shape=(0, REGION_STEPS, one_size),
                               maxshape=(None, REGION_STEPS, one_size), dtype='float32')
    valid_group.create_dataset('one_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    valid_group.create_dataset('two_data',
                               shape=(0, REGION_STEPS, two_size),
                               maxshape=(None, REGION_STEPS, two_size), dtype='float32')
    valid_group.create_dataset('two_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')

    return train_group, valid_group


def create_all_one_two_h5(feature_root, feature_name, is_instance=False):
    h5_writer = h5py.File(os.path.join(feature_root, feature_name), "w")

    train_group = h5_writer.require_group("train")
    valid_group = h5_writer.require_group("valid")


    if is_instance:
        n_steps = REGION_STEPS
        one_size = 75
        two_size = 150
        train_group.create_dataset('all_label',
                                   shape=(0,1),
                                   maxshape=(None, 1),
                                   dtype='int32')
        train_group.create_dataset('one_label',
                                   shape=(0,1),
                                   maxshape=(None, 1),
                                   dtype='int32')
        train_group.create_dataset('two_label',
                                   shape=(0,1),
                                   maxshape=(None, 1),
                                   dtype='int32')
        valid_group.create_dataset('all_label',
                                   shape=(0,1),
                                   maxshape=(None, 1),
                                   dtype='int32')
        valid_group.create_dataset('one_label',
                                   shape=(0,1),
                                   maxshape=(None, 1),
                                   dtype='int32')
        valid_group.create_dataset('two_label',
                                   shape=(0,1),
                                   maxshape=(None, 1),
                                   dtype='int32')
    else:
        n_steps = NUMBER_STEPS
        one_size = 75+2+1
        two_size = 150+2+1

    train_group.create_dataset('all_data',
                               shape=(0, n_steps, two_size),
                               maxshape=(None, n_steps, two_size), dtype='float32')
    train_group.create_dataset('all_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    train_group.create_dataset('one_data',
                               shape=(0, n_steps, one_size),
                               maxshape=(None, n_steps, one_size), dtype='float32')
    train_group.create_dataset('one_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    train_group.create_dataset('two_data',
                               shape=(0, n_steps, two_size),
                               maxshape=(None, n_steps, two_size), dtype='float32')
    train_group.create_dataset('two_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')

    valid_group.create_dataset('all_data',
                               shape=(0, n_steps, two_size),
                               maxshape=(None, n_steps, two_size), dtype='float32')
    valid_group.create_dataset('all_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    valid_group.create_dataset('one_data',
                               shape=(0, n_steps, one_size),
                               maxshape=(None, n_steps, one_size), dtype='float32')
    valid_group.create_dataset('one_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    valid_group.create_dataset('two_data',
                               shape=(0, n_steps, two_size),
                               maxshape=(None, n_steps, two_size), dtype='float32')
    valid_group.create_dataset('two_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')

    return train_group, valid_group


if __name__ == '__main__':
    DATA_ROOT = '../../Data/ICMEW/skeleton/train'
    LABEL_ROOT = '../../Data/ICMEW/label/train'

    DATA_FILE_NAMES = os.listdir(DATA_ROOT)
    np.random.shuffle(DATA_FILE_NAMES)
    CV_NUMBER = 10


    FEATURE_ROOT = '../result/ICMEW'

    TRAIN_GROUP, VALID_GROUP = create_all_one_two_h5(FEATURE_ROOT, "seq_data.hdf5", False)
    TRP_TRAIN_GROUP, TRP_VALID_GROUP = create_region_h5(FEATURE_ROOT, "region_data.hdf5")

    for i, file_name in enumerate(DATA_FILE_NAMES):
        data_path = os.path.join(DATA_ROOT, file_name)
        label_path = os.path.join(LABEL_ROOT, file_name)

        isvalid = ((i % CV_NUMBER) == 0)

        if isvalid:
            hgroup = VALID_GROUP
            trp_h5_group = TRP_VALID_GROUP
        else:
            hgroup = TRAIN_GROUP
            trp_h5_group = TRP_TRAIN_GROUP

        base_name = data_path.split('/')[-1].split('.')[0]
        print("Begin Processing: %s" % base_name)

        make_feature(data_path, label_path, hgroup, trp_h5_group)
