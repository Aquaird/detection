import os
import h5py
import numpy as np

from ICMEW_Action import Action

NUM_STEPS = 8605

def make_feature(action_object, hdf5_data, hdf5_length):
    '''
    get feature by path and write it into hdf5_data file
    Parameters:
        d_p: data_path
        l_p: label_path
        hdf5_data_writer: h5 files to write
    Return:
        None
    '''

    length = len(action_object.savitzky_seq)
    smooth_data = padding(action_object.savitzky_seq[:], NUM_STEPS, 'float32')
    t1_data = padding(action_object.taylor_seq[0][:], NUM_STEPS, 'float32')
    t2_data = padding(action_object.taylor_seq[1][:], NUM_STEPS, 'float32')

    if not action_object.TWO:
        smooth_data = smooth_data[:, :75]
        t1_data = t1_data[:, :75]
        t2_data = t2_data[:, :75]

    label_data = action_object.label_seq.astype('float32')
    regression_data = action_object.regression_seq

    hdf5_data.resize(hdata.shape[0]+1, axis=0)
    hdf5_data[-1:] = np.concatenate([
        padding(smooth_data, NUM_STEPS, 'float32'),
        padding(t1_data, NUM_STEPS, 'float32'),
        padding(t2_data, NUM_STEPS, 'float32'),
        padding(regression_data, NUM_STEPS, 'float32'),
        padding(label_data, NUM_STEPS, 'float32')
    ], axis=1)

    hdf5_length.resize(hlength.shape[0]+1, axis=0)
    hdf5_length[-1:] = np.array([length])


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


if __name__ == '__main__':
    DATA_ROOT = '../../Data/ICMEW/skeleton/train'
    LABEL_ROOT = '../../Data/ICMEW/label/train'

    FEATURE_ROOT = '../result/ICMEW'
    CV_NUMBER = 10

    H5_WRITER = h5py.File(os.path.join(FEATURE_ROOT, "renew_10_train_vs_valid_all.hdf5"), "w")
    H5_WRITER.attrs.modify('CV_NUMBER', CV_NUMBER)

    DATA_FILE_NAMES = os.listdir(DATA_ROOT)
    np.random.shuffle(DATA_FILE_NAMES)
    file_count = len(DATA_FILE_NAMES)

    #print(DATA_FILE_NAMES)
    CV_GROUP_SETS = []

    TRAIN_GROUP = H5_WRITER.require_group("train")
    VALID_GROUP = H5_WRITER.require_group("valid")
    TRAIN_GROUP.create_dataset('one_data',
                               shape=(0, NUM_STEPS, 225+2+1),
                               maxshape=(None, NUM_STEPS, 225+2+1), dtype='float32')
    TRAIN_GROUP.create_dataset('one_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    TRAIN_GROUP.create_dataset('two_data',
                               shape=(0, NUM_STEPS, 450+2+1),
                               maxshape=(None, NUM_STEPS, 450+2+1), dtype='float32')
    TRAIN_GROUP.create_dataset('two_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')

    VALID_GROUP.create_dataset('one_data',
                               shape=(0, NUM_STEPS, 225+2+1),
                               maxshape=(None, NUM_STEPS, 225+2+1), dtype='float32')
    VALID_GROUP.create_dataset('one_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')
    VALID_GROUP.create_dataset('two_data',
                               shape=(0, NUM_STEPS, 450+2+1),
                               maxshape=(None, NUM_STEPS, 450+2+1), dtype='float32')
    VALID_GROUP.create_dataset('two_length',
                               shape=(0, 1),
                               maxshape=(None, 1),
                               dtype='int32')


    for i, file_name in enumerate(DATA_FILE_NAMES):
        data_path = os.path.join(DATA_ROOT, file_name)
        label_path = os.path.join(LABEL_ROOT, file_name)

        isvalid = ((i % CV_NUMBER) == 0)
        action = Action(data_path, label_path)

        if isvalid:
            hgroup = VALID_GROUP
        else:
            hgroup = TRAIN_GROUP

        if action.TWO:
            hdata = hgroup.get("two_data")
            hlength = hgroup.get("two_length")
        else:
            hdata = hgroup.get("one_data")
            hlength = hgroup.get("one_length")

        base_name = data_path.split('/')[-1].split('.')[0]
        print("Begin Processing: %s" % base_name)


        make_feature(action, hdata, hlength)

