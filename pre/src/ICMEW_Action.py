import numpy as np
from general import taylor_one_dim, savitzky_one_dim, gaussion_curve


class Action():
    '''the Action class to get info from ICMEW dataset'''

    TWO_MAN = [12, 14, 16, 18, 21, 24, 26, 27]
    _threshold = 10
    _stdvv = 5
    _point_number = 50
    _label_number = 52

    def __init__(self, data_filename, label_filename):
        '''
        init an Action object.

        Parameters:
            data_filename - the filepath of the data file
            label_filename - the filepath of the label file

        Return:
            an object of Action
        '''

        self.TWO = False
        # action seq of the data_file
        self.action_seq = []
        # savitzky action seq
        self.savitzky_seq = None
        # taylor action seq
        self.taylor_seq = None
        # label seq of the label_file
        self.label_seq = None
        # regression value_seq
        self.regression_seq = None

        self._data_filename = data_filename
        self._label_filename = label_filename

        self.read_action()
        #print("action finished")
        self.read_label()
        #print("label finished")
        self.savitzky()
        #print("savitzky finished")
        #self.taylor()
        #print("taylor finished")

    def read_action(self):
        '''read action from _data_file'''

        with open(self._data_filename, 'r') as hfile:
            reader = hfile.readlines()
            for __row in reader:
                pose = []
                _row = __row.split(" ")
                for j in _row:
                    pose.append(float(j))
                self.action_seq.append(pose)
        hfile.close()
        #print(len(self.action_seq))
        #print(len(self.action_seq[0]))
        self.action_seq = np.array(self.action_seq)

    def read_label(self):
        '''
        read label from _label_file;
        make regression_seq
        '''

        self.start_end = []
        self.background = []
        prev_end = 0
        self.label_seq = np.zeros([len(self.action_seq), 1], dtype='float32')
        with open(self._label_filename, 'r') as hlabelfile:
            reader = hlabelfile.readlines()
            for __row in reader:
                info = __row.split(',')
                start = int(info[1])
                end = int(info[2])
                for i in range(start, end):
                    if i >= len(self.label_seq):
                        print("Error: %s" % self._data_filename)
                        break
                    if int(info[0]) in self.TWO_MAN:
                        self.TWO = True
                    self.label_seq[i] = int(info[0])

                self.start_end.append([start, end])
                self.background.append([prev_end, start])
                prev_end = end

        hlabelfile.close()

        # make regression data from the label seq
        self.regression_seq = np.zeros([len(self.action_seq), 2], dtype='float32')
        for i in self.start_end:
            for j in range(-self._threshold, self._threshold):
                if (i[0]+j < 0) or (i[1]+j >= len(self.action_seq)):
                    continue
                self.regression_seq[i[0] + j][0] = gaussion_curve(i[0] + j, i[0], self._stdvv)
                self.regression_seq[i[1] + j][1] = gaussion_curve(i[1] + j, i[1], self._stdvv)

    def savitzky(self):
        '''
        return the savitzky result of the action seq
        '''
        self.savitzky_seq = np.zeros(self.action_seq.shape, dtype='float32')
        for i in range(self._point_number * 3):
            self.savitzky_seq[:, i] = savitzky_one_dim(self.action_seq[:, i])

        return self.savitzky_seq

    def taylor(self):
        '''
        return the taylor result of the action sequence
        '''

        self.taylor_seq = np.zeros(
            [2, self.action_seq.shape[0], self.action_seq.shape[1]], dtype='float32')
        for i in range(self._point_number * 3):
            self.taylor_seq[0][:, i] = taylor_one_dim(self.action_seq[:, i])
            self.taylor_seq[1][:, i] = taylor_one_dim(self.taylor_seq[0][:, i])

