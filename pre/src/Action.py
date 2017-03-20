import numpy as np
from general import *

ROOT = 6


# read data to form a action seq
# return pose seq(20*frame) and point seq(frame*20)
# action_seq is consisted of pose in each frame
# point_seq is consisted of each points' traj
# each point_value is norm by its root
class Action():
    def __init__(self, filename, POINT_NUMBER):
        self.action_seq = []
        self.point_seq = []
        # action label of each frame
        self.label_seq = []
        # frame no of each frame
        self.frame_seq = []
        # the start & end frame no of actions
        self.start_end = []
        # list of start points
        self.start_list = []
        # normalized action seq
        self.norm_action_seq = []
        # normalized point seq
        self.norm_point_seq = []
        # regression point seq
        self.regression_seq = []

        self.savitzky_point_seq = []
        self.fourier_point_seq = []
        self.taylor_1_point_seq = []
        self.taylor_2_point_seq = []
        self.frame_number = 0
        self.point_number = 0
        self.MEAN = []

        self.threshold = 10
        self.stdvv = 5

        '''
        self.ROOT_INDEX = 6
        self.maxmin_MEAN = []
        self.pca_vector = []
        self.maxmin_action_seq = []
        self.maxmin_point_seq = []
        self.bounder = [[float("inf"),float("-inf")],[float("inf"),float("-inf")],[float("inf"),float("-inf")]]#x,y,z
        self.norm_bounder = [[float("inf"),float("-inf")],[float("inf"),float("-inf")],[float("inf"),float("-inf")]] #x,y,z
        self.bounders = []
        self.norm_bounders = []
        self.norm_fuzzy_feature = []
        '''

        self.filename = filename
        self.point_number = POINT_NUMBER
        self.read_action()
        self.normalization()
        self.savitzky()
        self.calculate_taylor()

        '''
        self.calculate_bounders()
        self.maxminnorm()
        self.calculate_pca()
        self.calculate_fuzzy_feature()
        '''


    def read_action(self):
        for i in range(0, self.point_number):
            self.point_seq.append([])
        # print(point_seq)
        with open(self.filename, 'r') as f:
            reader = f.readlines()
            i = 0
            pose = []
            for __row in reader:
                if (self.point_number) == i:
                    #print(__row)
                    #calculate mean
                    a_pose = np.array(pose)
                    mean_x = (a_pose[0][0] + a_pose[12][0] + a_pose[16][0]) / 3.0
                    mean_y = (a_pose[0][1] + a_pose[12][1] + a_pose[16][1]) / 3.0
                    mean_z = (a_pose[0][2] + a_pose[12][2] + a_pose[16][2]) / 3.0
                    mean_point=[mean_x, mean_y, mean_z]
                    self.MEAN.append(mean_point)

                    self.action_seq.append(pose)
                    pose = []
                    i = 0
                    self.frame_seq.append(int(__row[:-1]))
                else:
                    # print(__row)
                    _row = __row.split(" ")
                    #print(_row)
                    point_data = [float(_row[0]), float(_row[1]), float(_row[2])]
                    # print(point_data)
                    #print(__row)
                    self.point_seq[i].append(point_data)
                    pose.append(point_data)
                    i += 1

        self.frame_number = len(self.action_seq)
        return [self.action_seq, self.point_seq]

    def read_label(self, label_path):
        label_dic = {
            "drinking":1,
            "eating":2,
            "writing":3,
            "opening cupboard":4,
            "washing hands":5,
            "opening microwave oven":6,
            "sweeping":7,
            "gargling":8,
            "Throwing trash":9,
            "wiping":10,
            "nothing":0
        }
        self.start_end = []
        for i in range(11):
            self.start_end.append([])

        self.start_list = []
        # get label information
        with open(label_path, 'r') as f:
            label = 0
            reader = f.readlines()
            for __row in reader:
                if(__row[:-1]) in label_dic.keys():
                    label = label_dic[__row[:-1]]
                else:
                    point = __row.split(' ')
                    start = int(point[0])
                    end = int(point[1])
                    self.start_end[label].append([start, end])
                    self.start_list.append([start,end,label])
        f.close()

        self.start_list.sort(key=lambda x:int(x[0]))
        #print(self.start_list)

        # print(self.start_end)
        # get label_classification result
        j = 0
        for i in range(len(self.action_seq)):
            self.label_seq.append(1)
            self.regression_seq.append([0.0,0.0])
            if(j<len(self.start_list)):
                #print(self.frame_seq[i],j)
                if(self.frame_seq[i] >= self.start_list[j][0] - self.threshold):
                    if(self.frame_seq[i] <= self.start_list[j][1]):
                        self.label_seq[i] = (self.start_list[j][2])+1
                    if(self.frame_seq[i] <= self.start_list[j][1] + self.threshold):
                        self.regression_seq[i] = [gaussion_curve(self.frame_seq[i], self.start_list[j][0], self.stdvv),
                                                  gaussion_curve(self.frame_seq[i], self.start_list[j][1], self.stdvv)]
                    else:
                        self.label_seq[i] = 1
                        j += 1


        '''
        for i in range(len(self.start_end)):
            if(0 < len(self.start_end[i])):
                for j in self.start_end[i]:
                    if(j[0] not in self.frame_seq):
                        while 
                    for n in range(self.frame_seq.index(j[0]), self.frame_seq.index(j[1]+1)):
                        self.label_seq[n] = i
        '''
    def normalization(self):
        for i,pose in enumerate(self.action_seq):
            norm_pose = []
            for joint in pose:
                norm_joint = [0.0,0.0,0.0]
                for axi in range(0,3):
                    norm_joint[axi] = joint[axi] - self.MEAN[i][axi]
                norm_pose.append(norm_joint)
            self.norm_action_seq.append(norm_pose)

        for i in range(0, len(self.point_seq)):
            self.norm_point_seq.append([])
        for i in range(0, len(self.point_seq)):
            for j in range(0, self.frame_number):
                norm_joint = [0.0,0.0,0.0]
                for axi in range(0,3):
                    norm_joint[axi] = self.point_seq[i][j][axi] - self.MEAN[j][axi]
                self.norm_point_seq[i].append(norm_joint)
        return [self.norm_action_seq, self.norm_point_seq]

    # calculate Fourier features for point_seq
    # output Fourier_point_seq
    def calculate_fourier(self):

        return self.fourier_point_seq

    def savitzky(self):
        for i,j in enumerate(self.norm_point_seq):
            self.savitzky_point_seq.append(savitzky_filter(j))

    # calculate Taylor features for point_seq
    # output Taylor_point_seq
    def calculate_taylor(self):
        for i,j in enumerate(self.savitzky_point_seq):
            self.taylor_1_point_seq.append(taylor_transfer(j))
        for i,j in enumerate(self.taylor_1_point_seq):
            self.taylor_2_point_seq.append(taylor_transfer(j))
       # print(len(self.taylor_1_point_seq), len(self.taylor_2_point_seq))

    '''
    ######################################## useless features ###################################
    def maxminnorm(self):
        maxmin = [0.0,0.0,0.0]
        maxmin[0] = self.norm_bounder[0][1] - self.norm_bounder[0][0]
        maxmin[1] = self.norm_bounder[1][1] - self.norm_bounder[1][0]
        maxmin[2] = self.norm_bounder[2][1] - self.norm_bounder[2][0]

        for pose in self.norm_action_seq:
            maxmin_pose = []
            for joint in pose:
                maxmin_joint = [0.0,0.0,0.0]
                for axi in range(0,3):
                    maxmin_joint[axi] = (joint[axi] - self.norm_bounder[axi][0]) / maxmin[axi]

                maxmin_pose.append(maxmin_joint)
            self.maxmin_action_seq.append(maxmin_pose)

        for i in range(0, len(self.norm_point_seq)):
            self.maxmin_point_seq.append([])
        for i in range(0, len(self.norm_point_seq)):
            for j in range(0, self.frame_number):
                maxmin_joint = [0.0,0.0,0.0]
                for axi in range(0,3):
                    maxmin_joint[axi] = (self.norm_point_seq[i][j][axi] - self.norm_bounder[axi][0]) / maxmin[axi]
                self.maxmin_point_seq[i].append(maxmin_joint)

        for i in self.MEAN:
            point = [0.0,0.0,0.0]
            for axi in range(0,3):
                point[axi] = (0 - self.norm_bounder[axi][0]) / maxmin[axi]
            self.maxmin_MEAN.append(point)

        return [self.maxmin_action_seq, self.maxmin_point_seq]

    def calculate_pca(self):
        #calculate PCA
        for pose in self.norm_action_seq:
            a_pose = np.array(pose)
            self.pca_vector.append(PCA_vector(a_pose))

        return self.pca_vector



    # calculate the x,y,z bounder of data in one action_seq
    # todo calculate the bounder of each point
    def calculate_bounders(self):

        for points in self.point_seq:
            self.bounders.append(calculate_bounder(points))
        for points in self.norm_point_seq:
            self.norm_bounders.append(calculate_bounder(points))

        for i in self.bounders:
            if i[0][0] < self.bounder[0][0]:
                self.bounder[0][0] = i[0][0]
            if i[0][1] > self.bounder[0][1]:
                self.bounder[0][1] = i[0][1]
            if i[1][0] < self.bounder[1][0]:
                self.bounder[1][0] = i[1][0]
            if i[1][1] > self.bounder[1][1]:
                self.bounder[1][1] = i[1][1]
            if i[2][0] < self.bounder[2][0]:
                self.bounder[2][0] = i[2][0]
            if i[2][1] > self.bounder[2][1]:
                self.bounder[2][1] = i[2][1]

        for i in self.norm_bounders:
            if i[0][0] < self.norm_bounder[0][0]:
                self.norm_bounder[0][0] = i[0][0]
            if i[0][1] > self.norm_bounder[0][1]:
                self.norm_bounder[0][1] = i[0][1]
            if i[1][0] < self.norm_bounder[1][0]:
                self.norm_bounder[1][0] = i[1][0]
            if i[1][1] > self.norm_bounder[1][1]:
                self.norm_bounder[1][1] = i[1][1]
            if i[2][0] < self.norm_bounder[2][0]:
                self.norm_bounder[2][0] = i[2][0]
            if i[2][1] > self.norm_bounder[2][1]:
                self.norm_bounder[2][1] = i[2][1]
        return [self.bounder, self.norm_bounder]

    def calculate_fuzzy_feature(self):
        self.norm_fuzzy_feature = []
        for (i, point_seq) in enumerate(self.norm_point_seq):
            self.norm_fuzzy_feature.append([])
            for point in point_seq:
                feature = []
                for axi in range(0,3):
                    feature.append(fuzzy(point[axi], self.norm_bounder[axi][0], self.norm_bounder[axi][1]))
                feature = np.array(feature).flatten()
                self.norm_fuzzy_feature[i].append(feature)
        self.norm_fuzzy_feature = np.array(self.norm_fuzzy_feature)
        return self.norm_fuzzy_feature

    '''
