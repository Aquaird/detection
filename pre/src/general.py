import os
import numpy as np
import csv
from sklearn.decomposition import PCA

# calculate the var in one seq with window_size
def calculate_var(point_seq, window_size):
    point_var = []
    window_var = []
    window = []
    for index_of_point in range(0, 20):
        point_var.append([])
    for index_of_frame in range(0, round(len(point_seq[0]) / window_size)):
        window_var.append([])

    for index_of_point in range(0, len(point_seq)):
        point_data = point_seq[index_of_point]
        for index_of_frame in range(0, len(point_data)):
            if index_of_frame != 0 and index_of_frame % window_size == 0:
                var = numpy.var(window)
                # print(window, var)
                point_var[index_of_point].append(var)
                window_var[round(index_of_frame / window_size) - 1].append(var)
                window = []
            window.append(point_data[index_of_frame])
    return [point_var, window_var]


# smooth the sequence with (-3,12,17,12,-3)/3
def savitzky_filter(sequence):
    num_element = len(sequence)
    temp_seq = []
    for i in range(0, num_element):
        if (i < 2 or i + 2 >= num_element):
            temp = sequence[i]
        else:
            temp = [0.0,0.0,0.0]
            for j in range(len(temp)):
                temp[j] = (-3.0*(sequence[i - 2][j] + sequence[i + 2][j]) + 12.0*(sequence[i - 1][j]+sequence[i + 1][j]) + 17.0*sequence[i][j]) / 35.0
        temp_seq.append(temp)

    return temp_seq

# smooth the sequence with (-3,12,17,12,-3)/3
def savitzky_one_dim(sequence):
    num_element = len(sequence)
    temp_seq = []
    for i in range(0, num_element):
        if (i < 2) or (i + 2 >= num_element):
            temp = sequence[i]
        else:
            temp = (-3.0*(sequence[i - 2] + sequence[i + 2])
                    + 12.0*(sequence[i - 1]+sequence[i + 1])
                    + 17.0*sequence[i]) / 35.0
        temp_seq.append(temp)

    return temp_seq


def taylor_one_dim(sequence):
    num_element = len(sequence)
    temp_seq = []
    for i in range(1, num_element):
        if i+1 >= num_element:
            temp = temp_seq[i-1]
        else:
            temp = (sequence[i+1] - sequence[i-1])/(0.125 * 2.0)
        if i == 1:
            temp_seq.append(temp)

        temp_seq.append(temp)

    return temp_seq

def taylor_transfer(sequence):
    num_element = len(sequence)
    num_axi = len(sequence[0])
    temp_seq = []
    for i in range(1, num_element):
        if(i+1 >= num_element):
            temp = temp_seq[i-1]
        else:
            temp = [0.0,0.0,0.0]
            for j in range(num_axi):
                temp[j] = (sequence[i+1][j] - sequence[i-1][j])/(0.125 * 2.0)
        if(1 == i):
            temp_seq.append(temp)

        temp_seq.append(temp)

    return temp_seq


def gaussion_curve(i, s, stdvv):
    temp = - float((abs(s-i) **2)) / float(2*stdvv*stdvv)
    return np.exp(temp)


##############################  useless function #################################
'''
# calculate bounder of a sequence of (x,y,z)
def calculate_bounder(sequence):
    bounder = [[float("inf"),float("-inf")],[float("inf"),float("-inf")],[float("inf"),float("-inf")]]
    for i in sequence:
        x_value = i[0]
        y_value = i[1]
        z_value = i[2]
        if x_value < bounder[0][0]:
            bounder[0][0] = x_value
        if x_value > bounder[0][1]:
            bounder[0][1] = x_value
        if y_value < bounder[1][0]:
            bounder[1][0] = y_value
        if y_value > bounder[1][1]:
            bounder[1][1] = y_value
        if z_value < bounder[2][0]:
            bounder[2][0] = z_value
        if z_value > bounder[2][1]:
            bounder[2][1] = z_value
    return bounder

# get pca components of data sequence
def PCA_vector(data):
    pca = PCA(n_components=3)
    data_t = pca.fit_transform(data)
    return pca.components_

# calculate fuzzy(a) of the given a and its bounder(min, max)
def fuzzy(a, min_bounder, max_bounder):
    q = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    q[0] = min_bounder
    q[8] = max_bounder
    q[4] = 1.0 * (q[0] + q[8]) / 2.0
    q[2] = 1.0 * (q[0] + q[4]) / 2.0
    q[1] = 1.0 * (q[0] + q[2]) / 2.0
    q[3] = 1.0 * (q[2] + q[4]) / 2.0
    q[6] = 1.0 * (q[8] + q[4]) / 2.0
    q[7] = 1.0 * (q[6] + q[8]) / 2.0
    q[5] = 1.0 * (q[4] + q[6]) / 2.0

    fuzzy_value = [0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0]
    for i in range(0,8):
        if(a > q[i]):
            if(a > q[i+1]):
                fuzzy_value[i] = 0
            else:
                fuzzy_value[i] = maxmin_linear(a,q[i-1],q[i])
        else:
            fuzzy_value[i] = 1
   
    return fuzzy_value

def maxmin_linear(x, min_bounder, max_bounder):
    return 1.* (x-min_bounder) / (max_bounder - min_bounder)

'''
