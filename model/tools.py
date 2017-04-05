# coding=utf-8
from unittest import TestCase
import numpy as np
import random

# convert float sample to continuous 0s,1s format
def sample_convertor(sample, tau):
    converted_sample = []
    temp = ''
    for idx, value in enumerate(sample):
        current_pos_val = 1 if sample[idx] >= tau else 0
        if temp == '':
            temp = str(current_pos_val)
        elif int(temp[-1]) == current_pos_val:
            temp += str(current_pos_val)
        else:
            converted_sample.append(temp)
            temp = str(current_pos_val)
    converted_sample.append(temp)
    return converted_sample

def begin_locator(converted_sample, max_proposal_length, gamma=0):
    begin = 0
    index = []
    for idx, continuous_section in enumerate(converted_sample):
        if continuous_section[0] == '1':
            index.extend(begin_end_locator(converted_sample[idx:], max_proposal_length, begin, gamma))
        begin += len(continuous_section)

    return index

# return begin_end pos tuples for proposals
def begin_end_locator(converted_sample,  max_proposal_length, begin_pos, gamma=0):
    zero_count = 0
    end_pos = begin_pos
    index = []

    for continuous_section in converted_sample:
        # print idx
        # print continuous_section
        if continuous_section[0] == '0':
            if (end_pos-begin_pos+len(continuous_section)) * gamma > len(continuous_section)+zero_count:
                end_pos += len(continuous_section)
                zero_count += len(continuous_section)
            else:
                if end_pos-begin_pos < max_proposal_length:
                    index.append((begin_pos, end_pos))
                break
        else:
            end_pos += len(continuous_section)

    return index

#
def ground_truth_locator(sequence, background = 0.0):
    last_alpha = ''
    begin_pos = 0
    end_pos = 0
    valid_length = 0
    index = []
    for idx, alpha in enumerate(sequence):
        if alpha == last_alpha:
            valid_length+=1
        else:
            end_pos = begin_pos + valid_length
            if end_pos > begin_pos:
                if last_alpha != background:
                    index.append((begin_pos, end_pos, last_alpha))
            begin_pos = end_pos
            last_alpha = alpha
            valid_length = 1
    end_pos = begin_pos + valid_length
    if last_alpha != background:
        index.append((begin_pos, end_pos, last_alpha))
    return index

def proposal_generator(samples_list):
    tau_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    gamma_list = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    # tau_list = [0.5]
    # gamma_list = [0.1]
    tau_gamma_list = []
    index = []
    max_proposal_length = 100
    # length = []
    # temp_length = 0
    for t in tau_list:
        for g in gamma_list:
            tau_gamma_list.append((t, g))
    for idx, sample in enumerate(samples_list):
        temp_index = []
        for tau_gamma in tau_gamma_list:
            tau = tau_gamma[0]
            gamma = tau_gamma[1]
            converted_sample = sample_convertor(sample, tau)
            # print tau, gamma, converted_sample
            begin_end_index = begin_locator(converted_sample, max_proposal_length, gamma)
            temp_index.extend(begin_end_index)
            # index.append(begin_end_index)
        index.append(list(set(temp_index)))
        # length.append((temp_length, len(index)))
        # temp_length = len(index)
        # sample_proposal_length = len(index) - temp_length
        # temp_length = len(index)
        # length.append(sample_proposal_length)
    # return index, length
    return index

def valid_score(squence, beg_pos, end_pos):
    score = sum(squence[beg_pos:end_pos])/(end_pos - beg_pos)
    # print squence, (beg_pos, end_pos, score)
    return score

def IOU(begin_pos1, end_pos1, begin_pos2, end_pos2):
    if end_pos1 <= begin_pos2 or end_pos2 <= begin_pos1:
        return 0
    else:
        return 1.0*(min(end_pos1, end_pos2) - max(begin_pos1, begin_pos2)) / (max(end_pos1, end_pos2) - min(begin_pos1, begin_pos2))

def self_NMS(proposal_list, sample, threshold=0.95):
    remove_count = 0
    for idx, proposal in enumerate(proposal_list):
        proposal_length = len(proposal)
        remove_list = []
        # print proposal
        for i in range(0, proposal_length):
            for j in range(i + 1, proposal_length):
                iou = IOU(proposal[i][0],
                          proposal[i][1],
                          proposal[j][0],
                          proposal[j][1])
                # print iou
                if iou >= threshold:
                    valid_score1 = valid_score(sample[idx], proposal[i][0], proposal[i][1])
                    valid_score2 = valid_score(sample[idx], proposal[j][0], proposal[j][1])
                    if valid_score1 > valid_score2:
                        remove_list.append(proposal[j])
                    else:
                        remove_list.append(proposal[i])
        remove_list = list(set(remove_list))
        remove_count += len(remove_list)
        for item in remove_list:
            proposal.remove(item)
    return remove_count

def train_proposals(label_seq, proposals, snippet_size):
    ground_truths = ground_truth_locator(label_seq)
    index = []
    ground_truth_number = len(ground_truths)
    for proposal in proposals:
        count = 0
        for ground_truth in ground_truths:
            if IOU(ground_truth[0]//snippet_size, ground_truth[1]//snippet_size, proposal[0], proposal[1]) > 0.7:
                index.append([proposal[0]*snippet_size, proposal[1]*snippet_size, int(ground_truth[2])])
            elif IOU(ground_truth[0]//snippet_size, ground_truth[1]//snippet_size, proposal[0], proposal[1]) < 0.05:
                count += 1
            if count == ground_truth_number :
                index.append([proposal[0]*snippet_size, proposal[1]*snippet_size, 0])

    return index



class proposal_test(TestCase):

    str1 = '001101111110011'
    str2 = '001100110000000'
    str3 = ['0000','11111','0000000','1111']

    sample_list = [[0.1,0.3,0.5,0.6,0.2,0.7,0.8,0.9,0.8,0.2,0.3,0.4,0.5,0.6],
                   [0.1,0.2,0.6,0.7,0.3,0.4,0.6,0.8,0.1,0.2,0.3,0.4,0.5,0.7],
                   [0.2,0.3,0.4,0.7,0.8,0.6,0.4,0.5,0.5,0.6,0.1,0.2,0.8,0.8]]

    def test1(self):
        proposal_list = proposal_generator(self.sample_list)
        print(proposal_list)
        print(self_NMS(proposal_list, self.sample_list, 0.8))
        print(proposal_list)


    def test2(self):

        proposal_list = [[(5, 9), (6, 9), (12, 14), (13, 14), (0, 14), (1, 14), (1, 4), (5, 14), (10, 14), (11, 14), (2, 14), (3, 4),
         (2, 4), (7, 8)],
        [(7, 8), (0, 0), (9, 14), (6, 8), (13, 14), (0, 14), (1, 14), (1, 8), (10, 14), (11, 14), (2, 8), (2, 14),
         (3, 4), (2, 4), (5, 8), (12, 14)],
        [(0, 0), (12, 14), (3, 6), (0, 14), (1, 14), (2, 10), (9, 10), (4, 5), (0, 10), (11, 14), (7, 10), (1, 10),
         (2, 14), (3, 14), (3, 5)]]
        a = '1111555550066666660444440555555000000001'
        print(len(a))
        ground_truth_list = ground_truth_locator(a)
        index = []
        for idx, ground_truths in ground_truth_list:
            proposals = proposal_list[idx]
            cat_index = []
            for ground_truth in ground_truths:
                for proposal in proposals:
                    if IOU(ground_truth[0], ground_truth[1], proposal[0], proposal[1]) > 0.7:
                        cat_index.append((proposal[0], proposal[1], int(ground_truth[2])))
                    elif IOU(ground_truth[0], ground_truth[1], proposal[0], proposal[1]) < 0.05:
                        for ground_truth_test in ground_truths:
                            if IOU(ground_truth_test[0], ground_truth_test[1], proposal[0], proposal[1]) >= 0.05:
                                break
                            cat_index.append((proposal[0], proposal[1], 0))
                        pass
            index.append(cat_index)
        return index

    def test5(self):
        index = proposal_generator(self.sample_list)
        print(index)
