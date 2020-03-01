#!/usr/bin/env python
"""
Description: data imputation, using linear imputation method
Email: gzynus@gmail.com
Author: Zongyi Guo & Lijun An
"""
import os
import pickle
import copy
import numpy as np
import pandas as pd
from lib.load_data import loading_original_data
from lib.load_data import loading_means_stds
from lib.load_data import loading_fields_list


def linear_filling(input_nointerp__path, output_interp_path, flag):
    """
    Implement linear interpolation
    """
    assert flag in ('train', 'test', 'split_train', 'dev'), \
        'Wrong flag, only "train" , "split_train", "dev" or "test" is ' \
        'acceptable!'
    # loading original data
    data = loading_original_data(input_nointerp__path, flag)   # load pkl data: _nointerp.pkl
    means_data, _ = loading_means_stds(input_nointerp__path, flag)
    fields_list = loading_fields_list(input_nointerp__path)
    interp = {}
    for key in sorted(data.keys()):
        # read the original data
        truth = copy.deepcopy(data[key]['truth'])
        interp[key] = {}
        interp[key]['mask'] = copy.deepcopy(data[key]['mask'])
        interp[key]['truth'] = copy.deepcopy(truth)
        if flag in ('test', 'dev'):
            interp[key]['length'] = copy.deepcopy(data[key]['length'])
            interp[key]['split_index'] = copy.deepcopy(data[key]['split_index'])
        length = np.shape(truth)[0]
        inputs = np.full((length, 24), np.nan)
        for month in range(length):  # for month_bl
            inputs[month, 0] = int(month)
        for i in range(23):                         # 第一个月用平均值代替
            if np.isnan(truth[0, i + 1]):
                truth[0, i + 1] = means_data[fields_list[i + 1]]
        # linear interpolation for each subject
        inputs = linear_filling_individual(inputs, truth)    # 这里传的是每一个sub的inputs和truth
        interp[key]['input'] = inputs

    file = open(os.path.join(
        output_interp_path, 'zongyi_' + flag + '_linear.pkl'), 'wb')
    pickle.dump(interp, file)
    file.close()
    print('Finish Linear Interpolation' + os.path.join(
        output_interp_path, 'zongyi_' + flag + '_linear.pkl'))


def linear_filling_individual(inputs, truth_data):
    """
    linear interpolation for each subject
    :param inputs:
    :param truth_data:
    :return:
    """
    status_matrix = get_status_matrix(truth_data)
    length = truth_data.shape[0]
    for j in range(length):
        if status_matrix[j, 0] == 1:        # 若有部分valid data 那么会进这个循环
            # implement linear interpolation
            for feature in range(1, 24):
                linear_filling_individual_feature(inputs,
                                                  truth_data,
                                                  feature,
                                                  status_matrix,
                                                  j)
        elif status_matrix[j, 0] == 2:
            inputs[j, 1:] = truth_data[j, 1:]

    return inputs


def linear_filling_individual_feature(inputs, truth_data, feature, status_matrix, j):
    """
    linear interpolation for each subject, each feature
    """
    if ~np.isnan(truth_data[j, feature]):   # 有值的话 直接给valid data
        inputs[j, feature] = truth_data[j, feature]   # 对一个月的一个feature进行处理(外面循环了24次)
    else:
        # check whether we could use linear interpolation
        if ~np.isnan(status_matrix[j, 2 * feature]):
            # we could use linear interpolation
            last_month = int(status_matrix[j, 2 * feature - 1])
            next_month = int(status_matrix[j, 2 * feature])
            last_month_value = truth_data[last_month, feature]
            next_month_value = truth_data[next_month, feature]
            xp = [last_month, next_month]
            fp = [last_month_value, next_month_value]
            inputs[j, feature] = np.interp(j, xp, fp)
            # inputs[j, feature] = truth_data[last_month, feature]
            if feature == 1:
                inputs[j, feature] = int(inputs[j, feature])


def get_status_matrix(truth_data):
    """
    get the data status for a subject
    if status_matrix[month, 0] = 0, means the features at this month are all Nan
    if status_matrix[month, 0] = 1, means the features at this month are partial Nan
    if status_matrix[month, 0] = 1,
    we need to find its last and next corresponding feature's  month id
    if status_matrix[month, 0] = 2, means the features at this month all exist, No value is Nan
    """
    length = np.shape(truth_data)[0]
    status_matrix = np.full((length, 47), np.nan)
    # 2 positions for one features, total 23 features, 1 + 2 * 23 = 47
    for j in range(length):
        if np.isnan(truth_data[j, 1:]).sum() == 23:   # status_matrix的第一个数存的是这一行有没有valid data
            status_matrix[j, 0] = 0
        elif np.isnan(truth_data[j, 1:]).sum() == 0:
            status_matrix[j, 0] = 2
        else:
            status_matrix[j, 0] = 1
    # find its last and next feature point
    for j in range(length):
        if status_matrix[j, 0] == 1:                # 只对这一个月不全缺失的月份 做filling 全为nan的不做filling
            # backward search
            for feature in range(1, 24):            # 固定一个feature 遍历所有月份
                for k in range(j - 1, -1, -1):      # 倒数到0  步长为-1
                    if ~np.isnan(truth_data[k, feature]):
                        # the feature exists at k
                        status_matrix[j, 2*feature - 1] = k       # 找到往前最近的有值的那个月k
                        break
            # forward search
            for feature in range(1, 24):
                for k in range(j + 1, length):
                    if ~np.isnan(truth_data[k, feature]):
                        # the feature exists at k
                        status_matrix[j, 2 * feature] = k
                        break
    return status_matrix


if __name__ == '__main__':
    # root path for the project
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    linear_filling(data_path, data_path, flag='split_train')
    linear_filling(data_path, data_path, flag='dev')
    linear_filling(data_path, data_path, flag='test')
    # linear_filling(data_path, data_path, flag='train')