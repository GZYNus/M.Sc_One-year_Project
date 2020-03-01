#!/usr/bin/env python
"""
Description: generate training, validation and test set
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import csv
import copy
import datetime
import pickle
import numpy as np
from lib.helper import list2csv
from lib.helper import sort_dates
from lib.load_data import loading_subject_list
from lib.load_data import loading_baseline_dates
from lib.calc_date import month_between


def gen_train_set(data_path, raw_data_path, flag):
    """
    To generate train set, including full and split train set
    :param data_path:
    :param raw_data_path:
    :param flag:
    :return:
    """
    assert flag in ('train', 'split_train')
    # loading baseline_dates
    baselines_dates_path = os.path.join(data_path, 'zongyi_baseline_dates.pkl')
    baseline_dates = loading_baseline_dates(baselines_dates_path)
    # loading subjects list
    if flag is 'train':
        sub_list = loading_subject_list(raw_data_path, flag)
        print(len(sub_list))
    else:
        # flag is split_train
        full_train_subs = loading_subject_list(raw_data_path, 'train')
        dev_sub_list = loading_subject_list(raw_data_path, 'dev')
        sub_list = list(set(full_train_subs).difference(set(dev_sub_list)))
        print(len(sub_list))
    data_dict = {}
    data_list = {}
    for sub in sub_list:
        data_dict[sub] = {}
        data_list[sub] = []
    # read data.csv
    data_csv_path = os.path.join(data_path, 'data.csv')
    with open(data_csv_path) as file:
        data_csv = csv.reader(file)
        data_header = next(data_csv)
        for row in data_csv:
            if int(row[0]) in sub_list:
                data_list[int(row[0])].append(row)
    # write to data_dict
    for sub in sorted(sub_list):
        baseline_date = baseline_dates[sub]
        data_lists = sort_dates(data_list[sub])
        numPoints = len(data_lists)
        datapoint_index = []
        for tp in range(numPoints):
            current_date = data_lists[tp][1]
            current_date = datetime.datetime.strptime(current_date,
                                                      "%Y-%m-%d")
            datapoint_index.append(month_between(current_date, baseline_date))
        time_length = max(datapoint_index) + 1
        truth = np.full((time_length, 24), np.nan)
        # fill in the truth and mask with data points if exist
        for tp in range(numPoints):
            for feature in range(2, 25):
                if data_lists[tp][feature] is not 'Nan':
                    truth[datapoint_index[tp], feature-1] = \
                        data_lists[tp][feature]
        mask = ~np.isnan(truth)
        data_dict[sub]['truth'] = copy.deepcopy(truth)
        data_dict[sub]['mask'] = copy.deepcopy(mask)
    # save to disk
    data_savename = 'lijun_' + flag + '_nointerp.pkl'
    data_savepath = os.path.join(data_path, data_savename)
    file = open(data_savepath, 'wb')
    pickle.dump(data_dict, file)
    file.close()
    print('Finish generating dataset ' + flag + '!')


def gen_eval_set(data_path, raw_data_path, flag):
    """
    generate dev and test set
    Note that we need to split them into half
    :param data_path:
    :param raw_data_path:
    :param flag:
    :return:
    """
    assert flag in ('dev', 'test')
    # loading baseline_dates
    baselines_dates_path = os.path.join(data_path, 'lijun_baseline_dates.pkl')
    baseline_dates = loading_baseline_dates(baselines_dates_path)
    # loading subjects list
    sub_list = loading_subject_list(raw_data_path, flag)
    data_dict = {}
    data_list = {}
    for sub in sub_list:
        data_dict[sub] = {}
        data_list[sub] = []
    # read data.csv
    data_csv_path = os.path.join(data_path, 'data.csv')
    with open(data_csv_path) as file:
        data_csv = csv.reader(file)
        data_header = next(data_csv)
        for row in data_csv:
            if int(row[0]) in sub_list:
                data_list[int(row[0])].append(row)
    # write to data_dict
    for sub in sorted(data_dict.keys()):
        baseline_date = baseline_dates[sub]
        data_lists = sort_dates(data_list[sub])
        numPoints = len(data_lists)
        datapoint_index = []
        for tp in range(numPoints):
            current_date = data_lists[tp][1]
            current_date = datetime.datetime.strptime(current_date,
                                                      "%Y-%m-%d")
            datapoint_index.append(
                month_between(current_date, baseline_date))
        time_length = max(datapoint_index) + 1
        # get split_index
        if numPoints == 1:
            split_index = 1
        elif numPoints % 2 == 0:
            split_index = datapoint_index[int(numPoints / 2)]
        else:
            split_index = datapoint_index[int((numPoints + 1) / 2)]
        truth = np.full((time_length, 24), np.nan)
        # fill in the truth and mask with data points if exist
        for tp in range(numPoints):
            for feature in range(2, 25):
                if data_lists[tp][feature] is not 'Nan':
                    truth[datapoint_index[tp], feature - 1] = \
                        data_lists[tp][feature]
        mask = ~np.isnan(truth)
        data_dict[sub]['truth'] = copy.deepcopy(truth[:split_index, :])
        data_dict[sub]['mask'] = copy.deepcopy(mask[:split_index, :])
        data_dict[sub]['length'] = copy.deepcopy(time_length)
        data_dict[sub]['split_index'] = copy.deepcopy(split_index)

    # save to disk
    data_savename = 'lijun_' + flag + '_nointerp.pkl'
    data_savepath = os.path.join(data_path, data_savename)
    file = open(data_savepath, 'wb')
    pickle.dump(data_dict, file)
    file.close()
    print('Finish generating dataset ' + flag + '!')


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    raw_data_path = os.path.join(root_path, 'raw_data')
    gen_train_set(data_path, raw_data_path, 'train')
    gen_train_set(data_path, raw_data_path, 'split_train')
    gen_eval_set(data_path, raw_data_path, 'dev')
    gen_eval_set(data_path, raw_data_path, 'test')
