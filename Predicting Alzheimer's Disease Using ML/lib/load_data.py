#!/usr/bin/env python
"""
Description: loading dataset
Email: gzynus@gmail.com
Author: Zongyi Guo & Lijun An
"""
import os
import pickle
import numpy as np


def loading_baseline_dates(baseline_dates_path):
    """
    loading baseline_dates
    """
    # load baseline_dates.pkl
    try:
        with open(baseline_dates_path, 'rb') as baseline:
            baseline_dates = pickle.load(baseline, encoding='iso-8859-1')
        return baseline_dates
    except IOError:
        print('File is not accessible:' + baseline_dates_path)


def loading_subject_list(data_path, flag):
    """
    loading subjects list
    :param data_path:
    :param flag:
    :return:
    """
    assert flag in ('train', 'test', 'dev', 'full')
    file_name = flag + '_subject_list.txt'
    sub_list_path = os.path.join(data_path, file_name)
    sub_list = []
    file = open(sub_list_path)
    line = file.readline()
    while line:
        sub_list.append(int(line.strip()))
        line = file.readline()
    file.close()
    return sub_list


def loading_fields_list(data_path):
    """
    loading fields_list and baselines_dates
    :param data_path:
    :return:
    """
    fields_list_path = os.path.join(data_path, 'tutorial_fields.pkl')

    # loading fields list
    try:
        with open(fields_list_path, 'rb') as fields:
            fields_list = pickle.load(fields, encoding='iso-8859-1')
        return fields_list
    except IOError:
        print('File is not accessible:' + fields_list_path)


def loading_means_stds(data_path, flag):
    """
    loading means, stds fields_list, baseline_dates
    :return:
    """
    assert flag in ('train', 'split_train', 'dev', 'test'), \
        'Wrong Flag'
    if flag in ('train', 'test'):
        means_path = os.path.join(data_path, 'lijun_means_train.pkl')
        stds_path = os.path.join(data_path, 'lijun_stds_train.pkl')
    elif flag in ('split_train', 'dev'):
        means_path = os.path.join(data_path, 'lijun_means_split_train.pkl')
        stds_path = os.path.join(data_path, 'lijun_stds_split_train.pkl')

    try:
        with open(means_path, 'rb') as means:
            means_data = pickle.load(means, encoding='iso-8859-1')
        means_data['DX'] = 0  # add a 'mean' for DX
    except IOError:
        print('File is not accessible:' + means_path)
    # loding stds
    try:
        with open(stds_path, 'rb') as stds:
            stds_data = pickle.load(stds, encoding='iso-8859-1')
    except IOError:
        print('File is not accessible:' + stds_path)

    return means_data, stds_data


def loading_original_data(data_path, flag):
    """
    loading original(uninterpolated) data
    :param data_path:
    :param flag:
    :return:
    """
    assert flag in ('train', 'test', 'split_train', 'dev'), \
        'Wrong flag, only "train", "test", "split_train" ' \
        'or "dev" is acceptable!'
    file_name = 'zongyi_' + flag + '_nointerp.pkl'
    original_data_path = os.path.join(data_path, file_name)
    # check if file is existing
    try:
        with open(original_data_path, 'rb') as file:
            data = pickle.load(file, encoding='iso-8859-1')
        return data
    except IOError:
        print('File is not accessible:' + original_data_path)


def loading_interpolated_data(data_path, method, flag):
    """
    loading interpolated data(.pkl file)
    :param data_path:
    :param flag:
    :param method:
    :return:
    """
    name = 'Zongyi_' + flag + '_' + method + '.pkl'
    interp_data_path = os.path.join(data_path, name)
    # print(interp_data_path)
    try:
        with open(interp_data_path, 'rb') as file:
            data = pickle.load(file, encoding='iso-8859-1')
        return data
    except IOError:
        print('File is not accessible:' + interp_data_path)

              #data是一个pkl文件


def loading_znormalized_data(data_path, method, flag):
    # check input is legal
    assert flag in ('train', 'test', 'split_train', 'dev'), \
        'Wrong flag, ' \
        'only "train", "test", "split_train" or "dev" is ' \
        'acceptable!'
    assert method in ('nointerp', 'forward', 'linear'), \
        'Wrong flag, only "nointerp", "forward" or "linear" is acceptable!'
    # load interpolated data
    data = loading_interpolated_data(data_path, method, flag)
    # load means, stds and fields_list
    means, stds = loading_means_stds(data_path, flag)
    fields_list = loading_fields_list(data_path)
    baseline_dates = loading_baseline_dates(os.path.join(data_path,
                                                         'Zongyi_baseline_dates.pkl'))
    data_list = []
    mask_list = []
    baseline_list = []
    id_list = []
    length_list = []
    split_index_list = []
    for k in sorted(data.keys()):
        personal_data = np.delete(data[k]['input'], 0, axis=1)
        # z-normalization
        for feature in range(2, 24):
            personal_data[:, feature - 1] = (personal_data[:, feature - 1]
                                             - means[fields_list[feature]]) \
                                            / stds[fields_list[feature]]
        id_list.append(k)
        baseline_list.append(baseline_dates[k])
        data_list.append(personal_data)
        mask_list.append(np.delete(data[k]['mask'], 0, axis=1))
        if flag in ('dev', 'test'):
            length_list.append(data[k]['length'])
            split_index_list.append(data[k]['split_index'])

    return data_list, mask_list, baseline_list, id_list, length_list, split_index_list


# data loader
class TrainDataSet:
    """
    Data loader for training data
    """
    def __init__(self, inputs, masks):        #初始化要传入的两个参数 这里传入的input——tensor 和 mask--tensor
        self.inputs = inputs
        self.masks = masks

    def __getitem__(self, item):              #这里可以调用getitem函数 其实也是dataset这个抽象类需要继承的一个函数  传入item 可以提取其input——tensor 和mask——tensor
        input_tensor = self.inputs[item, :, :]
        mask_tensor = self.masks[item, :, :]

        return input_tensor, mask_tensor

    def __len__(self):                        #这个是dataset抽象类需要继承的第二个函数 长度
        return self.inputs.shape[0]



if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'raw_data')
    print(len(loading_subject_list(data_path, flag='test')))