#!/usr/bin/env python
"""
Description: Check whether the train dataset is containing non NaN label
Email: gzynus@gmail.com
Author: Zongyi
"""
import os
import numpy as np


def check_train_dataset(data_path):
    """
    check train data set
    :param data_path:
    :return:
    """
    labels = ['adas13', 'dx', 'ventricles']
    portion = [6, 6, 4]
    for test_fold in range(10):
        for label in labels:
            windows = portion[labels.index(label)]
            for window in range(1, windows+1):
                # get train dataset path
                tr_set = 'fold' + str(test_fold) + '_train_' + label + '_' + \
                         str(window) + '.npy'
                tr_path = os.path.join(data_path, str(test_fold), tr_set)
                tr_matrix = np.load(tr_path)
                non_nan = np.sum(~np.isnan(tr_matrix[:, 0]))
                print('For label', label, 'window', window, tr_matrix.shape, non_nan)


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    check_train_dataset(data_path)

