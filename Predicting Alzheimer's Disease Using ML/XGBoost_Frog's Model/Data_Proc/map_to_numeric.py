#!/usr/bin/env python
"""
Description: Map gender and marriage status to numeric variable
Email: gzynus@gmail.com
Author: Zongyi
"""
import os
import pandas as pd
import numpy as np


def Gender_conv(value):
    """
    convert gender, if gender is missing, default value is 0
    :param value:
    :return:
    """
    if value == 'Male':
        return 0.
    if value == 'Female':
        return 1.
    return 0.


def Marriage_conv(value):
    """
    convert marriage, if marriage is missing, default is 4(Unknown)
    :param value:
    :return:
    """
    if value == 'Married':
        return 0.
    if value == 'Widowed':
        return 1.
    if value == 'Divorced':
        return 0.
    if value == 'Never married':
        return 1.
    if value == 'Unknown':
        return 1.
    return 4.


CONVERTERS = {
    'GENDER': Gender_conv,
    'MARRY': Marriage_conv
}


def map_to_numeric(data_path, test_fold):
    """
    map the gender and marriage status to numeric
    Gender:
        Male: 0
        Female: 1
    Marriage status:
        Married: 0
        Widowed: 1
        Divorced: 2
        Never married: 3
        Unknown: 4
    :param data_path:
    :param test_fold:
    :return:
    """
    print('mapping for test fold', test_fold)
    # we need to read train_long, val_pred and test pred
    train_long_path = os.path.join(data_path, str(test_fold),
                                   'fold' + str(test_fold) + '_train_long.csv')
    val_pred_path = os.path.join(data_path, str(test_fold),
                                 'fold' + str(test_fold) + '_val_pred.csv')
    test_pred_path = os.path.join(data_path, str(test_fold),
                                  'fold' + str(test_fold) + '_test_pred.csv')
    train_long = pd.read_csv(train_long_path, converters=CONVERTERS)
    # delete D1, D2 for train_long, it is useless
    train_long.drop(columns=['D1', 'D2'], inplace=True)
    val_pred = pd.read_csv(val_pred_path, converters=CONVERTERS)
    test_pred = pd.read_csv(test_pred_path, converters=CONVERTERS)

    # save it to csv
    train_long.to_csv(train_long_path, sep=',', index=False)
    val_pred.to_csv(val_pred_path, sep=',', index=False)
    test_pred.to_csv(test_pred_path, sep=',', index=False)

if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    for test_fold in range(10):
        map_to_numeric(data_path, test_fold)

