#!/usr/bin/env python
"""
Description:
Fill the missing value for 113 input features
    1. for the feature terms, fill mean
    2. for the time related terms, fill 999
Email: gzynus@gmail.com
Author: Zongyi
"""
import os
import pickle
import pandas as pd
import numpy as np
from lib.misc import load_feature


def fill_missing_value(data_path, test_fold):
    """
    1. Fill the missing value for 113 input features
        1. for the feature terms, fill mean
        2. for the time related terms, fill 999
    :param data_path:
    :param test_fold:
    :return:
    """
    print('filling test fold', test_fold)
    # we need to read train_long, val_pred and test pred
    train_long_path = os.path.join(data_path, str(test_fold),
                                   'fold' + str(test_fold) +
                                   '_train_long_norm.csv')
    train_long = pd.read_csv(train_long_path)
    val_pred_path = os.path.join(data_path, str(test_fold),
                                 'fold' + str(test_fold) + '_val_pred.csv')
    val_pred = pd.read_csv(val_pred_path)
    test_pred_path = os.path.join(data_path, str(test_fold),
                                  'fold' + str(test_fold) + '_test_pred.csv')
    test_pred = pd.read_csv(test_pred_path)
    # save_path
    # we need to read train_long, val_pred and test pred
    filled_train_long_path = os.path.join(data_path, str(test_fold),
                                          'fold' + str(test_fold) +
                                          '_train_long_filled.csv')
    filled_val_pred_path = os.path.join(data_path, str(test_fold),
                                        'fold' + str(test_fold) +
                                        '_val_pred_filled.csv')
    filled_test_pred_path = os.path.join(data_path, str(test_fold),
                                         'fold' + str(test_fold) +
                                         '_test_pred_filled.csv')
    # means and stds
    means_stds_path = os.path.join(data_path, str(test_fold),
                                   'fold' + str(test_fold) + '_means_stds.pkl')
    with open(means_stds_path, 'rb') as file:
        means_stds = pickle.load(file)
    file.close()
    time_features = ['time_since_mr_dx', 'time_since_best_dx',
                     'time_since_worst_dx', 'time_since_midler',
                     'time_since_mr_ADAS13', 'time_since_low_ADAS13',
                     'time_since_high_ADAS13', 'time_since_mr_Ventricles',
                     'time_since_low_Ventricles', 'time_since_high_Ventricles',
                     'time_since_mr_Fusiform', 'time_since_low_Fusiform',
                     'time_since_high_Fusiform', 'time_since_mr_WholeBrain',
                     'time_since_low_WholeBrain', 'time_since_high_WholeBrain',
                     'time_since_mr_Hippocampus', 'time_since_low_Hippocampus',
                     'time_since_high_Hippocampus', 'time_since_mr_MidTemp',
                     'time_since_low_MidTemp', 'time_since_high_MidTemp',
                     'time_since_mr_ICV', 'time_since_low_ICV',
                     'time_since_high_ICV', 'time_since_mr_MMSE',
                     'time_since_low_MMSE', 'time_since_high_MMSE',
                     'time_since_mr_CDRSB', 'time_since_low_CDRSB',
                     'time_since_high_CDRSB', 'time_since_mr_FDG',
                     'time_since_low_FDG', 'time_since_high_FDG',
                     'time_since_mr_ADAS11', 'time_since_low_ADAS11',
                     'time_since_high_ADAS11', 'time_since_mr_RAVLT_im',
                     'time_since_low_RAVLT_im', 'time_since_high_RAVLT_im',
                     'time_since_mr_RAVLT_learn', 'time_since_low_RAVLT_learn',
                     'time_since_high_RAVLT_learn', 'time_since_mr_RAVLT_forget',
                     'time_since_low_RAVLT_forget', 'time_since_high_RAVLT_forget']
    # load features
    # input_features = load_feature()
    input_features = list(means_stds.keys())
    input_features.remove('curr_adas13')
    input_features.remove('curr_ventricles')
    for f in input_features:
        # # replace each column's NAN with corresponding mean for feature
        if f in time_features:
            train_long[f].replace(np.nan, 999, inplace=True)
            val_pred[f].replace(np.nan, 999, inplace=True)
            test_pred[f].replace(np.nan, 999, inplace=True)
        else:
            train_long[f].replace(np.nan, means_stds[f]['mean'], inplace=True)
            val_pred[f].replace(np.nan, means_stds[f]['mean'], inplace=True)
            test_pred[f].replace(np.nan, means_stds[f]['mean'], inplace=True)
        if np.sum(np.isnan(np.array(train_long[f]))) != 0:
            print('*****************feature is', f)
            print('******************',
                  np.sum(np.isnan(np.array(train_long[f]))))
    # save it
    train_long.to_csv(filled_train_long_path, sep=',', index=False)
    val_pred.to_csv(filled_val_pred_path, sep=',', index=False)
    test_pred.to_csv(filled_test_pred_path, sep=',',index=False)

if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    for test_fold in range(10):
        fill_missing_value(data_path, test_fold)