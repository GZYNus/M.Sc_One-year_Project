#!/usr/bin/env python
"""
Description: get means and stds for each feature
Date: 13/11/19 
Email: anlijuncn@gmail.com
Writen by Lijun AN and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import pickle
import pandas as pd
import numpy as np
from lib.misc import load_feature


def get_fetures(data_path, test_fold=0):
    """
    get features list
    :param data_path:
    :param test_fold:
    :return:
    """
    # read train_long.csv
    train_long_csv_path = os.path.join(data_path, str(test_fold),
                                       'fold' + str(test_fold) + 'train_long.csv')
    train_long = pd.read_csv(train_long_csv_path)
    # save features list, 113 features for input
    column_names = list(train_long.columns)
    features = [e for e in column_names if e not in ('RID', 'curr_dx',
                                                     'curr_adas13',
                                                     'curr_ventricles',
                                                     'curr_icv')]
    # save it to a text file
    features_save_path = os.path.join(data_path, 'features.txt')
    with open(features_save_path, 'w') as f:
        for item in features:
            f.write("%s\n" % item)


def get_means_stds(data_path, test_fold):
    """
    get mean and std for 113 input features
    :param data_path:
    :param test_fold:
    :return:
    """
    # read train_long.csv
    print('Calculating means and stds for test fold', test_fold)
    train_long_csv_path = os.path.join(data_path, str(test_fold),
                                       'fold' + str(test_fold) +
                                       '_train_long_norm.csv')
    train_long = pd.read_csv(train_long_csv_path)
    # fill the time term with 999 for denoting missing value
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
                     'time_since_high_RAVLT_learn',
                     'time_since_mr_RAVLT_forget',
                     'time_since_low_RAVLT_forget',
                     'time_since_high_RAVLT_forget']
    for f in time_features:
        train_long[f].replace(np.nan, 999, inplace=True)
    # load features
    features_path = os.path.join(data_path, 'features.txt')
    features = load_feature(features_path)
    # features += ['curr_adas13', 'curr_ventricles', 'curr_icv']
    means_stds = {}
    means_stds_save_path = os.path.join(data_path, str(test_fold),
                                        'fold' + str(test_fold) +
                                        '_means_stds.pkl')
    no_means_stds_features = ['GENDER', 'MARRY', 'APOE']
    int_features = ['APOE', 'am_positive', 'mr_dx',
                    'best_dx', 'worst_dx']
    for f in features:
        if f not in no_means_stds_features:
            f_array = np.array(train_long[f])
            mean = np.nanmean(f_array)
            std = np.nanstd(f_array)
            if f in int_features:
                mean = round(mean)
            means_stds[f] = {}
            means_stds[f]['mean'] = mean
            means_stds[f]['std'] = std
    means_stds['am_positive']['mean'] = 0
    # note for curr_adas13, curr_ventricles, curr_icv
    # the way we calculate mean and std is different
    lables = ['curr_adas13', 'curr_ventricles']
    # we need to normalize ventricles with ICV
    # drop rows with duplicate RID and month_bl
    train_long.drop_duplicates(subset=['RID', 'Month_bl'], keep='first',
                               inplace=True)
    # print(train_long['curr_ventricles'])
    # train_long['curr_ventricles'] /= train_long['curr_icv']
    # print(train_long['curr_ventricles'])
    for f in lables:
        f_array = np.array(train_long[f])
        mean = np.nanmean(f_array)
        std = np.nanstd(f_array)
        means_stds[f] = {}
        means_stds[f]['mean'] = mean
        means_stds[f]['std'] = std
    # save it to a pkl file
    file = open(means_stds_save_path, 'wb')
    pickle.dump(means_stds, file)


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    for test_fold in range(10):
        get_means_stds(data_path, test_fold)

