#!/usr/bin/env python
"""
Description: Z normalization the input features and curr_adas13,
curr_ventricles, curr_icv
Date: 14/11/19 
Email: anlijuncn@gmail.com
Writen by Lijun AN and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import pickle
import pandas as pd
import numpy as np


def z_normalize(data_path, test_fold):
    """
    Z normalize the input features and curr_adas13,
    curr_ventricles, curr_icv
    :param data_path:
    :param test_fold:
    :return:
    """
    print('z normalize for test fold', test_fold)
    # read filled train_long, val_pred, test_pred
    filled_train_long_path = os.path.join(data_path, str(test_fold),
                                          'fold' + str(
                                              test_fold) +
                                          '_train_long_filled.csv')
    filled_train_long = pd.read_csv(filled_train_long_path)
    filled_val_pred_path = os.path.join(data_path, str(test_fold),
                                        'fold' + str(
                                            test_fold) + '_val_pred_filled.csv')
    filled_val_pred = pd.read_csv(filled_val_pred_path)
    filled_test_pred_path = os.path.join(data_path, str(test_fold),
                                         'fold' + str(
                                             test_fold) +
                                         '_test_pred_filled.csv')
    filled_test_pred = pd.read_csv(filled_test_pred_path)
    # save path
    z_train_long_path = os.path.join(data_path, str(test_fold),
                                     'fold' + str(test_fold) +
                                     '_train_long_znorm.csv')
    z_val_pred_path = os.path.join(data_path, str(test_fold),
                                   'fold' + str(test_fold) +
                                   '_val_pred_znorm.csv')
    z_test_pred_path = os.path.join(data_path, str(test_fold),
                                    'fold' + str(test_fold) +
                                    '_test_pred_znorm.csv')
    # means and stds
    means_stds_path = os.path.join(data_path, str(test_fold),
                                   'fold' + str(test_fold) + '_means_stds.pkl')
    with open(means_stds_path, 'rb') as file:
        means_stds = pickle.load(file)
    no_need_znormalize = ['APOE', 'best_dx', 'mr_dx', 'worst_dx', 'am_positive']
    lables = ['curr_adas13', 'curr_ventricles']
    for f in means_stds.keys():
        # no need for APOE, it is categorial variable
        if f not in no_need_znormalize:
            mean = means_stds[f]['mean']
            std = means_stds[f]['std']
            # z normalize
            filled_train_long[f] = (filled_train_long[f] - mean) / std
            if f not in lables:
                filled_val_pred[f] = (filled_val_pred[f] - mean) / std
                filled_test_pred[f] = (filled_test_pred[f] - mean) / std
    # save it to a csv file
    filled_train_long.to_csv(z_train_long_path, sep=',', index=False)
    filled_val_pred.to_csv(z_val_pred_path, sep=',', index=False)
    filled_test_pred.to_csv(z_test_pred_path, sep=',', index=False)



if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    for test_fold in range(10):
        z_normalize(data_path, test_fold)