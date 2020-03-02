#!/usr/bin/env python
"""
Description:
Check whether RID, D1, D2, GENDER, MARRY, curr_age, Month_bl is missing
Date: 14/11/19 
Email: anlijuncn@gmail.com
Writen by Lijun AN and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import pandas as pd
import numpy as np


def check_demog_features(data_path, test_fold):
    """
    check whether demog featrues are missing
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
    demog_features = ['GENDER', 'EDUC', 'MARRY', 'curr_age', 'Month_bl', 'APOE']
    train_long = pd.read_csv(train_long_path, usecols=demog_features)
    val_pred = pd.read_csv(val_pred_path, usecols=demog_features)
    test_pred = pd.read_csv(test_pred_path, usecols=demog_features)
    print('****train_long', np.sum(np.isnan(np.array(train_long))))
    print('****val_pred', np.sum(np.isnan(np.array(val_pred))))
    print('****test_fold', np.sum(np.isnan(np.array(test_pred))))


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    for test_fold in range(10):
        check_demog_features(data_path, test_fold)