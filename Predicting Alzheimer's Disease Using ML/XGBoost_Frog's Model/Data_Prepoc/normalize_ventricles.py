#!/usr/bin/env python
"""
Description: We need to normalize curr_ventricles with curr_icv
Date: 17/11/19 
Email: anlijuncn@gmail.com
Writen by Lijun AN and CBIG under MIT license:
https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md
"""
import os
import pandas as pd
import numpy as np


def normalize_vents(data_path):
    """
    normalize curr_ventricles with curr_icv in fold{fold}_train_long.csv
    After normalization, we only has curr_ventricles
    :param data_path:
    :return:
    """
    for test_fold in range(10):
        # for 10 folds
        print('Normalize ventricles for test fold', test_fold)
        train_long_path = os.path.join(data_path, str(test_fold),
                                       'fold' + str(test_fold) +
                                       '_train_long.csv')
        # read train_long.csv
        train_long = pd.read_csv(train_long_path)
        # normalize ventricles by ICV
        train_long['curr_ventricles'] = train_long['curr_ventricles'] /\
                                        train_long['curr_icv']
        # drop curr_icv
        train_long.drop(columns=['curr_icv'], inplace=True)
        # save as csv file
        save_path = os.path.join(data_path, str(test_fold), 'fold' + str(
            test_fold) + '_train_long_norm.csv')
        train_long.to_csv(save_path, sep=',', index=False)


if __name__ == '__main__':
    # get root path
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    normalize_vents(data_path)