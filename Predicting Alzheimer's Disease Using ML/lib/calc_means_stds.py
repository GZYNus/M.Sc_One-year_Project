#!/usr/bin/env python
"""
Description: calculate and save the mean and standard deviation
Email: gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import pickle
import numpy as np
from lib.load_data import loading_original_data
from lib.load_data import loading_fields_list


def means_stds_calculation(data_path, flag):
    """
    calculate means and stds
    """
    assert flag in ('train', 'split_train'), \
        'Wrong Flag, only "train" or "split_train" is acceptable'
    # loading nointerp data
    data = loading_original_data(data_path, flag)
    means = {}
    stds = {}
    ven_icv_list = []
    # loading fields_list
    fields_list = loading_fields_list(data_path)
    for feature in range(2, 24):
        means[fields_list[feature]] = None
        stds[fields_list[feature]] = None
        data_list = []
        for key in sorted(data.keys()):
            for time in range(data[key]['truth'].shape[0]):
                if ~np.isnan(data[key]['truth'][time, feature]):
                    data_list.append(data[key]['truth'][time, feature])
                # for ventricles/ICV
                if ~np.isnan(data[key]['truth'][time, fields_list.index('ICV')]) \
                        and ~np.isnan(data[key]['truth'][time, fields_list.index('Ventricles')]):
                    ven_icv = data[key]['truth'][time, fields_list.index('Ventricles')] \
                             / data[key]['truth'][time, fields_list.index('ICV')]
                    ven_icv_list.append(ven_icv)
        means[fields_list[feature]] = np.mean(data_list)
        stds[fields_list[feature]] = np.std(data_list)
    means['venICV'] = np.mean(ven_icv_list)
    stds['venICV'] = np.std(ven_icv_list)
    # save means and stds
    means_save_path = os.path.join(data_path, 'lijun_means_' + flag + '.pkl')
    file = open(means_save_path, 'wb')
    pickle.dump(means, file)
    file.close()
    stds_save_path = os.path.join(data_path, 'lijun_stds_' + flag + '.pkl')
    file = open(stds_save_path, 'wb')
    pickle.dump(stds, file)
    file.close()
    print('Finish mean and std calculation!')


if __name__ == '__main__':
    # root path for the project
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    means_stds_calculation(data_path, flag='train')
    means_stds_calculation(data_path, flag='split_train')