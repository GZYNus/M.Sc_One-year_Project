#!/usr/bin/env python
"""
Description:
Create train, val and test set for feeding the model, also taking time window into consideration
Email: gzynus@gmail.com
Author: Zongyi
"""
import os
import pickle
import pandas as pd
import numpy as np
from lib.misc import load_feature


def transfer_train_to_matrix(data_path):
    """
    transformer z-normalized csv file,
    we need to create npy according to time window
    Very careful for the order
    For clinic, adas13 and ventricles, we need to create according to time
    window
    For Clinic:
    Time window: 0-8, 8-15, 27-39, 39-60, 60-500
    For ADAS13:
    Time window: 0-9, 9-15, 15-27, 27-39, 39-54, 54-500
    For Ventricle:
    Time window: 0-9, 9-15, 15-30, 30-500
    :param data_path:
    :return:
    """
    labels = ['curr_dx', 'curr_adas13', 'curr_ventricles']
    # in case of consistency
    features = load_feature(os.path.join(data_path, 'features.txt'))
    columns = ['curr_dx', 'curr_adas13', 'curr_ventricles'] + features
    time_since_list = ['time_since_mr_dx', 'time_since_mr_ADAS13',
                       'time_since_mr_Ventricles']
    for test_fold in range(10):
        print('processing test fold', test_fold)
        # we need to z-normalize time terms as well
        means_stds_path = os.path.join(data_path, str(test_fold),
                                       'fold' + str(test_fold) + '_means_stds.pkl')
        with open(means_stds_path, 'rb') as file:
            means_stds = pickle.load(file)
        time_window = {}
        time_window['curr_dx'] = [0, 8, 15, 27, 39, 60, 1000]
        # print(time_window['curr_dx'])
        time_window['curr_dx'] = (time_window['curr_dx'] - means_stds['time_since_mr_dx']['mean']) / means_stds['time_since_mr_dx']['std']
        # print(time_window['curr_dx'])
        time_window['curr_adas13'] = [0, 9, 15, 27, 39, 54, 1000]
        # print(time_window['curr_adas13'])
        time_window['curr_adas13'] = (time_window['curr_adas13'] - means_stds['time_since_mr_ADAS13']['mean']) / means_stds['time_since_mr_ADAS13']['std']
        # print(time_window['curr_adas13'])
        time_window['curr_ventricles'] = [0, 9, 15, 30, 1000]
        # print(time_window['curr_ventricles'])
        time_window['curr_ventricles'] = (time_window['curr_ventricles'] - means_stds['time_since_mr_Ventricles']['mean']) / means_stds['time_since_mr_Ventricles']['std']
        # print(time_window['curr_ventricles'])
        # get z normalized train long csv
        z_norm_csv_path = os.path.join(data_path, str(test_fold),
                                       'fold' + str(test_fold) +
                                       '_train_long_znorm.csv')
        z_norm_csv = pd.read_csv(z_norm_csv_path, usecols=columns)
        z_norm_csv = z_norm_csv[columns]  # change coulumn order
        total_rows = z_norm_csv.shape[0]
        for label in labels:
            print('****processing lable', label)
            time_since = time_since_list[labels.index(label)]
            month_bl_idx = columns.index(time_since) - 3
            label_array = np.array(z_norm_csv[label])
            input_array = np.array(z_norm_csv)
            input_array = input_array[:, 3:]  # please do not use magic number
            for upper_idx in range(1, len(time_window[label])):
                upper = time_window[label][upper_idx]
                lower = time_window[label][upper_idx - 1]
                mask = (input_array[:, month_bl_idx] >= lower) & (input_array[:, month_bl_idx] < upper)
                masked_label_array = label_array[mask]
                masked_label_array = np.reshape(masked_label_array,
                                                (masked_label_array.shape[0], 1))
                masked_input_array = input_array[mask, :]
                # concat masked label and masked input
                masked_array = np.concatenate((masked_label_array,
                                               masked_input_array), axis=1)
                save_path = os.path.join(data_path, str(test_fold),
                                         'fold' + str(test_fold) + '_train_'
                                         + label[5:] + '_' + str(upper_idx) +
                                         '.npy')
                np.save(save_path, masked_array)


def transfer_pred_to_matrix(data_path):
    """
    transformer z-normalized test/val pred csv file,
    we need to create npy according to time window
    Very careful for the order
    For clinic, adas13 and ventricles, we need to create according to time
    window
    For Clinic:
    Time window: 0-8, 8-15, 27-39, 39-60, 60-500
    For ADAS13:
    Time window: 0-9, 9-15, 15-27, 27-39, 39-54, 54-500
    For Ventricle:
    Time window: 0-9, 9-15, 15-30, 30-500
    :param data_path:
    :return:
    """
    # note maybe not all time window has to pred
    features = load_feature(os.path.join(data_path, 'features.txt'))
    columns = ['RID', 'Forecast_month'] + features
    time_since_list = ['time_since_mr_dx', 'time_since_mr_ADAS13',
                       'time_since_mr_Ventricles']
    labels = ['curr_dx', 'curr_adas13', 'curr_ventricles']
    for test_fold in range(10):
        # means and std
        print('processing test fold', test_fold)
        # we need to z-normalize time terms as well
        means_stds_path = os.path.join(data_path, str(test_fold),
                                       'fold' + str(
                                           test_fold) + '_means_stds.pkl')
        with open(means_stds_path, 'rb') as file:
            means_stds = pickle.load(file)
        time_window = {}
        time_window['curr_dx'] = [0, 8, 15, 27, 39, 60, 1000]
        time_window['curr_dx'] = (time_window['curr_dx'] -
                                  means_stds['time_since_mr_dx']['mean']) / \
                                 means_stds['time_since_mr_dx']['std']
        time_window['curr_adas13'] = [0, 9, 15, 27, 39, 54, 1000]
        time_window['curr_adas13'] = (time_window['curr_adas13'] -
                                      means_stds['time_since_mr_ADAS13']['mean']) / \
                                     means_stds['time_since_mr_ADAS13']['std']
        time_window['curr_ventricles'] = [0, 9, 15, 30, 1000]
        time_window['curr_ventricles'] = (time_window['curr_ventricles'] -
                                          means_stds['time_since_mr_Ventricles']['mean']) / \
                                         means_stds['time_since_mr_Ventricles']['std']
        # pred z-normalized val_pred and test_pred
        # get z normalized val pred and test pred csv
        val_norm_csv_path = os.path.join(data_path, str(test_fold),
                                         'fold' + str(test_fold) +
                                         '_val_pred_znorm.csv')
        val_norm_csv = pd.read_csv(val_norm_csv_path, usecols=columns)
        val_norm_csv = val_norm_csv[columns]

        test_norm_csv_path = os.path.join(data_path, str(test_fold),
                                          'fold' + str(test_fold) +
                                          '_test_pred_znorm.csv')
        test_norm_csv = pd.read_csv(test_norm_csv_path, usecols=columns)
        test_norm_csv = test_norm_csv[columns]
        for label in labels:
            print('****processing lable', label)
            time_since = time_since_list[labels.index(label)]
            month_bl_idx = columns.index(time_since)
            val_array = np.array(val_norm_csv)
            test_array = np.array(test_norm_csv)
            for upper_idx in range(1, len(time_window[label])):
                upper = time_window[label][upper_idx]
                lower = time_window[label][upper_idx - 1]
                val_mask = (val_array[:, month_bl_idx] >= lower) & (
                        val_array[:, month_bl_idx] < upper)
                test_mask = (test_array[:, month_bl_idx] >= lower) & (
                        test_array[:, month_bl_idx] < upper)
                masked_val_array = val_array[val_mask]
                masked_test_array = test_array[test_mask]
                # save it as npy file
                val_path = os.path.join(data_path, str(test_fold),
                                        'fold' + str(test_fold) + '_val_' +
                                        label[5:] + '_' + str(upper_idx) +
                                        '.npy')
                test_path = os.path.join(data_path, str(test_fold),
                                         'fold' + str(test_fold) + '_test_' +
                                         label[5:] + '_' + str(upper_idx) +
                                         '.npy')
                np.save(val_path, masked_val_array)
                np.save(test_path, masked_test_array)


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    transfer_pred_to_matrix(data_path)