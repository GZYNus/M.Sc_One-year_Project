#!/usr/bin/env python
"""
Description: Create pred truth csv files according to label and time windows
Email:gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


def month_between(end, start):
    """ Get duration (in months) between *end* and *start* dates """
    # assert end >= start
    end = datetime.strptime(end, "%Y-%m-%d")
    start = datetime.strptime(start, "%Y-%m-%d")
    diff = relativedelta(end, start)
    months = 12 * diff.years + diff.months
    to_next = relativedelta(end + relativedelta(months=1, days=-diff.days),
                            end).days
    to_prev = diff.days
    return months + (to_next < to_prev)


def create_pred_truth(data_path):
    """
    create prediction truth according to label and time windiows
    :param data_path:
    :return:
    """
    labels = ['dx', 'adas13', 'ventricles']
    time_window = {}
    time_window['dx'] = [0, 8, 15, 27, 39, 60, 1000]
    time_window['adas13'] = [0, 9, 15, 27, 39, 54, 1000]
    time_window['ventricles'] = [0, 9, 15, 30, 1000]
    # load folds.npy
    folds = np.load(os.path.join(data_path, 'folds.npy'))
    for test_fold in range(10):
        print('processing for test fold', test_fold)
        val_fold = (test_fold + 1) % len(folds)
        val_subjs = folds[val_fold]
        # read train basic csv to get last date in first half
        basic_df_path = os.path.join(data_path, str(test_fold),
                                     'fold' + str(
                                         test_fold) + '_train_basic.csv')
        basic_df = pd.read_csv(basic_df_path)
        # get last input date for each val subject
        last_input_date = {}
        last_input_date['dx'] = {}
        last_input_date['adas13'] = {}
        last_input_date['ventricles'] = {}
        for val_subj in val_subjs:
            subj_mask = (basic_df.RID == val_subj)
            # for dx
            subj_data_dx = basic_df.loc[subj_mask, ['DX', 'EXAMDATE']]
            # drop NaN dx
            subj_data_dx.dropna(axis=0, how='any', inplace=True)
            if subj_data_dx.shape[0] == 0:
                last_date_dx = 9999
            else:
                # get last input date
                last_date_dx = np.sort(subj_data_dx.EXAMDATE)[-1]
            last_input_date['dx'][val_subj] = last_date_dx

            # for adas13
            subj_data_adas13 = basic_df.loc[subj_mask, ['ADAS13', 'EXAMDATE']]
            # drop NaN adas13
            subj_data_adas13.dropna(axis=0, how='any', inplace=True)
            if subj_data_adas13.shape[0] == 0:
                last_date_adas13 = 9999
            else:
                last_date_adas13 = np.sort(subj_data_adas13.EXAMDATE)[-1]
            last_input_date['adas13'][val_subj] = last_date_adas13

            # for ventricles
            subj_data_vents = basic_df.loc[subj_mask, ['Ventricles', 'ICV',
                                                       'EXAMDATE']]
            # drop NaN vents
            subj_data_vents.dropna(axis=0, how='any', inplace=True)
            if subj_data_vents.shape[0] == 0:
                last_date_vents = 9999
            else:
                last_date_vents = np.sort(subj_data_vents.EXAMDATE)[-1]
            last_input_date['ventricles'][val_subj] = last_date_vents

        # read val.csv
        val_path = os.path.join(data_path, str(test_fold),
                                'fold' + str(test_fold) + '_val.csv')
        val_df = pd.read_csv(val_path)
        columns = ['RID', 'CognitiveAssessmentDate','Diagnosis', 'ADAS13',
                        'ScanDate', 'Ventricles']
        val_df = val_df[columns]
        nb_rows = val_df.shape[0]
        for label in labels:
            print('****processing lable', label)
            # get upper bound and lower bound for time window
            for upper_idx in range(1, len(time_window[label])):
                print('--------processing for window', upper_idx)
                upper = time_window[label][upper_idx]
                lower = time_window[label][upper_idx - 1]
                # create corresponding data frame
                df = pd.DataFrame(columns=['RID', 'CognitiveAssessmentDate',
                                           'Diagnosis', 'ADAS13', 'ScanDate',
                                           'Ventricles'])
                # we need to read val_df row by row
                for i in range(nb_rows):
                    row = val_df.iloc[i]
                    rid = row.iloc[0]
                    last_date = last_input_date[label][rid]
                    if last_date == 9999:
                        if upper_idx == len(time_window[label]) - 1:
                            df = df.append(row, ignore_index=True)

                    else:
                        date = row.iloc[1]
                        # get time_since_mr_xxx
                        months = month_between(date, last_date)
                        if lower <= months < upper:
                            df = df.append(row, ignore_index=True)
                print('##############shape', df.shape)
                # get save path
                save_path = os.path.join(data_path, str(test_fold),
                                         'fold' + str(test_fold) + '_val_gt_'
                                         + label + '_' + str(upper_idx) +
                                         '.csv')
                df.to_csv(save_path, sep=',', index=False)


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    create_pred_truth(data_path)
