#!/usr/bin/env python
"""
Description: Expand train_basic.csv into train_long.csv
Email:gzynus@gmail.com
Author: Zongyi Guo
"""
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool


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


def create_long_features(df_array, dates, jj, kk):
    # create
    past_mask = (dates <= dates[jj])
    past = df_array[past_mask]
    past_dates = dates[: jj + 1]
    # get non nan
    if np.sum(~np.isnan(past)) > 0:
        non_nan_past = past[~np.isnan(past)]
        non_nan_dates = past_dates[~np.isnan(past)]
        most_recent = non_nan_past[-1]
        # time_since_most_recent = month_between(dates[kk], non_nan_dates[-1])
        time_since_most_recent = dates[kk] - non_nan_dates[-1]
        lowest = np.min(non_nan_past)
        # time_since_lowest = month_between(dates[kk], non_nan_dates[non_nan_past == lowest][0])
        time_since_lowest = dates[kk] - non_nan_dates[non_nan_past == lowest][0]
        highest = np.max(non_nan_past)
        # time_since_highest = month_between(dates[kk], non_nan_dates[non_nan_past == highest][0])
        time_since_highest = dates[kk] - non_nan_dates[non_nan_past == highest][0]
        if np.sum(~np.isnan(past)) > 1:
            mr_change = non_nan_past[-1] - non_nan_past[-2]
            # time_diff = month_between(dates[kk], non_nan_dates[-2])
            time_diff = dates[kk] - non_nan_dates[-2]
            mr_change /= time_diff
        else:
            mr_change = np.nan

        return [most_recent, time_since_most_recent, mr_change, lowest,
                time_since_lowest, highest, time_since_highest]

    else:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


def create_dx_features(df_array, dates, jj, kk):
    # create a expanded dx features
    expanded = create_long_features(df_array, dates, jj, kk)
    # we need to remove mr_change and add milder, time_since_milder
    expanded.pop(2)  # remove mr_change
    past_mask = dates <= dates[jj]
    past = df_array[past_mask]
    past_dates = dates[: jj + 1]
    non_nan_past = past[~np.isnan(past)]
    non_nan_dates = past_dates[~np.isnan(past)]

    most_recent = expanded[0]
    tmp = non_nan_past[non_nan_past < most_recent]
    tmp_dates = non_nan_dates[non_nan_past < most_recent]
    if len(tmp) == 0:
        expanded.append(0)
        expanded.append(999)
    else:
        expanded.append(1)
        # expanded.append(month_between(dates[kk], tmp_dates[-1]))
        expanded.append(dates[kk] - tmp_dates[-1])

    return expanded


def feature_names(feature):
    """
    generate a list of expanded feature names
    :param feature:
    :return:
    """
    if feature == 'DX':
        return ['mr_dx', 'time_since_mr_dx', 'best_dx', 'time_since_best_dx',
                'worst_dx', 'time_since_worst_dx', 'milder',
                'time_since_midler']
    else:
        return ['mr_' + feature, 'time_since_mr_' + feature,
                'mr_change_' + feature, 'low_' + feature, 'time_since_low_' +
                feature, 'high_' + feature, 'time_since_high_' + feature]


def create_long_train_csv(data_path, test_fold):
    """
    create the train_long.csv
    1. expand the features
    1.1 longitudinal features into 7 categories
    1.2 Dx features into 8 categories
    2. increase instances by using M*(M-1)/2
    """
    time_begin = time.time()
    # for test_fold in range(10):
    print('Processing for fold', test_fold)
    # read {test_fold}_train_basic.csv
    train_basic_path = os.path.join(data_path, str(test_fold),
                                    'fold' + str(test_fold) +
                                    '_train_basic.csv')
    tb_df = pd.read_csv(train_basic_path)
    # create a train_long dataframe
    column_names = ['RID', 'D1', 'D2', 'APOE', 'GENDER', 'EDUC', 'MARRY',
                    'am_positive', 'curr_age', 'Month_bl', 'curr_dx',
                    'curr_adas13', 'curr_ventricles', 'curr_icv']
    # adding longitudinal features name
    column_names += feature_names('DX')
    column_names += feature_names('ADAS13')
    column_names += feature_names('Ventricles')
    column_names += feature_names('Fusiform')
    column_names += feature_names('WholeBrain')
    column_names += feature_names('Hippocampus')
    column_names += feature_names('MidTemp')
    column_names += feature_names('ICV')
    column_names += feature_names('MMSE')
    column_names += feature_names('CDRSB')
    column_names += feature_names('FDG')
    column_names += feature_names('ADAS11')
    column_names += feature_names('RAVLT_im')
    column_names += feature_names('RAVLT_learn')
    column_names += feature_names('RAVLT_forget')
    train_long = pd.DataFrame(columns=column_names)
    # get train subjects
    folds = np.load(os.path.join(data_path, 'folds.npy'))
    val_fold = (test_fold + 1) % len(folds)
    train_folds = [
        i for i in range(len(folds)) if (i != test_fold and i != val_fold)
    ]

    train_subjs = np.concatenate([folds[i] for i in train_folds], axis=0)
    count = 0
    for subj in train_subjs:
        count += 1
        print(count, time.time() - time_begin)
        # get sorted months for each subject
        subj_mask = (tb_df.RID == subj)
        # sort datframe according to 'EXAMDATE'
        subj_data = tb_df[subj_mask].sort_values('EXAMDATE', ascending=True)
        months = np.sort(tb_df.Month_bl[subj_mask])
        nb_visits = months.shape[0]
        # # first we need to expand instances
        if nb_visits < 2:
            continue
        for kk in range(1, nb_visits):
            for jj in range(0, kk):
                date = np.array(np.sort(tb_df.EXAMDATE[subj_mask]))[kk]
                date_mask = (subj_data.EXAMDATE == date)
                # in this way, we could get M*(M-1)/2 instances
                # firstly, fill in the cross sectional variable
                row = []
                # append cross sectional variables
                row.append(subj)  # RID
                row.append(np.array(subj_data.loc[date_mask, 'D1'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'D2'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'APOE4'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'PTGENDER'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'PTEDUCAT'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'PTMARRY'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'am_positive'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'curr_age'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'Month_bl'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'curr_dx_numeric'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'ADAS13'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'Ventricles'])[0])
                row.append(np.array(subj_data.loc[date_mask, 'ICV'])[0])
                # append dx features, we need to expand into 8 categories
                row += create_dx_features(np.array(subj_data['curr_dx_numeric']), months, jj, kk)
                # append longitudinal features, mapped into 7 categories
                row += create_long_features(np.array(subj_data['ADAS13']), months, jj, kk)
                row += create_long_features(np.array(subj_data['Ventricles']) / np.array(subj_data['ICV']),  months, jj, kk)
                row += create_long_features(np.array(subj_data['Fusiform']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['WholeBrain']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['Hippocampus']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['MidTemp']) / np.array(subj_data['ICV']), months, jj, kk)
                row += create_long_features(np.array(subj_data['ICV']),  months, jj, kk)
                row += create_long_features(np.array(subj_data['MMSE']), months, jj, kk)
                row += create_long_features(np.array(subj_data['CDRSB']), months, jj, kk)
                row += create_long_features(np.array(subj_data['FDG']), months,  jj, kk)
                row += create_long_features(np.array(subj_data['ADAS11']),  months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_immediate']), months, jj, kk)
                row += create_long_features(np.array(subj_data['RAVLT_learning']),  months, jj, kk)
                row += create_long_features( np.array(subj_data['RAVLT_perc_forgetting']), months, jj, kk)
                # append it to train_long
                train_long = train_long.append(pd.DataFrame([row], columns=column_names),
                                               ignore_index=True)
    # save train_long
    train_long.to_csv(os.path.join(data_path, str(test_fold), 'fold' + str(
        test_fold) + '_train_long.csv'),
                      sep=',', index=False)


if __name__ == '__main__':
    # get root path
    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(root_path, 'data')
    data_path_list = []
    test_fold_list = []
    for test_fold in range(10):
        data_path_list.append(data_path)
        test_fold_list.append(test_fold)
    cores = multiprocessing.cpu_count()
    pool = Pool(processes=cores)
    pool.map(create_long_train_csv, data_path_list, test_fold_list)
    # create_long_train_csv(data_path, 0)
